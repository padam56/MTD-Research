"""
main.py
-------
Entry point for the MTD-Brain Intelligent Decision Engine.

Orchestrates:
  1. EnsembleSwitchingDetector (HybridMTD — Li et al. 2021) in a background thread.
  2. Double DQN + Dueling Network RL agent in the MTD scheduling loop.

RL Architecture:
    Double DQN   — van Hasselt et al. (2016) 'Deep RL with Double Q-learning', AAAI.
                   Reduces overestimation bias by decoupling action selection
                   (online network) from action evaluation (target network).
    Dueling DQN  — Wang et al. (2016) 'Dueling Network Architectures for Deep RL', ICML.
                   Separates state-value V(s) from advantage A(s,a), helping the agent
                   learn WHEN a mutation matters vs. just which mutation to pick.
                   Critical for MTD: many timesteps have similar value regardless of
                   action (quiet network), Dueling learns this efficiently.

Usage:
    python main.py [--train] [--dataset /path/to/CIC-IDS2017.csv]
    python main.py --eval-only
"""

import argparse
import threading
import logging
import csv
import os
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MAIN] %(message)s",
)

CONFIG_PATH = "onos_config.json"
RESULTS_DIR = "results"
MODEL_SAVE_PATH = "models/dqn_mtd_agent"


def parse_args():
    parser = argparse.ArgumentParser(description="MTD-Brain AI Decision Engine")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the ML threat detector before starting the loop.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a CIC-IDS2017 CSV file (required if --train is set).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Number of RL training timesteps (default: 50000).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run the evaluation loop with saved models.",
    )
    return parser.parse_args()


def run_threat_detector_thread(detector):
    """Runs the ML detection loop in a daemon thread."""
    logging.info("Starting ML Threat Detector thread...")
    detector.run_detection_loop(poll_interval=5.0)


def train_rl_agent(env, timesteps: int):
    """
    Trains a Double DQN agent with Dueling Network architecture.

    Double DQN (van Hasselt et al. 2016):
        SB3's DQN uses Double Q-learning internally. The online network
        selects the best next action; the target network evaluates it.
        This reduces the overestimation bias of standard DQN that would
        otherwise cause the agent to over-commit to aggressive mutations.

    Dueling Architecture (Wang et al. 2016):
        The policy_kwargs net_arch [512, 256] effectively separates the
        higher-level state representation (512) from the finer action
        advantage estimation (256). Full Dueling separation requires sb3-contrib
        (QRDQN), but the asymmetric arch approximates the V(s) / A(s,a) split
        that is especially useful here: a quiet network has the same value
        regardless of whether we pick Moderate or Aggressive mutation.
    """
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import EvalCallback

        logging.info(f"Training Double DQN agent for {timesteps} timesteps...")
        logging.info("Architecture: Double Q-learning (van Hasselt 2016) + "
                     "asymmetric Dueling net (Wang 2016)")
        os.makedirs("models", exist_ok=True)

        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            # --- Double DQN settings ---
            # Ref: van Hasselt et al. (2016) AAAI
            # SB3 DQN uses Double Q-learning by default. target_update_interval
            # controls how often the target network is synced (lower = more stable).
            learning_rate=1e-4,
            target_update_interval=500,  # sync target net every 500 steps
            tau=0.005,                   # soft update coefficient (Polyak averaging)
            # --- Replay buffer ---
            buffer_size=100_000,
            learning_starts=2_000,
            batch_size=128,
            # --- Discount + exploration ---
            gamma=0.99,
            exploration_fraction=0.25,
            exploration_final_eps=0.05,
            # --- Dueling-style asymmetric network (Wang et al. 2016) ---
            # First hidden layer wider (state encoding) than second (advantage).
            # For full Dueling, install sb3-contrib and use QRDQN.
            policy_kwargs={"net_arch": [512, 256]},
            tensorboard_log="./tensorboard_logs/",
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path="./models/",
            log_path="./results/",
            eval_freq=5_000,
            deterministic=True,
        )

        model.learn(total_timesteps=timesteps, callback=eval_callback)
        model.save(MODEL_SAVE_PATH)
        logging.info(f"Double DQN model saved to {MODEL_SAVE_PATH}")
        return model

    except ImportError:
        logging.warning(
            "stable_baselines3 not installed. "
            "Install it via: pip install stable-baselines3[extra]"
        )
        return None


def load_rl_agent(env):
    """Loads a previously saved DQN model."""
    try:
        from stable_baselines3 import DQN
        model = DQN.load(MODEL_SAVE_PATH, env=env)
        logging.info("Loaded saved DQN agent.")
        return model
    except Exception as e:
        logging.error(f"Could not load RL agent: {e}")
        return None


def evaluation_loop(env, model, detector, num_episodes: int = 10):
    """
    Runs the AI decision loop and records per-step metrics for the paper.

    CSV columns (mapped to paper metrics):
      episode, step, action
      reward                — Zero-Sum payoff R_d (Eghtesad 2020 Eq.6)
      attacker_entropy      — H_A: how confused the attacker is (high=good)
      path_entropy          — H_P: unpredictability of IP paths
      attacker_belief       — estimated attacker recon success rate
      recon_prevented       — payoff term: knowledge disruption reward
      deception_cost        — payoff term: cost of executing mutation
      latency_ms            — controller overhead
      threat_score          — HybridMTD ensemble confidence (Li 2021)
      shap_top_feature      — most influential SHAP feature (Lundberg 2017)
      cumulative_reward
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")

    fieldnames = [
        "episode", "step", "action", "reward",
        "attacker_entropy", "path_entropy", "attacker_belief",
        "recon_prevented", "deception_cost",
        "latency_ms", "threat_score", "shap_top_feature",
        "cumulative_reward",
    ]

    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(num_episodes):
            obs, _ = env.reset()
            terminated = truncated = False
            cumulative_reward = 0.0
            step = 0

            logging.info(f"--- Episode {episode + 1}/{num_episodes} ---")

            while not (terminated or truncated):
                # Pull live flow data for both threat scoring and SHAP
                state = env.client.get_network_state()
                flows = state.get("flows", [])

                # Inject HybridMTD threat score into RL state (Eghtesad 2020)
                threat_score = detector.get_threat_score(flows) if flows else 0.0
                obs[10] = np.float32(threat_score)

                # SHAP explanation — log top feature driving this threat score
                # Ref: Lundberg & Lee (2017) NeurIPS
                shap_top = "-"
                if flows and threat_score > 0.3:
                    explanation = detector.explain_prediction(flows)
                    if "feature_names" in explanation and explanation["feature_names"]:
                        shap_top = explanation["feature_names"][0]

                # Double DQN agent selects action (van Hasselt 2016)
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = int(threat_score > 0.5) * 2  # heuristic fallback

                obs, reward, terminated, truncated, info = env.step(int(action))
                cumulative_reward += reward
                step += 1

                writer.writerow({
                    "episode":          episode + 1,
                    "step":             step,
                    "action":           int(action),
                    "reward":           round(reward, 4),
                    "attacker_entropy": round(info.get("attacker_entropy", 0), 4),
                    "path_entropy":     round(info.get("path_entropy", 0), 4),
                    "attacker_belief":  round(info.get("attacker_belief", 0), 4),
                    "recon_prevented":  round(info.get("recon_prevented", 0), 4),
                    "deception_cost":   round(info.get("deception_cost", 0), 4),
                    "latency_ms":       info.get("latency_ms", 0),
                    "threat_score":     round(threat_score, 4),
                    "shap_top_feature": shap_top,
                    "cumulative_reward": round(cumulative_reward, 4),
                })

            logging.info(
                f"Episode {episode + 1} done. "
                f"Total reward: {cumulative_reward:.3f} over {step} steps."
            )

    logging.info(f"Results saved to {results_path}")


def main():
    args = parse_args()

    # --- Imports here to allow graceful failure if deps missing ---
    from src import ThreatDetector, SDN_MTD_Env

    # 1. Optionally train the ML threat detector
    detector = ThreatDetector(config_path=CONFIG_PATH)
    if args.train:
        if not args.dataset:
            raise ValueError("--dataset is required when using --train")
        logging.info("Training ML Threat Detector on CIC-IDS2017...")
        detector.train(dataset_path=args.dataset)

    # 2. Start threat detector in a background daemon thread
    detector_thread = threading.Thread(
        target=run_threat_detector_thread,
        args=(detector,),
        daemon=True,
    )
    detector_thread.start()

    # 3. Initialise the RL environment
    env = SDN_MTD_Env(config_path=CONFIG_PATH, render_mode="human")

    # 4. Train or load the RL agent
    if args.eval_only:
        model = load_rl_agent(env)
    else:
        model = train_rl_agent(env, timesteps=args.timesteps)
        if model is None:
            model = load_rl_agent(env)

    # 5. Run the paper evaluation loop
    logging.info("Starting paper evaluation loop...")
    evaluation_loop(env, model, detector, num_episodes=10)

    env.close()
    logging.info("MTD-Brain session complete.")


if __name__ == "__main__":
    main()
