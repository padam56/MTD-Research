"""
train.py — Train DQN agent for MTD-Playground (offline mode)
=============================================================
Trains a Double DQN agent on the simulated MTD-Playground environment.

RL Method:
    Double DQN — van Hasselt, Guez & Silver (2016)
    "Deep Reinforcement Learning with Double Q-learning", AAAI 2016.

MTD Formulation:
    Eghtesad, Vorobeychik & Laszka (2020)
    "Adversarial Deep RL Based Adaptive Moving Target Defense", GameSec 2020.

Usage:
    python train.py                     # Train with defaults (50k steps)
    python train.py --timesteps 100000  # Train longer
    python train.py --seed 42           # Reproducible run
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Train DQN for MTD-Playground")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="../models")
    args = parser.parse_args()

    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("Run: pip install stable-baselines3[extra]")
        sys.exit(1)

    from mtd_env import MTDPlaygroundEnv

    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Creating environment (seed={args.seed})...")
    env = MTDPlaygroundEnv(seed=args.seed)
    eval_env = MTDPlaygroundEnv(seed=args.seed + 1000)

    print(f"Training Double DQN for {args.timesteps} timesteps...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        tau=0.005,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [256, 128]},
        seed=args.seed,
        # tensorboard_log=os.path.join(args.model_dir, "tb_logs"),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=os.path.join(args.model_dir, "eval_logs"),
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    save_path = os.path.join(args.model_dir, "dqn_mtd_final")
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
