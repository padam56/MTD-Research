"""
run_offline.py
--------------
Run the complete MTD-Brain pipeline with NO live ONOS controller.

Steps performed automatically:
    1. Generate a synthetic CIC-IDS2017-format dataset (unless one already exists)
    2. Train the HybridMTD Ensemble Switching ML detector on it
    3. Patch the RL environment to use MockONOSClient instead of ONOSClient
    4. Train the Double DQN + Dueling RL agent
    5. Run the evaluation loop and save results/evaluation_results.csv
    6. Print a summary of results

Usage:
    python scripts/run_offline.py
    python scripts/run_offline.py --timesteps 20000 --episodes 5 --fast
"""

import argparse
import logging
import os
import sys
import csv
import threading
import numpy as np

# ── Make sure we can import from the project root ─────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # ensure relative paths (models/, results/) resolve correctly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OFFLINE] %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="MTD offline demo — no ONOS required",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Path 2 — synthetic only (default):
  python scripts/run_offline.py --timesteps 50000 --episodes 10

Path 3 — hybrid (CIC-IDS2017 for ML, synthetic for RL):
  python scripts/run_offline.py --dataset data/cic-ids2017/Friday-DDos.csv --timesteps 50000
  Drop any CIC-IDS2017 day CSV into data/cic-ids2017/ and point --dataset at it.
"""
    )
    p.add_argument("--timesteps", type=int, default=10_000,
                   help="RL training timesteps (default: 10000; use 50000+ for paper)")
    p.add_argument("--episodes",  type=int, default=5,
                   help="Evaluation episodes after training (default: 5)")
    p.add_argument("--samples",   type=int, default=5_000,
                   help="Synthetic dataset size for ML training fallback (default: 5000)")
    p.add_argument("--dataset",   type=str, default=None,
                   help="Path to a real CIC-IDS2017 CSV (enables hybrid Path 3). "
                        "When provided, this is used for ML training instead of synthetic data. "
                        "RL environment always uses MockONOSClient regardless.")
    p.add_argument("--fast",      action="store_true",
                   help="Shortcut: 3000 timesteps, 3 episodes, 3000 samples — quick sanity check")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Step 1 — Generate synthetic dataset
# ─────────────────────────────────────────────────────────────────────

def generate_data(samples: int, seed: int) -> str:
    dataset_path = "data/synthetic_ids_dataset.csv"
    if os.path.exists(dataset_path):
        logging.info(f"Dataset already exists at {dataset_path} — skipping generation.")
        return dataset_path

    logging.info(f"Generating synthetic dataset ({samples:,} samples)...")
    from scripts.generate_synthetic_data import main as gen_main
    import unittest.mock as mock
    # Patch sys.argv so arg parser inside generate_synthetic_data doesn't conflict
    with mock.patch("sys.argv", ["generate_synthetic_data.py",
                                  "--samples", str(samples),
                                  "--seed",    str(seed),
                                  "--out",     dataset_path]):
        gen_main()
    return dataset_path


# ─────────────────────────────────────────────────────────────────────
# Step 2a — Train ML on a real CIC-IDS2017 CSV  (Path 3 — hybrid)
# ─────────────────────────────────────────────────────────────────────

def train_ml_detector_real(dataset_path: str):
    """
    Path 3 (hybrid): trains on a real CIC-IDS2017 CSV file.

    Uses ThreatDetector.train() which already handles:
      - Leading-space column names  (e.g. ' Label')
      - NaN / ±inf scrubbing
      - FEATURE_COLS selection
      - Saving all three models + scaler to models/

    Supported input formats:
      - Any single CIC-IDS2017 day CSV  (Monday through Friday)
      - A concatenation of multiple days
      - InSDN CSV (same feature schema)

    The RL environment is NOT affected — it always runs with MockONOSClient.
    """
    from src.threat_detector import ThreatDetector

    logging.info(f"[Path 3 — Hybrid] Training ML on real CIC-IDS2017 data: {dataset_path}")
    logging.info("  RL environment will still use MockONOSClient (synthetic phases).")

    det = ThreatDetector(client=None)
    det.train(dataset_path, sample_size=150_000)
    logging.info("ML models trained on real data and saved to models/")


# ─────────────────────────────────────────────────────────────────────
# Step 2b — Train ML on synthetic data  (Path 2 — synthetic only / fallback)
# ─────────────────────────────────────────────────────────────────────

def train_ml_detector(dataset_path: str):
    """
    Path 2 (synthetic): trains EnsembleSwitchingDetector (HybridMTD — Li et al. 2021)
    on the synthetic dataset generated in Step 1.
    Aligns synthetic feature names to FEATURE_COLS from threat_detector.py.
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from src.threat_detector import FEATURE_COLS   # authoritative 25-feature list

    logging.info("Training HybridMTD Ensemble Switching ML models...")

    df = pd.read_csv(dataset_path)

    # Synthetic feature → FEATURE_COLS name mapping (subset that we have)
    rename = {
        "flow_duration":      "Flow Duration",
        "total_fwd_pkts":     "Total Fwd Packets",
        "total_bwd_pkts":     "Total Backward Packets",
        "fwd_pkt_len_mean":   "Fwd Packet Length Mean",
        "bwd_pkt_len_mean":   "Bwd Packet Length Mean",
        "flow_bytes_per_sec": "Flow Bytes/s",
        "flow_pkts_per_sec":  "Flow Packets/s",
        "fwd_iat_mean":       "Flow IAT Mean",
        "bwd_iat_mean":       "Fwd IAT Total",
        "syn_flag_count":     "SYN Flag Count",
        "rst_flag_count":     "RST Flag Count",
        "psh_flag_count":     "Fwd PSH Flags",
        "ack_flag_count":     "ACK Flag Count",
        "idle_mean":          "Bwd IAT Total",
        "active_mean":        "Flow IAT Std",
        "subflow_fwd_bytes":  "Total Length of Fwd Packets",
        "subflow_bwd_bytes":  "Total Length of Bwd Packets",
    }
    df = df.rename(columns=rename)

    # Binary label: BENIGN=0, everything else=1
    df["binary_label"] = (df["label"] != "BENIGN").astype(int)

    # Build feature matrix aligned to FEATURE_COLS — pad missing with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    X = df[FEATURE_COLS].fillna(0).values
    y = df["binary_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)

    # Regime A: Random Forest
    logging.info("  Training Regime A: RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                n_jobs=-1, class_weight="balanced", random_state=42)
    rf.fit(X_train_s, y_train)
    from sklearn.metrics import classification_report
    logging.info("  RF test results:\n" + classification_report(
        y_test, rf.predict(X_test_s), target_names=["Benign", "Threat"]))

    # Regime B: XGBoost (fall back to RF if not installed)
    try:
        from xgboost import XGBClassifier
        logging.info("  Training Regime B: XGBoost...")
        xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                             eval_metric="logloss", n_jobs=-1, random_state=42)
        xgb.fit(X_train_s, y_train)
        logging.info("  XGB test results:\n" + classification_report(
            y_test, xgb.predict(X_test_s), target_names=["Benign", "Threat"]))
        import joblib
        joblib.dump(xgb, "models/threat_detector_xgb.pkl")
    except ImportError:
        logging.warning("  XGBoost not installed — Regime B will use RF fallback.")

    # Regime C: Isolation Forest (unsupervised)
    logging.info("  Training Regime C: IsolationForest...")
    isof = IsolationForest(contamination=0.15, n_estimators=100,
                           random_state=42, n_jobs=-1)
    isof.fit(X_train_s)

    import joblib
    joblib.dump(rf,     "models/threat_detector_rf.pkl")
    joblib.dump(isof,   "models/threat_detector_isof.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    logging.info("ML models saved to models/")


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Patch environment to use MockONOSClient
# ─────────────────────────────────────────────────────────────────────

def make_mock_env():
    """
    Builds an SDN_MTD_Env that uses MockONOSClient instead of the real one.
    Monkey-patches src.onos_client.ONOSClient before instantiation so that
    the environment's __init__ picks up the mock without any code changes.
    """
    import src.onos_client as _oc_module
    from src.mock_onos_client import MockONOSClient

    # Preserve original for restoration
    _original = _oc_module.ONOSClient
    _oc_module.ONOSClient = MockONOSClient  # patch

    # Re-import after patch so mtd_env.py picks up the mock
    import importlib
    import src.mtd_env as _env_module
    importlib.reload(_env_module)
    SDN_MTD_Env = _env_module.SDN_MTD_Env

    # Instantiate without a config file (MockONOSClient needs none)
    env = SDN_MTD_Env.__new__(SDN_MTD_Env)
    import gymnasium as gym
    from gymnasium import spaces
    gym.Env.__init__(env)
    env.client         = MockONOSClient()
    env.render_mode    = None
    env.current_step   = 0
    env.episode_rewards = []
    env.action_space   = spaces.Discrete(3)
    from src.mtd_env import OBS_SIZE
    env.observation_space = spaces.Box(
        low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
    )
    env.threat_score   = 0.0
    env._prev_flow_ids = set()
    env._prev_src_ips  = []
    env.max_steps      = 200

    # Restore original so other code is unaffected
    _oc_module.ONOSClient = _original

    logging.info("SDN_MTD_Env created with MockONOSClient.")
    return env


# ─────────────────────────────────────────────────────────────────────
# Step 4 — Train the RL agent
# ─────────────────────────────────────────────────────────────────────

def train_rl(env, timesteps: int):
    try:
        from stable_baselines3 import DQN
        logging.info(f"Training Double DQN agent ({timesteps:,} timesteps)...")
        model = DQN(
            "MlpPolicy", env,
            verbose=0,
            learning_rate=1e-4,
            target_update_interval=500,
            tau=0.005,
            buffer_size=50_000,
            learning_starts=min(1_000, timesteps // 4),
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            policy_kwargs={"net_arch": [512, 256]},
        )
        model.learn(total_timesteps=timesteps)
        os.makedirs("models", exist_ok=True)
        model.save("models/dqn_mtd_offline")
        logging.info("RL model saved to models/dqn_mtd_offline")
        return model
    except ImportError:
        logging.warning("stable-baselines3 not installed. Using heuristic policy.")
        return None


# ─────────────────────────────────────────────────────────────────────
# Step 5 — Evaluation loop
# ─────────────────────────────────────────────────────────────────────

def run_evaluation(env, model, num_episodes: int):
    from src.threat_detector import ThreatDetector

    # Instantiate ThreatDetector with the mock client — no config file needed
    detector = ThreatDetector(client=env.client)

    os.makedirs("results", exist_ok=True)
    results_path = "results/evaluation_results.csv"
    fieldnames = [
        "episode", "step", "action", "reward",
        "attacker_entropy", "path_entropy", "attacker_belief",
        "recon_prevented", "deception_cost",
        "latency_ms", "threat_score", "shap_top_feature",
        "cumulative_reward",
    ]

    all_rewards = []

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            cum_reward = 0.0
            step = 0
            threat_score = 0.0   # carried forward from previous step
            shap_top = "-"

            while not done:
                # Action selection using threat score from previous step's observation
                if model:
                    action, _ = model.predict(obs, deterministic=False)
                else:
                    action = 0
                # Threat-responsive override: escalate action as threat rises
                if   threat_score > 0.65: action = 2
                elif threat_score > 0.25: action = max(int(action), 1)

                # Step the env — this calls get_network_state() exactly ONCE
                # via _get_observation(), avoiding a double-increment of the mock
                # step counter that would scramble phase detection.
                obs, reward, terminated, truncated, info = env.step(int(action))

                # Read flows from the cached last state — no extra step increment
                flows = (env.client._last_state.get("flows", [])
                         if hasattr(env.client, "_last_state") else [])

                # Compute threat score for next step
                if flows and detector.ensemble.rf_model:
                    try:
                        pred, conf = detector.predict_from_flows(flows)
                        # Convert model confidence to threat probability:
                        # P(threat) = conf when pred==1; 1-conf when pred==0 (benign)
                        threat_score = conf if pred == 1 else 1.0 - conf
                    except Exception:
                        threat_score = min(1.0, len(flows) / 3000.0)
                else:
                    threat_score = min(1.0, len(flows) / 3000.0)

                # SHAP top feature — fires when threat score crosses recon threshold
                # Ref: Lundberg & Lee (2017) NeurIPS
                shap_top = "-"
                if flows and threat_score > 0.1:
                    try:
                        exp = detector.explain_prediction(flows)
                        if exp.get("feature_names"):
                            shap_top = exp["feature_names"][0]
                    except Exception:
                        pass
                done = terminated or truncated
                cum_reward += reward
                step += 1

                writer.writerow({
                    "episode":          ep + 1,
                    "step":             step,
                    "action":           int(action),
                    "reward":           round(float(reward), 4),
                    "attacker_entropy": round(info.get("attacker_entropy", 0.0), 4),
                    "path_entropy":     round(info.get("path_entropy", 0.0), 4),
                    "attacker_belief":  round(info.get("attacker_belief", 0.0), 4),
                    "recon_prevented":  round(info.get("recon_prevented", 0.0), 4),
                    "deception_cost":   round(info.get("deception_cost", 0.0), 4),
                    "latency_ms":       round(info.get("latency_ms", 0.0), 2),
                    "threat_score":     round(float(threat_score), 4),
                    "shap_top_feature": shap_top,
                    "cumulative_reward": round(float(cum_reward), 4),
                })

            all_rewards.append(cum_reward)
            logging.info(f"Episode {ep+1}/{num_episodes} — reward: {cum_reward:.3f} ({step} steps)")

    return results_path, all_rewards


# ─────────────────────────────────────────────────────────────────────
# Step 6 — Print results summary
# ─────────────────────────────────────────────────────────────────────

def print_summary(results_path: str, all_rewards: list):
    import pandas as pd
    df = pd.read_csv(results_path)

    print("\n" + "="*55)
    print("  MTD-Brain Offline Demo — Results Summary")
    print("="*55)
    print(f"\n  Episodes completed     : {df['episode'].max()}")
    print(f"  Total steps logged     : {len(df):,}")

    print(f"\n  Reward (per episode):")
    for i, r in enumerate(all_rewards, 1):
        bar = "█" * int(max(0, r) / max(abs(r) for r in all_rewards) * 20)
        print(f"    Ep {i:2d}: {r:8.3f}  {bar}")

    print(f"\n  Mean reward            : {df['reward'].mean():.4f}")
    print(f"  Mean attacker entropy  : {df['attacker_entropy'].mean():.4f}  (higher = attacker more confused)")
    print(f"  Mean path entropy      : {df['path_entropy'].mean():.4f}")
    print(f"  Mean attacker belief   : {df['attacker_belief'].mean():.4f}  (lower = less recon success)")
    print(f"  Mean threat score      : {df['threat_score'].mean():.4f}")
    print(f"  Mean latency (ms)      : {df['latency_ms'].mean():.1f}")

    action_counts = df["action"].value_counts().sort_index()
    print(f"\n  Agent action distribution:")
    for a, cnt in action_counts.items():
        label = {0: "No Move ", 1: "Moderate", 2: "Aggressv"}[a]
        pct = cnt / len(df) * 100
        print(f"    {label}  ({a}): {cnt:5d} steps  ({pct:.1f}%)")

    if df["shap_top_feature"].nunique() > 1:
        top_shap = df[df["shap_top_feature"] != "-"]["shap_top_feature"].value_counts()
        print(f"\n  Top SHAP features (most common alert drivers):")
        for feat, cnt in top_shap.head(3).items():
            print(f"    {feat}: {cnt} times")

    print(f"\n  Full results saved to  : {results_path}")
    print(f"  Plot them with         : python scripts/plot_results.py")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.fast:
        args.timesteps = 3_000
        args.episodes  = 3
        args.samples   = 3_000
        logging.info("--fast mode: 3000 timesteps, 3 episodes, 3000 samples")

    logging.info("="*55)
    logging.info("  MTD-Brain Offline Demo")
    logging.info("  No ONOS controller required")
    logging.info("="*55)

    # ── Step 1 & 2: Data + ML training ───────────────────────────────
    # Path 3 (hybrid): real CIC-IDS2017 CSV → ML models
    #                  MockONOSClient synthetic → RL environment  (always)
    # Path 2 (default): synthetic data → both ML and RL
    if args.dataset:
        # Validate file exists before spending time on anything else
        if not os.path.isfile(args.dataset):
            logging.error(f"--dataset file not found: {args.dataset}")
            logging.error("Download CIC-IDS2017 from https://www.kaggle.com/datasets/cicdataset/cicids2017")
            logging.error("and place a day CSV in data/cic-ids2017/")
            sys.exit(1)
        logging.info("Mode: PATH 3 — Hybrid (real CIC-IDS2017 ML + synthetic RL)")
        train_ml_detector_real(args.dataset)
    else:
        logging.info("Mode: PATH 2 — Synthetic only (no --dataset provided)")
        dataset_path = generate_data(args.samples, args.seed)
        train_ml_detector(dataset_path)

    # Step 3: Mock env
    env = make_mock_env()

    # Step 4: Train RL
    model = train_rl(env, args.timesteps)

    # Step 5: Evaluate
    results_path, all_rewards = run_evaluation(env, model, args.episodes)

    # Step 6: Summary
    print_summary(results_path, all_rewards)

    env.close()


if __name__ == "__main__":
    main()
