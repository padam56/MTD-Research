"""
generate_synthetic_data.py
--------------------------
Generates a synthetic training dataset that mimics the CIC-IDS2017 format.

Produces 5 traffic classes:
    BENIGN        — normal background traffic
    DoS           — high-volume packet floods (SYN flag heavy)
    PortScan      — low-volume, many dst IPs, fast idle times
    Brute-Force   — repeated short flows to a single dst
    Infiltration  — low-and-slow exfiltration (large packets, infrequent)

Output: data/synthetic_ids_dataset.csv
         data/synthetic_ids_dataset_summary.txt

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --samples 20000 --seed 99
"""

import argparse
import os
import random
import math
import numpy as np
import pandas as pd

# ── Feature names matching CIC-IDS2017 column schema ──────────────────
FEATURES = [
    "flow_duration",
    "total_fwd_pkts",
    "total_bwd_pkts",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "idle_mean",
    "active_mean",
    "subflow_fwd_bytes",
    "subflow_bwd_bytes",
]

LABEL_COL = "label"


def _add_noise(value: float, noise_frac: float = 0.15) -> float:
    """Multiplicative Gaussian noise to avoid perfectly clean synthetic data."""
    return max(0.0, value * (1 + random.gauss(0, noise_frac)))


def generate_benign(n: int, rng) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        rows.append({
            "flow_duration":      _add_noise(rng.uniform(0.5, 120.0)),
            "total_fwd_pkts":     int(_add_noise(rng.integers(5, 200))),
            "total_bwd_pkts":     int(_add_noise(rng.integers(3, 150))),
            "fwd_pkt_len_mean":   _add_noise(rng.uniform(200, 1200)),
            "bwd_pkt_len_mean":   _add_noise(rng.uniform(100, 800)),
            "flow_bytes_per_sec": _add_noise(rng.uniform(100, 5000)),
            "flow_pkts_per_sec":  _add_noise(rng.uniform(0.5, 50)),
            "fwd_iat_mean":       _add_noise(rng.uniform(0.001, 1.5)),
            "bwd_iat_mean":       _add_noise(rng.uniform(0.001, 1.5)),
            "syn_flag_count":     rng.integers(0, 3),
            "rst_flag_count":     rng.integers(0, 2),
            "psh_flag_count":     rng.integers(0, 5),
            "ack_flag_count":     rng.integers(5, 60),
            "urg_flag_count":     0,
            "idle_mean":          _add_noise(rng.uniform(0.01, 2.0)),
            "active_mean":        _add_noise(rng.uniform(0.1, 5.0)),
            "subflow_fwd_bytes":  int(_add_noise(rng.integers(100, 10000))),
            "subflow_bwd_bytes":  int(_add_noise(rng.integers(50, 8000))),
            LABEL_COL:            "BENIGN",
        })
    return pd.DataFrame(rows)


def generate_dos(n: int, rng) -> pd.DataFrame:
    """DDoS / SYN-flood: massive flow rate, tiny packets, high SYN count."""
    rows = []
    for _ in range(n):
        rows.append({
            "flow_duration":      _add_noise(rng.uniform(0.001, 5.0)),
            "total_fwd_pkts":     int(_add_noise(rng.integers(500, 5000))),
            "total_bwd_pkts":     int(_add_noise(rng.integers(0, 10))),
            "fwd_pkt_len_mean":   _add_noise(rng.uniform(40, 80)),  # tiny pkts
            "bwd_pkt_len_mean":   _add_noise(rng.uniform(0, 40)),
            "flow_bytes_per_sec": _add_noise(rng.uniform(50000, 500000)),
            "flow_pkts_per_sec":  _add_noise(rng.uniform(500, 10000)),
            "fwd_iat_mean":       _add_noise(rng.uniform(0.00001, 0.001)),
            "bwd_iat_mean":       _add_noise(rng.uniform(0.0, 0.001)),
            "syn_flag_count":     rng.integers(400, 1000),
            "rst_flag_count":     rng.integers(50, 300),
            "psh_flag_count":     rng.integers(0, 5),
            "ack_flag_count":     rng.integers(0, 10),
            "urg_flag_count":     0,
            "idle_mean":          _add_noise(rng.uniform(0.0, 0.005)),
            "active_mean":        _add_noise(rng.uniform(0.001, 0.1)),
            "subflow_fwd_bytes":  int(_add_noise(rng.integers(500, 5000))),
            "subflow_bwd_bytes":  int(_add_noise(rng.integers(0, 100))),
            LABEL_COL:            "DoS",
        })
    return pd.DataFrame(rows)


def generate_portscan(n: int, rng) -> pd.DataFrame:
    """Port scan: many very short flows, no response (bwd=0), RST-heavy."""
    rows = []
    for _ in range(n):
        rows.append({
            "flow_duration":      _add_noise(rng.uniform(0.0001, 0.5)),
            "total_fwd_pkts":     int(rng.integers(1, 4)),
            "total_bwd_pkts":     0,
            "fwd_pkt_len_mean":   _add_noise(rng.uniform(40, 60)),
            "bwd_pkt_len_mean":   0.0,
            "flow_bytes_per_sec": _add_noise(rng.uniform(100, 2000)),
            "flow_pkts_per_sec":  _add_noise(rng.uniform(2, 20)),
            "fwd_iat_mean":       _add_noise(rng.uniform(0.0001, 0.1)),
            "bwd_iat_mean":       0.0,
            "syn_flag_count":     rng.integers(1, 3),
            "rst_flag_count":     rng.integers(1, 5),
            "psh_flag_count":     0,
            "ack_flag_count":     0,
            "urg_flag_count":     0,
            "idle_mean":          _add_noise(rng.uniform(0.5, 5.0)),
            "active_mean":        _add_noise(rng.uniform(0.0001, 0.01)),
            "subflow_fwd_bytes":  int(rng.integers(40, 200)),
            "subflow_bwd_bytes":  0,
            LABEL_COL:            "PortScan",
        })
    return pd.DataFrame(rows)


def generate_bruteforce(n: int, rng) -> pd.DataFrame:
    """Brute-force login: repeated medium flows to one dst, low diversity."""
    rows = []
    for _ in range(n):
        rows.append({
            "flow_duration":      _add_noise(rng.uniform(0.01, 3.0)),
            "total_fwd_pkts":     int(_add_noise(rng.integers(2, 20))),
            "total_bwd_pkts":     int(_add_noise(rng.integers(1, 15))),
            "fwd_pkt_len_mean":   _add_noise(rng.uniform(60, 300)),
            "bwd_pkt_len_mean":   _add_noise(rng.uniform(40, 200)),
            "flow_bytes_per_sec": _add_noise(rng.uniform(200, 8000)),
            "flow_pkts_per_sec":  _add_noise(rng.uniform(1, 30)),
            "fwd_iat_mean":       _add_noise(rng.uniform(0.01, 0.5)),
            "bwd_iat_mean":       _add_noise(rng.uniform(0.01, 0.5)),
            "syn_flag_count":     rng.integers(1, 5),
            "rst_flag_count":     rng.integers(0, 3),
            "psh_flag_count":     rng.integers(2, 10),
            "ack_flag_count":     rng.integers(2, 20),
            "urg_flag_count":     0,
            "idle_mean":          _add_noise(rng.uniform(0.1, 1.0)),
            "active_mean":        _add_noise(rng.uniform(0.05, 0.5)),
            "subflow_fwd_bytes":  int(_add_noise(rng.integers(100, 2000))),
            "subflow_bwd_bytes":  int(_add_noise(rng.integers(50, 1500))),
            LABEL_COL:            "BruteForce",
        })
    return pd.DataFrame(rows)


def generate_infiltration(n: int, rng) -> pd.DataFrame:
    """Low-and-slow exfiltration: large packets, infrequent, long idle."""
    rows = []
    for _ in range(n):
        rows.append({
            "flow_duration":      _add_noise(rng.uniform(60.0, 600.0)),
            "total_fwd_pkts":     int(_add_noise(rng.integers(2, 15))),
            "total_bwd_pkts":     int(_add_noise(rng.integers(50, 300))),
            "fwd_pkt_len_mean":   _add_noise(rng.uniform(800, 1500)),
            "bwd_pkt_len_mean":   _add_noise(rng.uniform(600, 1200)),
            "flow_bytes_per_sec": _add_noise(rng.uniform(50, 800)),
            "flow_pkts_per_sec":  _add_noise(rng.uniform(0.01, 1.0)),
            "fwd_iat_mean":       _add_noise(rng.uniform(5.0, 60.0)),
            "bwd_iat_mean":       _add_noise(rng.uniform(5.0, 60.0)),
            "syn_flag_count":     rng.integers(0, 2),
            "rst_flag_count":     0,
            "psh_flag_count":     rng.integers(5, 20),
            "ack_flag_count":     rng.integers(10, 50),
            "urg_flag_count":     0,
            "idle_mean":          _add_noise(rng.uniform(10.0, 100.0)),
            "active_mean":        _add_noise(rng.uniform(0.1, 2.0)),
            "subflow_fwd_bytes":  int(_add_noise(rng.integers(5000, 50000))),
            "subflow_bwd_bytes":  int(_add_noise(rng.integers(2000, 30000))),
            LABEL_COL:            "Infiltration",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IDS training data")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Total samples to generate (default: 10000)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out",     type=str, default="data/synthetic_ids_dataset.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    n = args.samples

    # Class proportions roughly matching CIC-IDS2017 distribution
    counts = {
        "benign":        int(n * 0.55),
        "dos":           int(n * 0.20),
        "portscan":      int(n * 0.12),
        "bruteforce":    int(n * 0.08),
        "infiltration":  int(n * 0.05),
    }
    # Fix rounding
    counts["benign"] += n - sum(counts.values())

    print(f"\nGenerating {n:,} synthetic samples (seed={args.seed}):")
    for cls, cnt in counts.items():
        print(f"  {cls:15s}: {cnt:,}")

    frames = [
        generate_benign(counts["benign"], rng),
        generate_dos(counts["dos"], rng),
        generate_portscan(counts["portscan"], rng),
        generate_bruteforce(counts["bruteforce"], rng),
        generate_infiltration(counts["infiltration"], rng),
    ]

    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=args.seed)

    # Clip negatives (noise can rarely produce tiny negatives)
    numeric_cols = [c for c in df.columns if c != LABEL_COL]
    df[numeric_cols] = df[numeric_cols].clip(lower=0.0)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = (
        f"Dataset Summary\n"
        f"{'='*40}\n"
        f"Total samples : {len(df):,}\n"
        f"Features      : {len(FEATURES)}\n"
        f"Classes       : {df[LABEL_COL].nunique()}\n\n"
        f"Class distribution:\n{df[LABEL_COL].value_counts().to_string()}\n\n"
        f"Feature stats:\n{df[numeric_cols].describe().to_string()}\n"
    )

    summary_path = args.out.replace(".csv", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\nSaved dataset  → {args.out}")
    print(f"Saved summary  → {summary_path}")
    print(f"\nClass distribution:\n{df[LABEL_COL].value_counts().to_string()}")


if __name__ == "__main__":
    main()
