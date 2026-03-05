"""
plot_results.py
---------------
Generates paper-ready figures from results/evaluation_results.csv.

Produces 5 plots (saved to results/figures/):
    1. cumulative_reward.png    — Reward curve per episode
    2. attacker_entropy.png     — H_A over time (key Eghtesad 2020 metric)
    3. threat_vs_action.png     — Threat score with RL actions annotated
    4. payoff_breakdown.png     — Recon Prevented vs. Deception Cost per step
    5. shap_features.png        — Bar chart of top SHAP drivers (Lundberg 2017)

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --input results/evaluation_results.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Plot MTD evaluation results")
    p.add_argument("--input",  default="results/evaluation_results.csv",
                   help="Path to evaluation CSV (default: results/evaluation_results.csv)")
    p.add_argument("--outdir", default="results/figures",
                   help="Output directory for figures (default: results/figures)")
    p.add_argument("--dpi",    type=int, default=150,
                   help="Figure DPI (default: 150; use 300 for print)")
    return p.parse_args()


ACTION_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
ACTION_LABELS = {0: "No Move", 1: "Moderate", 2: "Aggressive"}


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Results file not found: {path}")
        print("        Run 'python scripts/run_offline.py' first to generate it.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")
    print(f"  Episodes : {df['episode'].nunique()}")
    print(f"  Columns  : {list(df.columns)}\n")
    return df


# ─────────────────────────────────────────────────────────────────────
# Plot 1 — Cumulative reward per episode
# ─────────────────────────────────────────────────────────────────────

def plot_reward_curve(df: pd.DataFrame, outdir: str, dpi: int):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: per-step reward (smoothed)
    ax = axes[0]
    window = max(1, len(df) // 100)
    df_sorted = df.sort_values(["episode", "step"])
    smoothed = df_sorted["reward"].rolling(window=window, min_periods=1).mean()
    ax.plot(smoothed.values, color="#1565C0", linewidth=1.0, alpha=0.8, label="Smoothed reward")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward $R_d$")
    ax.set_title("Per-step Reward (Zero-Sum Payoff, Eghtesad 2020 Eq. 6)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: cumulative reward per episode
    ax = axes[1]
    ep_rewards = df.groupby("episode")["reward"].sum()
    bars = ax.bar(ep_rewards.index, ep_rewards.values,
                  color=["#1565C0" if v >= 0 else "#C62828" for v in ep_rewards.values],
                  alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Cumulative Reward per Episode")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "cumulative_reward.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# Plot 2 — Attacker entropy over time
# ─────────────────────────────────────────────────────────────────────

def plot_attacker_entropy(df: pd.DataFrame, outdir: str, dpi: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))

    window = max(1, len(df) // 80)
    df_s = df.sort_values(["episode", "step"])

    h_a = df_s["attacker_entropy"].rolling(window, min_periods=1).mean()
    h_p = df_s["path_entropy"].rolling(window, min_periods=1).mean()

    ax.plot(h_a.values, color="#6A1B9A", linewidth=1.2, label="$H_A$ — Attacker Knowledge Entropy")
    ax.plot(h_p.values, color="#00838F", linewidth=1.0, linestyle="--",
            label="$H_P$ — Path Entropy")
    ax.fill_between(range(len(h_a)), h_a.values, alpha=0.1, color="#6A1B9A")

    # Mark aggressive mutations
    agg_steps = df_s[df_s["action"] == 2].index
    for i, row_i in enumerate(df_s.index):
        if row_i in set(agg_steps):
            ax.axvline(x=list(df_s.index).index(row_i), color="#F44336",
                       linewidth=0.4, alpha=0.3)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Attacker Knowledge Entropy $H_A$ over Time\n"
                 "Red lines = Aggressive MTD mutations  |  "
                 "Ref: Eghtesad et al. (2020) GameSec")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "attacker_entropy.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# Plot 3 — Threat score with RL action overlay
# ─────────────────────────────────────────────────────────────────────

def plot_threat_vs_action(df: pd.DataFrame, outdir: str, dpi: int):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 4.5))

    df_s = df.sort_values(["episode", "step"]).reset_index(drop=True)
    window = max(1, len(df_s) // 80)
    threat_smooth = df_s["threat_score"].rolling(window, min_periods=1).mean()

    ax.plot(threat_smooth.values, color="#B71C1C", linewidth=1.2,
            label="Threat Score (HybridMTD — Li 2021)")
    ax.axhline(0.9, color="#B71C1C", linewidth=0.6, linestyle="--", alpha=0.5,
               label="Alert threshold (90%)")

    # Scatter action decisions
    for action_val, color in ACTION_COLORS.items():
        mask = df_s["action"] == action_val
        ax.scatter(df_s.index[mask], df_s["threat_score"][mask],
                   c=color, s=6, alpha=0.5, zorder=3)

    patches = [mpatches.Patch(color=c, label=f"Action {a}: {ACTION_LABELS[a]}")
               for a, c in ACTION_COLORS.items()]
    ax.legend(handles=[ax.lines[0], ax.lines[1]] + patches, fontsize=8, loc="upper right")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Threat Score")
    ax.set_title("HybridMTD Threat Score with RL Agent Actions\n"
                 "Ref: Li et al. (2021) IEEE TDSC  |  Double DQN: van Hasselt et al. (2016)")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "threat_vs_action.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# Plot 4 — Payoff breakdown: recon_prevented vs. deception_cost
# ─────────────────────────────────────────────────────────────────────

def plot_payoff_breakdown(df: pd.DataFrame, outdir: str, dpi: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))

    df_s = df.sort_values(["episode", "step"]).reset_index(drop=True)
    window = max(1, len(df_s) // 80)

    rp = df_s["recon_prevented"].rolling(window, min_periods=1).mean()
    dc = df_s["deception_cost"].rolling(window, min_periods=1).mean()
    net = rp - dc

    ax.fill_between(range(len(rp)),  rp.values,  alpha=0.3, color="#2E7D32", label="Recon Prevented")
    ax.fill_between(range(len(dc)),  dc.values,  alpha=0.3, color="#C62828", label="Deception Cost")
    ax.plot(net.values, color="#1565C0", linewidth=1.3, label="Net Payoff (Rp − Dc)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Payoff Component")
    ax.set_title("Zero-Sum Payoff Decomposition\n"
                 "Ref: Eghtesad et al. (2020) Eq. (6): $R_d = W_{rp}(1-b) - W_{dc}·a_d - ...$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "payoff_breakdown.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# Plot 5 — SHAP top features
# ─────────────────────────────────────────────────────────────────────

def plot_shap_features(df: pd.DataFrame, outdir: str, dpi: int):
    import matplotlib.pyplot as plt

    shap_col = df[df["shap_top_feature"] != "-"]["shap_top_feature"]
    if len(shap_col) == 0:
        print("  SHAP features: no data (models may not have been SHAP-evaluated yet).")
        return

    counts = shap_col.value_counts().head(10)

    fig, ax = plt.subplots(figsize=(9, max(3, len(counts) * 0.45)))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color="#1565C0", alpha=0.8)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Times ranked as top SHAP feature")
    ax.set_title("Top Flow Features Driving Threat Detections (TreeSHAP)\n"
                 "Ref: Lundberg & Lee (2017) NeurIPS — "
                 "feature attribution for RandomForest / XGBoost")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "shap_features.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — no display required
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
    except ImportError:
        print("[ERROR] matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    df = load_data(args.input)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Generating figures → {args.outdir}/")
    plot_reward_curve(df, args.outdir, args.dpi)
    plot_attacker_entropy(df, args.outdir, args.dpi)
    plot_threat_vs_action(df, args.outdir, args.dpi)
    plot_payoff_breakdown(df, args.outdir, args.dpi)
    plot_shap_features(df, args.outdir, args.dpi)

    print(f"\nAll figures saved to {args.outdir}/")
    print("Files:")
    for fname in sorted(os.listdir(args.outdir)):
        if fname.endswith(".png"):
            fpath = os.path.join(args.outdir, fname)
            kb = os.path.getsize(fpath) // 1024
            print(f"  {fname:<35} {kb} KB")


if __name__ == "__main__":
    main()
