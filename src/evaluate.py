"""
evaluate.py — Evaluate MTD strategies and generate paper metrics + plots
=========================================================================
Runs the trained DQN agent (and baselines) on the MTD-Playground environment,
computes all metrics from Section 7 of the MTD-Playground paper, and
generates CSV results + matplotlib plots.

Evaluation Metrics (MTD-Playground paper Section 7.2):
    1. Attack Success Rate (ASR)
    2. Attack Completion Time (ACT)
    3. Reconnaissance Accuracy
    4. Path Entropy (H_P)
    5. Network Unpredictability (Attacker Entropy H_A)
    6. End-to-End Latency
    7. Throughput proxy (service availability)
    8. Controller Overhead (mutation count)

Strategies compared:
    - No MTD (baseline)
    - Random MTD (random action each step)
    - Periodic MTD (mutate every N steps)
    - DQN-based adaptive MTD (Eghtesad et al. 2020)

Usage:
    python evaluate.py                          # Use trained model
    python evaluate.py --no-model               # Baselines only
    python evaluate.py --episodes 50 --seed 123
"""

import argparse
import csv
import os
import sys
import numpy as np
from collections import defaultdict

from mtd_env import MTDPlaygroundEnv


def run_strategy(env, strategy_fn, num_episodes, seed):
    """Run a strategy for multiple episodes and collect metrics."""
    all_metrics = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_data = {
            "steps": [],
            "actions": [],
            "rewards": [],
            "path_entropy": [],
            "attacker_entropy": [],
            "attacker_knowledge": [],
            "recon_accuracy": [],
            "latency_ms": [],
            "service_availability": [],
            "attacker_stage": [],
        }

        step = 0
        while not done:
            action = strategy_fn(obs, step)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_data["steps"].append(step)
            ep_data["actions"].append(action)
            ep_data["rewards"].append(reward)
            ep_data["path_entropy"].append(info["path_entropy"])
            ep_data["attacker_entropy"].append(info["attacker_entropy"])
            ep_data["attacker_knowledge"].append(info["attacker_knowledge"])
            ep_data["recon_accuracy"].append(info["recon_accuracy"])
            ep_data["latency_ms"].append(info["latency_ms"])
            ep_data["service_availability"].append(info["service_availability"])
            ep_data["attacker_stage"].append(info["attacker_stage"])
            step += 1

        # Compute episode-level metrics (Section 7.2 of paper)
        exfiltrated = env.attack_succeeded
        ep_metrics = {
            "episode": ep + 1,
            "attack_success": 1 if exfiltrated else 0,
            "attack_completion_time": step if exfiltrated else MAX_EPISODE_STEPS,
            "final_attacker_stage": ep_data["attacker_stage"][-1],
            "mean_recon_accuracy": np.mean(ep_data["recon_accuracy"]),
            "mean_path_entropy": np.mean(ep_data["path_entropy"]),
            "mean_attacker_entropy": np.mean(ep_data["attacker_entropy"]),
            "mean_latency_ms": np.mean(ep_data["latency_ms"]),
            "mean_service_availability": np.mean(ep_data["service_availability"]),
            "total_mutations": sum(1 for a in ep_data["actions"] if a > 0),
            "total_reward": sum(ep_data["rewards"]),
            "steps": step,
            "timeseries": ep_data,
        }
        all_metrics.append(ep_metrics)

    return all_metrics


MAX_EPISODE_STEPS = 200


def strategy_no_mtd(obs, step):
    return 0


def strategy_random(obs, step):
    return np.random.randint(0, 3)


def strategy_periodic(obs, step, period=10):
    return 2 if step % period == 0 else 0


def make_dqn_strategy(model):
    def strategy(obs, step):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return strategy


def compute_summary(name, metrics_list):
    """Compute aggregate statistics across episodes."""
    n = len(metrics_list)
    asr = sum(m["attack_success"] for m in metrics_list) / n
    mean_act = np.mean([m["attack_completion_time"] for m in metrics_list])
    mean_recon = np.mean([m["mean_recon_accuracy"] for m in metrics_list])
    mean_path_ent = np.mean([m["mean_path_entropy"] for m in metrics_list])
    mean_att_ent = np.mean([m["mean_attacker_entropy"] for m in metrics_list])
    mean_latency = np.mean([m["mean_latency_ms"] for m in metrics_list])
    mean_avail = np.mean([m["mean_service_availability"] for m in metrics_list])
    mean_mutations = np.mean([m["total_mutations"] for m in metrics_list])
    mean_reward = np.mean([m["total_reward"] for m in metrics_list])

    return {
        "strategy": name,
        "ASR": round(asr, 4),
        "mean_ACT": round(mean_act, 1),
        "mean_recon_accuracy": round(mean_recon, 4),
        "mean_path_entropy": round(mean_path_ent, 4),
        "mean_attacker_entropy": round(mean_att_ent, 4),
        "mean_latency_ms": round(mean_latency, 2),
        "mean_service_availability": round(mean_avail, 4),
        "mean_mutations": round(mean_mutations, 1),
        "mean_total_reward": round(mean_reward, 2),
    }


def save_detailed_csv(results_dir, strategy_name, metrics_list):
    """Save per-step timeseries data for each strategy."""
    path = os.path.join(results_dir, f"timeseries_{strategy_name}.csv")
    fieldnames = [
        "episode", "step", "action", "reward",
        "path_entropy", "attacker_entropy", "attacker_knowledge",
        "recon_accuracy", "latency_ms", "service_availability", "attacker_stage",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            ts = m["timeseries"]
            for i in range(len(ts["steps"])):
                writer.writerow({
                    "episode": m["episode"],
                    "step": ts["steps"][i],
                    "action": ts["actions"][i],
                    "reward": round(ts["rewards"][i], 4),
                    "path_entropy": round(ts["path_entropy"][i], 4),
                    "attacker_entropy": round(ts["attacker_entropy"][i], 4),
                    "attacker_knowledge": round(ts["attacker_knowledge"][i], 4),
                    "recon_accuracy": round(ts["recon_accuracy"][i], 4),
                    "latency_ms": round(ts["latency_ms"][i], 2),
                    "service_availability": round(ts["service_availability"][i], 4),
                    "attacker_stage": ts["attacker_stage"][i],
                })
    print(f"  Saved: {path}")


def generate_plots(results_dir, all_summaries, all_results):
    """Generate matplotlib plots for the paper."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed. Skipping plots.")
        return

    strategies = [s["strategy"] for s in all_summaries]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"][:len(strategies)]

    # --- Figure 1: Attack Success Rate (Bar) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    asrs = [s["ASR"] for s in all_summaries]
    bars = ax.bar(strategies, asrs, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=12)
    ax.set_title("Attack Success Rate by MTD Strategy", fontsize=13)
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2%}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig1_attack_success_rate.png"), dpi=150)
    plt.close()

    # --- Figure 2: Mean Path Entropy (Bar) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ents = [s["mean_path_entropy"] for s in all_summaries]
    bars = ax.bar(strategies, ents, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Mean Path Entropy (H_P)", fontsize=12)
    ax.set_title("Network Unpredictability by MTD Strategy", fontsize=13)
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, ents):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig2_path_entropy.png"), dpi=150)
    plt.close()

    # --- Figure 3: Security vs Performance Trade-off (Scatter) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(all_summaries):
        ax.scatter(s["mean_latency_ms"], 1.0 - s["ASR"],
                   s=200, c=colors[i], label=s["strategy"],
                   edgecolors="black", linewidth=0.8, zorder=3)
    ax.set_xlabel("Mean Latency (ms)", fontsize=12)
    ax.set_ylabel("Defense Effectiveness (1 - ASR)", fontsize=12)
    ax.set_title("Security vs Performance Trade-off", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig3_security_vs_performance.png"), dpi=150)
    plt.close()

    # --- Figure 4: Attacker Stage Progression Over Time (Line) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, metrics) in enumerate(all_results.items()):
        # Average attacker stage across episodes at each step
        max_steps = max(len(m["timeseries"]["attacker_stage"]) for m in metrics)
        avg_stages = np.zeros(max_steps)
        counts = np.zeros(max_steps)
        for m in metrics:
            stages = m["timeseries"]["attacker_stage"]
            avg_stages[:len(stages)] += stages
            counts[:len(stages)] += 1
        counts[counts == 0] = 1
        avg_stages /= counts
        ax.plot(range(max_steps), avg_stages, label=name, color=colors[i], linewidth=2)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Mean Attacker Stage", fontsize=12)
    ax.set_title("Attacker Progression Over Time", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["Recon", "Initial\nAccess", "Lateral\nMovement", "Priv\nEscalation", "Exfiltration"])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig4_attacker_progression.png"), dpi=150)
    plt.close()

    # --- Figure 5: Cumulative Reward (Line) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, metrics) in enumerate(all_results.items()):
        max_steps = max(len(m["timeseries"]["rewards"]) for m in metrics)
        avg_cum_reward = np.zeros(max_steps)
        counts = np.zeros(max_steps)
        for m in metrics:
            rewards = np.cumsum(m["timeseries"]["rewards"])
            avg_cum_reward[:len(rewards)] += rewards
            counts[:len(rewards)] += 1
        counts[counts == 0] = 1
        avg_cum_reward /= counts
        ax.plot(range(max_steps), avg_cum_reward, label=name, color=colors[i], linewidth=2)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.set_title("Defender Cumulative Reward Over Time", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig5_cumulative_reward.png"), dpi=150)
    plt.close()

    # --- Figure 6: Metrics Comparison Radar / grouped bar ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Latency
    axes[0].bar(strategies, [s["mean_latency_ms"] for s in all_summaries],
                color=colors, edgecolor="black")
    axes[0].set_ylabel("Mean Latency (ms)")
    axes[0].set_title("End-to-End Latency")
    # Service Availability
    axes[1].bar(strategies, [s["mean_service_availability"] for s in all_summaries],
                color=colors, edgecolor="black")
    axes[1].set_ylabel("Service Availability")
    axes[1].set_title("Service Availability")
    axes[1].set_ylim(0.5, 1.05)
    # Mutations
    axes[2].bar(strategies, [s["mean_mutations"] for s in all_summaries],
                color=colors, edgecolor="black")
    axes[2].set_ylabel("Mean Mutations / Episode")
    axes[2].set_title("Controller Overhead")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig6_performance_metrics.png"), dpi=150)
    plt.close()

    print(f"  All plots saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MTD strategies")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument("--model-dir", type=str, default="../models")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip DQN agent (baselines only)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    env = MTDPlaygroundEnv(seed=args.seed)

    # Define strategies
    strategies = {
        "No MTD": strategy_no_mtd,
        "Random MTD": strategy_random,
        "Periodic MTD": lambda obs, step: strategy_periodic(obs, step, period=10),
    }

    # Load DQN model if available
    if not args.no_model:
        try:
            from stable_baselines3 import DQN
            model_path = os.path.join(args.model_dir, "best_model.zip")
            if not os.path.exists(model_path):
                model_path = os.path.join(args.model_dir, "dqn_mtd_final.zip")
            if os.path.exists(model_path):
                model = DQN.load(model_path, env=env)
                strategies["DQN Adaptive MTD"] = make_dqn_strategy(model)
                print(f"Loaded DQN model from {model_path}")
            else:
                print(f"No model found at {model_path}. Running baselines only.")
        except ImportError:
            print("stable-baselines3 not installed. Running baselines only.")

    # Run evaluations
    all_results = {}
    all_summaries = []

    for name, fn in strategies.items():
        print(f"\nEvaluating: {name} ({args.episodes} episodes)...")
        metrics = run_strategy(env, fn, args.episodes, args.seed)
        all_results[name] = metrics
        summary = compute_summary(name, metrics)
        all_summaries.append(summary)

        # Print summary
        print(f"  ASR={summary['ASR']:.2%}  "
              f"ACT={summary['mean_ACT']:.0f}  "
              f"H_P={summary['mean_path_entropy']:.3f}  "
              f"Latency={summary['mean_latency_ms']:.1f}ms  "
              f"Avail={summary['mean_service_availability']:.3f}  "
              f"Mutations={summary['mean_mutations']:.0f}")

        save_detailed_csv(args.results_dir, name.replace(" ", "_").lower(), metrics)

    # Save summary CSV
    summary_path = os.path.join(args.results_dir, "summary_comparison.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\nSummary saved to {summary_path}")

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(args.results_dir, all_summaries, all_results)

    # Print final comparison table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Strategy':<22} {'ASR':>6} {'ACT':>6} {'H_P':>6} {'Latency':>8} {'Avail':>7} {'Mut':>5} {'Reward':>8}"
    print(header)
    print("-" * 80)
    for s in all_summaries:
        print(f"{s['strategy']:<22} {s['ASR']:>6.2%} {s['mean_ACT']:>6.0f} "
              f"{s['mean_path_entropy']:>6.3f} {s['mean_latency_ms']:>7.1f}ms "
              f"{s['mean_service_availability']:>6.3f} {s['mean_mutations']:>5.0f} "
              f"{s['mean_total_reward']:>8.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
