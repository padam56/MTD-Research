"""
Utility Functions — Data generation, plotting, entropy calculations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


def generate_synthetic_data(n_samples: int = 20000, seed: int = 42) -> str:
    """
    Generate synthetic CIC-IDS2017 format flow data.
    
    Returns:
        Path to CSV file
    """
    np.random.seed(seed)
    
    print(f"  Generating {n_samples} synthetic flows...")
    
    flows = []
    
    # 70% benign, 30% attack
    n_benign = int(n_samples * 0.7)
    n_attack = n_samples - n_benign
    
    # Benign flows
    for _ in range(n_benign):
        flows.append({
            'duration': np.random.exponential(5),
            'bytes': np.random.exponential(10000),
            'packets': np.random.exponential(100),
            'syn_flags': np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]),
            'fin_flags': np.random.choice([0, 1], p=[0.8, 0.2]),
            'rst_flags': np.random.choice([0, 1], p=[0.95, 0.05]),
            'ack_flags': np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2]),
            'bytes_per_sec': np.random.exponential(5000),
            'packets_per_sec': np.random.exponential(100),
            'protocol_tcp': np.random.choice([0, 1], p=[0.4, 0.6]),
            'protocol_udp': np.random.choice([0, 1], p=[0.7, 0.3]),
            'src_port_entropy': np.random.uniform(0, 10),
            'dst_port_entropy': np.random.uniform(0, 10),
            'flag_rate': np.random.exponential(0.1),
            'tcp_fraction': np.random.uniform(0.3, 0.8),
            'udp_fraction': np.random.uniform(0.1, 0.6),
            'icmp_fraction': np.random.uniform(0, 0.1),
            'flow_iat_mean': np.random.exponential(0.1),
            'flow_iat_std': np.random.exponential(0.01),
            'fwd_packets_mean': np.random.exponential(50),
            'bwd_packets_mean': np.random.exponential(50),
            'src_ports_unique': np.random.randint(1, 5),
            'dst_ports_unique': np.random.randint(1, 3),
            'packet_size_mean': np.random.uniform(64, 1500),
            'packet_size_std': np.random.uniform(10, 500),
            'label': 0,
        })
    
    # Attack flows (DDoS, scanning, etc.)
    for _ in range(n_attack):
        flows.append({
            'duration': np.random.exponential(0.5),  # Shorter
            'bytes': np.random.exponential(50000),  # Larger
            'packets': np.random.exponential(500),  # More packets
            'syn_flags': np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3]),  # HIGH
            'fin_flags': np.random.choice([0, 1], p=[0.6, 0.4]),
            'rst_flags': np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3]),  # HIGH
            'ack_flags': np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3]),
            'bytes_per_sec': np.random.exponential(50000),  # Much higher
            'packets_per_sec': np.random.exponential(1000),  # Much higher
            'protocol_tcp': np.random.choice([0, 1], p=[0.2, 0.8]),
            'protocol_udp': np.random.choice([0, 1], p=[0.5, 0.5]),
            'src_port_entropy': np.random.uniform(0, 3),  # Lower diversity
            'dst_port_entropy': np.random.uniform(5, 10),  # High diversity
            'flag_rate': np.random.exponential(1.0),  # HIGH
            'tcp_fraction': np.random.uniform(0.7, 0.95),
            'udp_fraction': np.random.uniform(0, 0.3),
            'icmp_fraction': np.random.uniform(0, 0.1),
            'flow_iat_mean': np.random.exponential(0.01),  # Very low
            'flow_iat_std': np.random.exponential(0.001),
            'fwd_packets_mean': np.random.exponential(200),  # HIGH
            'bwd_packets_mean': np.random.exponential(50),
            'src_ports_unique': 1,
            'dst_ports_unique': np.random.randint(5, 20),
            'packet_size_mean': np.random.uniform(64, 256),
            'packet_size_std': np.random.uniform(5, 100),
            'label': 1,
        })
    
    df = pd.DataFrame(flows)
    
    # Save to CSV
    output_dir = Path("./data")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "synthetic_flows.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  ✓ Saved {len(df)} flows to {csv_path}")
    print(f"    - Benign: {n_benign} ({n_benign/n_samples*100:.1f}%)")
    print(f"    - Attack: {n_attack} ({n_attack/n_samples*100:.1f}%)")
    
    return str(csv_path)


def plot_results(results_csv: str, output_dir: str = "./results"):
    """
    Generate evaluation plots from results CSV.
    
    Creates:
    1. cumulative_reward.png
    2. threat_vs_action.png
    3. attacker_entropy.png
    4. payoff_breakdown.png
    5. shap_features.png
    """
    
    df = pd.read_csv(results_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Common settings
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 10
    
    # === Plot 1: Cumulative Reward per Episode ===
    fig, ax = plt.subplots()
    episodes = df.groupby('episode')['episode_reward'].max()
    ax.plot(episodes.index, episodes.values, marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Eghtesad MTD — Cumulative Reward per Episode')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_reward.png'), dpi=300)
    print(f"  ✓ {os.path.join(output_dir, 'cumulative_reward.png')}")
    plt.close()
    
    # === Plot 2: Threat Level vs Action Taken ===
    fig, ax = plt.subplots()
    actions = df['action'].unique()
    colors = {0: 'green', 1: 'orange', 2: 'red'}
    for action in sorted(actions):
        mask = df['action'] == action
        ax.scatter(df[mask]['total_step'], df[mask]['threat_level'], 
                  label=f'Action {int(action)}', alpha=0.6, s=20, color=colors.get(action))
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Threat Level')
    ax.set_title('Threat Level vs RL Action Taken')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threat_vs_action.png'), dpi=300)
    print(f"  ✓ {os.path.join(output_dir, 'threat_vs_action.png')}")
    plt.close()
    
    # === Plot 3: Attacker Knowledge vs Path Entropy ===
    fig, ax = plt.subplots()
    ax.plot(df['total_step'], df['attacker_entropy'], label='H_A (Attacker Entropy)', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(df['total_step'], df['path_entropy'], label='H_P (Path Entropy)', 
            color='orange', linewidth=2)
    ax.set_xlabel('Global Step')
    ax.set_ylabel('H_A', color='tab:blue')
    ax2.set_ylabel('H_P', color='orange')
    ax.set_title('Network Entropy Metrics During Training')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attacker_entropy.png'), dpi=300)
    print(f"  ✓ {os.path.join(output_dir, 'attacker_entropy.png')}")
    plt.close()
    
    # === Plot 4: Payoff Breakdown ===
    fig, ax = plt.subplots()
    df['detection_reward'] = df['detection_success'].rolling(window=20, min_periods=1).mean()
    df['cost_moving'] = df['deception_cost'].rolling(window=20, min_periods=1).mean()
    ax.plot(df['total_step'], df['detection_reward'], label='Detection Reward', linewidth=2)
    ax.plot(df['total_step'], -df['cost_moving'], label='Cost (Negative)', linewidth=2)
    ax.fill_between(df['total_step'], 0, df['reward'].rolling(window=20, min_periods=1).mean(),
                    alpha=0.2, label='Net Reward')
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Reward Value')
    ax.set_title('Reward Components Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'payoff_breakdown.png'), dpi=300)
    print(f"  ✓ {os.path.join(output_dir, 'payoff_breakdown.png')}")
    plt.close()
    
    # === Plot 5: SHAP Feature Importance ===
    fig, ax = plt.subplots()
    features = df['shap_top_feature'].value_counts()
    features = features.head(8)  # Top 8 features
    ax.barh(range(len(features)), features.values, color='steelblue')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features.index)
    ax.set_xlabel('Frequency (steps where feature was top SHAP)')
    ax.set_title('Top SHAP Features Driving Threat Predictions')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_features.png'), dpi=300)
    print(f"  ✓ {os.path.join(output_dir, 'shap_features.png')}")
    plt.close()
    
    print(f"\n  All plots saved to {output_dir}/")
