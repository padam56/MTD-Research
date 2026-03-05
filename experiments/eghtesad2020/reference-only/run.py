#!/usr/bin/env python3
"""
Eghtesad (2020) — Reference-Only Implementation
Offline Markov game simulator for adaptive MTD learning
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from mtd_env import SDN_MTD_Env
from threat_detector import ThreatDetector
from rl_agent import train_rl_agent
from utils import generate_synthetic_data, plot_results


def train_ml_detector(dataset_path: str) -> ThreatDetector:
    """Train ML threat detector on synthetic dataset."""
    print(f"[1] Training ML detector on {dataset_path}...")
    detector = ThreatDetector()
    detector.train(dataset_path)
    print(f"    ✓ Detector trained (RF + XGBoost + IsolationForest)")
    return detector


def train_and_evaluate(
    detector: ThreatDetector,
    episodes: int = 10,
    timesteps_per_episode: int = 200,
    output_dir: str = "./results",
    seed: int = 42
) -> pd.DataFrame:
    """Train RL agent on environment and evaluate."""
    print(f"\n[2] Training RL agent ({episodes} episodes × {timesteps_per_episode} steps)...")
    
    # Create environment
    env = SDN_MTD_Env(threat_detector=detector, seed=seed)
    
    # Train agent
    model = train_rl_agent(env, total_timesteps=episodes * timesteps_per_episode)
    
    # Evaluation loop
    print(f"\n[3] Evaluating agent...")
    os.makedirs(output_dir, exist_ok=True)
    
    results_list = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(timesteps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Log step metrics
            results_list.append({
                'episode': ep + 1,
                'step': step + 1,
                'total_step': ep * timesteps_per_episode + step + 1,
                'threat_level': info.get('threat_level', 0.0),
                'action': int(action),
                'reward': reward,
                'episode_reward': episode_reward,
                'path_entropy': info.get('path_entropy', 0.0),
                'attacker_entropy': info.get('attacker_entropy', 0.0),
                'deception_cost': info.get('deception_cost', 0.0),
                'detection_success': 1 if info.get('threat_level', 0) > 0.5 and action > 0 else 0,
                'shap_top_feature': info.get('shap_top_feature', 'none'),
                'network_phase': info.get('phase', 'unknown'),
            })
            
            if terminated or truncated:
                break
        
        print(f"  Episode {ep+1:2d}/{episodes}: Reward = {episode_reward:6.1f}, "
              f"Threat = {np.mean([r['threat_level'] for r in results_list[-timesteps_per_episode:]]):.3f}")
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_csv = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n[4] Results saved: {results_csv}")
    print(f"    Total steps: {len(results_df)}")
    print(f"    Mean threat: {results_df['threat_level'].mean():.3f}")
    print(f"    Mean action: {results_df['action'].mean():.2f}")
    print(f"    Action dist: {{0: {(results_df['action']==0).sum()/len(results_df)*100:.1f}%, "
          f"1: {(results_df['action']==1).sum()/len(results_df)*100:.1f}%, "
          f"2: {(results_df['action']==2).sum()/len(results_df)*100:.1f}%}}")
    print(f"    Cumulative reward: {results_df['episode_reward'].max():.1f}")
    
    env.close()
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Eghtesad (2020) — Offline MTD RL Training"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of training episodes (default: 10)"
    )
    parser.add_argument(
        "--timesteps-per-episode", type=int, default=200,
        help="Steps per episode (default: 200)"
    )
    parser.add_argument(
        "--ml-samples", type=int, default=20000,
        help="Synthetic samples for ML training (default: 20000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Output directory for results (default: ./results)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Eghtesad (2020) — Adaptive MTD via Adversarial Deep RL")
    print("=" * 70)
    
    # Generate synthetic data
    print(f"[0] Generating {args.ml_samples} synthetic flows...")
    dataset_path = generate_synthetic_data(args.ml_samples, seed=args.seed)
    print(f"    ✓ Dataset: {dataset_path}")
    
    # Train ML detector
    detector = train_ml_detector(dataset_path)
    
    # Train RL agent and evaluate
    results_df = train_and_evaluate(
        detector=detector,
        episodes=args.episodes,
        timesteps_per_episode=args.timesteps_per_episode,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Generate plots
    print(f"\n[5] Generating evaluation figures...")
    plot_results(os.path.join(args.output_dir, "evaluation_results.csv"), args.output_dir)
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"  Results: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
