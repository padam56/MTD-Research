"""
RL Agent Trainer — Double DQN with Dueling Architecture
Uses Stable-Baselines3 for training
"""

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np


def train_rl_agent(env, total_timesteps: int = 100000, model_path: str = None):
    """
    Train RL agent using Double DQN with Dueling networks.
    
    Args:
        env: Gymnasium environment
        total_timesteps: Total training steps
        model_path: Optional path to save model
        
    Returns:
        Trained DQN model
    """
    
    print(f"\n[RL] Initializing Double DQN agent...")
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=100,
        batch_size=32,
        tau=0.005,  # Polyak averaging for target network
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={
            "net_arch": [512, 256],  # Dueling architecture
            "dueling": True,
            "dueling_net_arch": ([256], [256]),
        },
        verbose=0,
        device="cpu",
    )
    
    print(f"    - Policy: MlpPolicy with Dueling architecture")
    print(f"    - Learning rate: 1e-3")
    print(f"    - Target update interval: 500 steps")
    print(f"    - Exploration: epsilon-decay from 1.0 → 0.05")
    print(f"    - Training for {total_timesteps} timesteps...")
    
    # Train
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    
    # Save if path provided
    if model_path:
        model.save(model_path)
        print(f"    ✓ Model saved to {model_path}")
    
    return model
