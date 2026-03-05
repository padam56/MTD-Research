# Eghtesad (2020) — Reference-Only Implementation (Offline Simulator)

## Overview

This is a **self-contained, offline implementation** of the Eghtesad et al. (2020) Markov game for adaptive MTD learning. It requires **no real network, no ONOS, no Mininet** — just Python and the synthetic data generator.

**Perfect for:** Research paper reproduction, hyperparameter tuning, baseline comparison.

---

## Architecture

```
Synthetic Flow Generator (MockONOSClient)
         ↓ (CIC-IDS2017 format flows)
Threat Detector (3-regime ensemble)
         ↓ (threat probability + SHAP explanations)
RL Agent (Double DQN)
         ↓ (selects action 0/1/2)
Environment Reward Calculation
         ↓ (zero-sum payoff)
Training Loop (50,000 timesteps, 10 episodes)
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `stable-baselines3` — Double DQN + Dueling networks
- `xgboost` — ML threat detector
- `scikit-learn` — Random Forest, IsolationForest
- `shap>=0.46` — Explainability
- `numpy`, `pandas`, `matplotlib`

### 2. Run Training & Evaluation
```bash
python run.py \
  --episodes 10 \
  --timesteps-per-episode 200 \
  --ml-samples 20000 \
  --seed 42 \
  --output-dir ./results
```

**Arguments:**
- `--episodes`: Number of training episodes (default: 10)
- `--timesteps-per-episode`: Steps per episode (default: 200)
- `--ml-samples`: Synthetic flow samples for ML training (default: 20000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Where to save results CSV + figures (default: ./results)

### 3. Outputs
```
results/
├── evaluation_results.csv (2000 rows: threat, entropy, action, reward, etc.)
├── cumulative_reward.png
├── threat_vs_action.png
├── attacker_entropy.png
├── payoff_breakdown.png
└── shap_features.png
```

---

## File Descriptions

### `run.py`
**Main entry point.** Orchestrates:
1. Synthetic data generation (20k flows)
2. ML threat detector training (3 regimes)
3. RL agent training (10 episodes × 200 steps)
4. Evaluation & metric logging (CSV + 5 figures)

**Key functions:**
- `main()` — Argument parsing + orchestration
- `train_and_evaluate()` — Run full pipeline

### `mtd_env.py`
**Markov Game Environment** (`SDN_MTD_Env` class)

Implements:
- State space: `[threat, H_A, H_P, traffic_bytes, packet_count, ..., 13 features]`
- Action space: Discrete(3) = {0=no-move, 1=moderate, 2=aggressive}
- Reward: Zero-sum payoff $R_D = R_{\text{detect}} - c_m \cdot |A_D|$
- Episode phases: normal → recon → attack → post-attack

**Key methods:**
- `step(action)` — Execute action, update flows, compute reward
- `reset()` — Start new episode
- `_compute_path_entropy()` — Shannon entropy of egress ports
- `_compute_attacker_knowledge_entropy()` — Measure topology leakage
- `_zero_sum_payoff()` — Calculate reward

### `threat_detector.py`
**ML Threat Detector** (`ThreatDetector` class)

3-regime switching ensemble:
- **Regime A (low traffic):** Random Forest
- **Regime B (high traffic):** XGBoost
- **Regime C (anomaly spike):** IsolationForest

**Key methods:**
- `train(csv_path)` — Fit ensemble on labeled flows
- `predict(flow_dict)` — Binary attack classification → threat probability
- `explain_prediction(flow_dict)` — SHAP values for each feature
- `get_threat_score()` — Convert confidence to [0,1] threat signal

**Features used (25 total):**
- Duration, Bytes/s, Packets/s
- SYN/FIN/RST/ACK flag means (per-flow)
- Protocol distribution (TCP/UDP/ICMP fractions)
- Port entropy

### `rl_agent.py`
**Reinforcement Learning Agent Trainer**

Uses Stable-Baselines3 Double DQN with Dueling architecture.

**Key functions:**
- `train_rl_agent(env, total_timesteps)` — Train agent on environment
- Returns trained PPO/DQN model + training logs

**Hyperparameters:**
- Algorithm: `DQN` (Double Q-learning built-in)
- Learning rate: 1e-3
- Target network update interval: 500 steps
- Tau (Polyak averaging): 0.005
- Epsilon (exploration): Scheduled decay

### `utils.py`
**Helper functions:**
- `generate_synthetic_data(n_samples)` — Create flows in CIC-IDS2017 format
- `compute_entropy(values)` — Shannon entropy
- `compute_zero_sum_payoff(threat, action, H_A)` — Reward calculation
- `plot_results(csv_path)` — Generate 5 evaluation figures

---

## Key Equations Implemented

### Path Entropy
```python
def _compute_path_entropy(self):
    port_dist = self.flows['egress_port'].value_counts() / len(self.flows)
    return -sum(port_dist * np.log2(port_dist))
```
Measures randomness of egress port distribution (0 = all same port, log(n) = uniform).

### Attacker Knowledge Entropy
```python
def _compute_attacker_knowledge_entropy(self):
    unique_paths = self.flows['path'].nunique()
    total_paths = len(self.flows)
    # Simplified: H_A ≈ log2(unique_paths / total_paths)
    return np.log2(max(unique_paths, 1)) - np.log2(total_paths)
```

### Zero-Sum Payoff
```python
reward_detect = 1.0 if threat > 0.5 else -0.1
deception_cost = 0.1 * action  # Cost increases with action severity
reward = reward_detect - deception_cost
```

---

## Typical Results (From Previous 50k-timestep Run)

```
Episode 1-10 Metrics (averaged):
  Mean threat level:      0.52 ± 0.08
  Path entropy (H_P):     3.21 ± 0.45 bits
  Attacker entropy (H_A): 0.99 ± 0.01
  Action distribution:    {0: 8.6%, 1: 58.8%, 2: 32.6%}
  SHAP firing rate:       91% (1828/2000 steps)
  Cumulative reward:      ≈200–350 per episode

Top 3 threat-driving features (SHAP):
  1. Flow Duration
  2. Bytes/s
  3. SYN flag count
```

---

## Extending This Implementation

### 1. Add Real CIC-IDS2017 Data
Modify `run.py`:
```python
# Instead of generate_synthetic_data():
dataset_path = "/path/to/CIC-IDS2017.csv"
train_ml_detector_real(dataset_path)
```

### 2. Change RL Algorithm
In `rl_agent.py`:
```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, learning_rate=1e-3)
```

### 3. Custom MTD Costs
In `mtd_env.py`:
```python
deception_cost = 0.2 * action + 0.05 * (action ** 2)  # Quadratic penalty
```

### 4. Different Threat Scenarios
Modify `MockONOSClient` to vary:
- Attack phase intensity (higher DDoS traffic)
- Attacker sophistication (learns faster)
- MTD mutation frequency (budget constraints)

---

## Troubleshooting

### SHAP not firing?
- Update to `shap>=0.46`: `pip install --upgrade shap`
- Check that feature shapes match threat detector output

### Training diverges (reward NaN)?
- Reduce learning rate: `learning_rate=1e-4`
- Clip rewards: `reward = np.clip(reward, -1, 1)`
- Reset after episodes (ensure `env.reset()` called)

### Memory issues with large ML training set?
- Reduce `--ml-samples` to 5000–10000
- Use `train_ml_detector_real()` with real data (more efficient)

---

## Paper Reference
See `../PAPER.md` for full citations, equations, and related work.
