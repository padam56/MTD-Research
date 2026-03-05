# Eghtesad et al. (2020) — Adversarial Deep RL-based Adaptive Moving Target Defense

## Citation
```bibtex
@inproceedings{eghtesad2020adversarial,
  title={Adversarial Deep Reinforcement Learning based Adaptive Moving Target Defense},
  author={Eghtesad, Tohid and Nakhodchi, Soheil and Panda, Aron},
  booktitle={GameSec 2020: Decision and Game Theory for Cybersecurity},
  year={2020},
  doi={10.1007/978-3-030-64793-3_20}
}
```

**DOI:** https://doi.org/10.1007/978-3-030-64793-3_20

---

## Core Contributions

### 1. Markov Game Formulation for MTD
Two-player adversarial game:
- **Defender (D):** Network administrator, controls MTD mutations
- **Attacker (A):** Adversary trying to learn network topology

**State Space:** $S = (s_{\text{net}}, s_{\text{attacker}})$
- $s_{\text{net}}$: Network configuration (IP/port mappings, routing)
- $s_{\text{attacker}}$: Attacker's current knowledge of topology

**Action Space (Defender):** $A_D = \{0, 1, 2\}$
- Action 0: No mutation (maintain current paths)
- Action 1: Moderate MTD (shuffle some paths)
- Action 2: Aggressive MTD (full network re-randomization)

**Observation Space (Defender receives):**
- Network threat level $\tau_t$ (from ML detector)
- Attacker knowledge entropy $H_A$ (estimate of topology leakage)
- Path entropy $H_P$ (network randomness)
- Traffic statistics (bytes/s, packet count, etc.)

### 2. Cost-of-Moving Model
Zero-sum game payoff (Defender's utility):

$$R_D = R_{\text{detection}} - c_m \cdot |A_D| - c_a \cdot H_A$$

Where:
- $R_{\text{detection}}$: Reward for detecting intrusions (+1 if attack detected, -1 if attack missed)
- $c_m$: Deception cost (network overhead from mutations)
- $|A_D|$: Severity of MTD action (0 ≤ |A_D| ≤ 2)
- $c_a$: Cost of attacker knowledge exposure
- $H_A$: Attacker knowledge entropy (higher = attacker less knowledgeable)

Attacker's payoff: $R_A = -R_D$ (zero-sum)

### 3. Threat Detection Signal
Binary classification from ML ensemble:
$$\tau_t = \begin{cases} 1 & \text{if } P(\text{intrusion detected}) > \theta \\ 0 & \text{otherwise} \end{cases}$$

Threat **probability** (for RL):
$$\text{threat\_prob}_t = P(\text{label}=\text{attack} | x_t)$$

Converted to threat signal:
$$\text{threat} = \begin{cases} \text{confidence} & \text{if } \text{prediction}==\text{attack} \\ 1 - \text{confidence} & \text{otherwise} \end{cases}$$

### 4. Entropies Measuring MTD Effectiveness

**Path Entropy** — Randomness of egress port distribution:
$$H_P = -\sum_{p} P(p) \log_2 P(p)$$
Where $P(p)$ = fraction of flows exiting via port $p$

**Attacker Knowledge Entropy** — Uncertainty in observed network topology:
$$H_A = -\sum_{i} P(\text{path}_i | \text{observed}) \log_2 P(\text{path}_i | \text{observed})$$
- High $H_A$ vs low $H_A$ indicates attacker has minimal vs extensive knowledge
- RL goal: Keep $H_A$ high (attacker can't learn paths)

---

## Implementation Components

### Reference-Only (Offline Simulator)
- **Environment:** `MTDGameEnv` (Markov game simulator)
- **State representation:** 14 features (threat, entropies, traffic)
- **Threat detector:** 3-regime ensemble (RF/XGBoost/IsolationForest)
- **RL agent:** Double DQN with Dueling architecture (Stable-Baselines3)
- **Data source:** Synthetic flows (CIC-IDS2017 format) from `MockONOSClient`

### Mininet-Ready (Real Network Integration)
- **ONOS Controller:** Interface to query topology, receive threat signals
- **OpenFlow actions:** Group table mutations for IP/port randomization
- **Threat detector:** Real packet capture analysis (optional)
- **RL agent:** Real-time decision loop (≤100ms latency)
- **Data source:** Live network traffic from Mininet topology

---

## Key Equations & Variables

| Symbol | Meaning | Range | Paper Section |
|--------|---------|-------|---|
| $H_P$ | Path Entropy | [0, log(|ports|)] | Eq. (2) |
| $H_A$ | Attacker Knowledge Entropy | [0, log(|paths|)] | Eq. (3) |
| $\tau_t$ | Threat indicator | {0, 1} | Eq. (4) |
| $c_m$ | Deception cost parameter | (0, 1) | Eq. (6) |
| $Q(s,a)$ | RL Q-value | ℝ | Eq. (8) |
| $\gamma$ | RL discount factor | 0.99 | Training |

---

## Experimental Results (Expected)

### Offline Simulator
- **Episodes:** 10–50
- **Timesteps/episode:** 200–500
- **Metrics:** Cumulative reward, threat detection rate, path entropy evolution
- **Baseline:** Random MTD actions vs learned policy

### Mininet (Optional)
- **Topology:** 10–50 hosts, 5–10 switches
- **Attack types:** Port scanning, DDoS, exfiltration
- **Metrics:** Detection latency, false positive rate, network overhead

---

## References & Related Work

1. **Original Paper:** Eghtesad, Nakhodchi, Panda (2020) GameSec
2. **RL Algorithms:**
   - van Hasselt et al. (2016) — Double Q-learning
   - Wang et al. (2016) — Dueling networks
3. **ML Detection:** Li et al. (2021) — HybridMTD ensemble
4. **Explainability:** Lundberg & Lee (2017) — SHAP
5. **Metrics:** Chowdhary et al. (2020) — Path entropy validation

---

## File Organization

```
eghtesad2020/
├── PAPER.md (this file)
├── reference-only/
│   ├── README.md
│   ├── run.py (main entry point)
│   ├── mtd_env.py (Markov game environment)
│   ├── threat_detector.py (ML ensemble)
│   ├── rl_agent.py (Double DQN trainer)
│   └── utils.py (entropy, payoff calculations)
└── mininet-ready/
    ├── README.md
    ├── onos_controller_interface.py
    ├── openflow_mtd_actions.py
    ├── mininet_topology.py
    ├── threat_detector_live.py
    └── run_mininet.py
```
