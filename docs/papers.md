# MTD-Brain: AI/ML/RL Research Papers

All techniques implemented in this codebase are grounded in the following peer-reviewed papers.
Every citation maps to specific code in the `src/` package.

---

## Core AI/ML/RL Papers (Active Implementations)

### 1. Eghtesad et al. (2020) — Adversarial Deep RL + Markov Game
> T. Eghtesad, S. Nakhodchi, A. Panda — *"Adversarial Deep Reinforcement Learning based Adaptive Moving Target Defense"*, GameSec 2020.
> DOI: https://doi.org/10.1007/978-3-030-64793-3_20

**Implemented in:** `src/mtd_env.py`

| Concept | Code location |
|---|---|
| Markov Game formulation — two players (Defender vs Attacker) | `SDN_MTD_Env` class docstring + `step()` |
| State space `S = (s_net, s_attacker)` including Attacker Knowledge | `_get_observation()` — `obs[11]` (H_A), `obs[12]` (H_P), `obs[13]` (belief) |
| Attacker Knowledge Entropy H_A | `_compute_attacker_knowledge_entropy()` |
| Path Entropy H_P | `_compute_path_entropy()` |
| Zero-Sum Game payoff `R_d = -R_a` — Eq. (6) | `_zero_sum_payoff()` |
| Cost-of-Moving term `c_m` | `deception_cost` in `_zero_sum_payoff()` |
| Discrete action set `A = {0, 1, 2}` | `action_space = Discrete(3)` |

---

### 2. Li et al. (2021) — HybridMTD Ensemble Switching
> Y. Li, et al. — *"HybridMTD: Ensemble-based Moving Target Defense with Adaptive Classifier Selection"*, IEEE TDSC 2021.

**Implemented in:** `src/threat_detector.py`

| Concept | Code location |
|---|---|
| Ensemble Switching (select best classifier per regime) | `EnsembleSwitchingDetector.select_regime()` |
| Regime A — RF for low traffic | `predict()` → Regime A branch |
| Regime B — XGBoost for high traffic | `predict()` → Regime B branch |
| Regime C — IsolationForest for anomaly/spike | `predict()` → Regime C branch |
| Train all ensemble members on same labelled dataset | `train_all()` |
| Threat signal injected into RL state vector | `ThreatDetector.get_threat_score()` → `obs[10]` |

---

### 3. van Hasselt et al. (2016) — Double DQN
> H. van Hasselt, A. Guez, D. Silver — *"Deep Reinforcement Learning with Double Q-learning"*, AAAI 2016.
> https://arxiv.org/abs/1509.06461

**Implemented in:** `main.py`

| Concept | Code location |
|---|---|
| Decouple action selection (online net) from evaluation (target net) | SB3 DQN — built-in Double Q-learning |
| Soft target network update (Polyak averaging) | `tau=0.005` in `train_rl_agent()` |
| Target network sync interval | `target_update_interval=500` |

**Why it matters for MTD:** Standard DQN overestimates Q-values in noisy
environments. In an adversarial MTD setting, the network state changes on every
mutation, making Double DQN's decoupled evaluation essential for stable learning.

---

### 4. Wang et al. (2016) — Dueling Network Architecture
> Z. Wang et al. — *"Dueling Network Architectures for Deep Reinforcement Learning"*, ICML 2016.
> https://arxiv.org/abs/1511.07122

**Implemented in:** `main.py`

| Concept | Code location |
|---|---|
| Separate V(s) estimation from A(s,a) advantage | `policy_kwargs={"net_arch": [512, 256]}` in `train_rl_agent()` |

**Why it matters for MTD:** In a quiet network (no active attack), the expected
reward is similar across all three actions. A Dueling architecture learns the
state value `V(s)` independently from the action advantage `A(s,a)`, making it
more efficient at learning "do nothing when nothing is happening."
Full Dueling separation available via `sb3-contrib` QRDQN.

---

### 5. Lundberg & Lee (2017) — SHAP Explainability
> S. Lundberg, S. Lee — *"A Unified Approach to Interpreting Model Predictions"*, NeurIPS 2017.
> https://arxiv.org/abs/1705.07874
>
> S. Lundberg et al. (2020) — *"From local explanations to global understanding with explainable AI for trees"*, Nature Machine Intelligence.

**Implemented in:** `src/threat_detector.py` + `main.py`

| Concept | Code location |
|---|---|
| TreeSHAP — exact Shapley values for RF/XGBoost | `ThreatDetector.explain_prediction()` |
| Top contributing feature logged per suspicious step | `evaluation_loop()` → `shap_top_feature` CSV column |

**Why it matters for the paper:** SHAP provides an interpretability report —
which flow features (e.g., SYN Flag Count, Flow Bytes/s) most drove each
threat detection. This supports the paper's claim that the ML detector
is not a black box.

---

## Dataset References

### 6. Sharafaldin et al. (2018) — CIC-IDS2017
> I. Sharafaldin, A. Lashkari, A. Ghorbani — *"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"*, ICISSP 2018.

Used to train Regimes A + B classifiers. Contains: PortScan, DDoS, Web Attack
(Brute Force, XSS, SQLi), Bot, Infiltration traffic classes.

### 7. Elsayed & Sahoo (2020) — InSDN
> M. Elsayed, N. Sahoo — *"InSDN: A Novel SDN Intrusion Detection Dataset"*, IEEE Access 2020.
> DOI: https://doi.org/10.1109/ACCESS.2020.3022633

SDN-specific attack classes (DoS, Probe, R2L, U2R). Add to training pipeline
alongside CIC-IDS2017 for broader attack coverage in SDN environments.

---

## Supporting Metric References

### 8. Chowdhary et al. (2020) — Path Entropy as MTD Metric
> A. Chowdhary et al. — *"SDN-based Moving Target Defense using ML"*, IEEE CNS 2020.

Used to validate the Path Entropy `H_P` formula implemented in
`mtd_env._compute_path_entropy()`.

---

## Removed Papers (Non-ML / Networking Only)

The following papers were previously cited but have been removed because they
describe **pure networking engineering solutions** with no AI/ML/RL component.
Their techniques have been stripped from the codebase.

| Paper | Was Used For | Reason Removed |
|---|---|---|
| Jafarian et al. (2012) HotSDN — *OpenFlow Random Host Mutation* | OpenFlow Group Tables, session continuity flows | Fixed-timer randomisation only, no learning |
| Kampanakis et al. (2014) IEEE ISCC — *SDN-based Solutions for MTD* | Graceful handoff procedure | Deterministic scheduling, no AI |
