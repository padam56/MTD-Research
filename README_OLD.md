# MTD-Brain: Intelligent Decision Engine for SDN-MTD

## 1. Project Overview
This repository contains the **AI/ML/RL Decision Layer** for the MTD-Playground. As part of a comparative evaluation, these scripts provide the intelligence required to trigger SDN-based IP and Flow randomization dynamically.

> **Note on Code Origin:** All scripts in this repository were written **from scratch** for this project, based on the architecture defined in `IMPLEMENTATION_GUIDE..md`. No code was copied or adapted from any external codebase.

---

## 2. Repository Structure

```
MTD/
├── main.py                     # Entry point — orchestrates RL agent + ML detector
├── onos_config.json            # Controller IP/auth config (edit before running)
├── requirements.txt            # Python dependencies
├── .gitignore
│
├── src/                        # Core AI/ML source package
│   ├── __init__.py             # Package exports (ONOSClient, SDN_MTD_Env, ThreatDetector)
│   ├── onos_client.py          # ONOSClient — Northbound REST API + Session Continuity
│   ├── mtd_env.py              # SDN_MTD_Env — Markov Game Gymnasium environment
│   └── threat_detector.py      # ThreatDetector — Ensemble Switching (HybridMTD)
│
├── docs/                       # Research reference documents
│   ├── IMPLEMENTATION_GUIDE..md
│   ├── MTD_CCS_Threat-Model.pdf
│   ├── MTD Evaluation.docx
│   ├── MTD_Analysis_AI_ML-Padam.xlsx
│   └── MTD_Research_Analysis_SDN-AI-ML-Game-Approach_Template_Final.xlsx
│
├── models/                     # Saved ML + RL model files (auto-generated, gitignored)
│   └── .gitkeep
└── results/                    # CSV output for paper metrics (auto-generated, gitignored)
    └── .gitkeep
```

## 3. Integration Architecture
The "Brain" operates as a Northbound Application. It communicates with the **ONOS Controller** via REST API to observe network states and execute MTD mutations.

- **Infrastructure:** Mininet + Open vSwitch (Lead Author)
- **Controller:** ONOS (Lead Author)
- **Decision Engine:** Python + Reinforcement Learning (Second Author — this repo)

---

## 4. Phase 1: Reinforcement Learning (RL) — The Adaptive Scheduler

**File:** `src/mtd_env.py` + `main.py`

The RL component is modelled as a **two-player zero-sum Markov Game**. Three AI/ML papers drive this layer:

### 4a. Markov Game Environment (Eghtesad 2020)
* **Ref:** Eghtesad et al. (2020) — *"Adversarial Deep RL based Adaptive MTD"*, GameSec 2020.
* **Action Space:** `Discrete(3)` → `0 = No Move`, `1 = Moderate`, `2 = Aggressive`
* **State Space (OBS_SIZE = 24):** Includes network flow features **plus Attacker Knowledge metrics**:
  * `obs[10]` — HybridMTD ensemble threat score (injected from `threat_detector.py`)
  * `obs[11]` — **Attacker Knowledge Entropy H_A**: Shannon entropy of egress port distribution (high = MTD working)
  * `obs[12]` — **Path Entropy H_P**: diversity of active forwarding path IDs
  * `obs[13]` — **Attacker Prior Belief**: fraction of repeated IP pairs (how much attacker has mapped)
* **Reward — Zero-Sum Game Payoff (Eghtesad Eq. 6):**

$$R_d = W_{rp} \cdot (1 - belief) - W_{dc} \cdot a_d - W_{rs} \cdot belief - W_{oh} \cdot latency$$

  Where $R_d = -R_a$ (strictly competitive). Terms: *Recon Prevented*, *Cost of Deception*, *Recon Success Penalty*, *Overhead*.

### 4b. Double DQN (van Hasselt et al. 2016)
* **Ref:** van Hasselt, Guez, Silver — *"Deep RL with Double Q-learning"*, AAAI 2016.
* Reduces overestimation bias by decoupling **action selection** (online network) from **action evaluation** (target network). Essential in a noisy, adversarial state space.
* Configured via: `target_update_interval=500`, `tau=0.005` (Polyak soft update).

### 4c. Dueling Network Architecture (Wang et al. 2016)
* **Ref:** Wang et al. — *"Dueling Network Architectures for Deep RL"*, ICML 2016.
* Separates **state-value V(s)** from **advantage A(s,a)**. Critical for MTD: when the network is quiet, all actions have similar value — Dueling learns this efficiently without conflating state quality with action choice.
* Configured via: `policy_kwargs={"net_arch": [512, 256]}` (wide state encoder, narrow advantage estimator).

## 5. Phase 2: Machine Learning (ML) — The Anomaly Trigger

**File:** `src/threat_detector.py`

### 5a. Ensemble Switching — HybridMTD (Li et al. 2021)
* **Ref:** Li et al. (2021) — *"HybridMTD: Ensemble-based MTD with Adaptive Classifier Selection"*, IEEE TDSC.
* **Dataset:** CIC-IDS2017 + InSDN (SDN-specific attack flows). Ref: Elsayed & Sahoo (2020), IEEE Access.
* Three classifiers trained; active model selected at inference by live traffic volume:

| Regime | Condition | Classifier | Rationale |
|---|---|---|---|
| A | ≤ 500 flows | Random Forest | High recall, interpretable |
| B | 500–2000 flows | XGBoost | Speed + accuracy under load |
| C | > 2000 or 3× spike | Isolation Forest | Unsupervised, catches zero-days |

* **Alert Threshold:** ≥ 90% confidence (A/B) or anomaly score < −0.05 (C) → immediate aggressive MTD.

### 5b. SHAP Explainability (Lundberg & Lee 2017)
* **Ref:** Lundberg & Lee — *"A Unified Approach to Interpreting Model Predictions"*, NeurIPS 2017.
* **Method:** `ThreatDetector.explain_prediction()` uses **TreeSHAP** to compute per-feature Shapley values, showing exactly which flow feature (e.g., `SYN Flag Count`, `Flow Bytes/s`) most drove each threat classification.
* **Paper value:** Provides the interpretability results section — proves the detector is not a black box. Top SHAP feature is logged per step in `results/evaluation_results.csv`.

## 6. The Bridge — `src/onos_client.py`

`ONOSClient` is the pure API wrapper between the Python AI layer and the ONOS controller.
It has **no decision logic** — it only observes and acts on behalf of the AI models above.

| Method | Description | Used by |
|---|---|---|
| `get_network_state()` | Flow rules + port stats for RL state vector | `SDN_MTD_Env._get_observation()` |
| `get_controller_latency()` | Round-trip ms — overhead cost in Eghtesad Eq. (6) | `SDN_MTD_Env.step()` |
| `execute_mutation(level)` | POST Intent to ONOS for IP/path shuffle | `SDN_MTD_Env.step()` |
| `trigger_high_alert_mutation()` | Force level-2 mutation on ML high-alert | `ThreatDetector.run_detection_loop()` |

---

## 7. Deliverables & Evaluation Metrics (Task A4)

`main.py` automatically writes `results/evaluation_results.csv` with per-step columns:

| CSV Column | Metric | Source |
|---|---|---|
| `reward` | Zero-Sum game payoff $R_d$ | Eghtesad (2020) Eq. 6 |
| `attacker_entropy` | $H_A$ — attacker knowledge entropy | Eghtesad (2020) Sec 3.2 |
| `path_entropy` | $H_P$ — routing unpredictability | Chowdhary (2020) |
| `attacker_belief` | Estimated recon success rate | Eghtesad (2020) |
| `recon_prevented` | Knowledge disruption reward term | Eghtesad (2020) |
| `deception_cost` | Cost-of-moving penalty term | Eghtesad (2020) |
| `threat_score` | Ensemble classifier confidence | Li / HybridMTD (2021) |
| `shap_top_feature` | Highest-SHAP flow feature | Lundberg & Lee (2017) |
| `latency_ms` | Controller round-trip overhead | Eghtesad (2020) cost term |

---

## 8. How to Run

### Option A — Offline (no ONOS controller needed)

Runs a full training + evaluation cycle using `MockONOSClient` and synthetic data. Produces real results and graphs.

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full offline demo (fast sanity check)
python scripts/run_offline.py --fast

# 3. Run with more training for better results (paper-quality)
python scripts/run_offline.py --timesteps 50000 --episodes 10 --samples 20000

# 4. Generate paper figures from results
python scripts/plot_results.py
# → results/figures/*.png
```

### Option B — Live (with ONOS controller)

```bash
# 1. Install dependencies (in venv as above)
pip install -r requirements.txt

# 2. Set your ONOS controller IP
nano onos_config.json

# 3a. Train the ML detector on CIC-IDS2017, then start the full loop
python main.py --train --dataset /path/to/CIC-IDS2017.csv

# 3b. OR skip training (use saved model) and go straight to evaluation
python main.py --eval-only
```

---

## 9. GitHub Copilot Master Prompts (Dev Reference)

#### Prompt 1: The API Bridge (`onos_client.py`)
> "Create a Python class `ONOSClient` as a pure REST API wrapper for ONOS. Use `requests` with Basic Auth. Implement `get_network_state()` to fetch flows + port stats (feeds the RL state vector from Eghtesad 2020), `get_controller_latency()` for the cost-of-moving term, and `execute_mutation(level)` to POST Intents for IP/path randomisation at levels 0/1/2."

#### Prompt 2: The RL Environment (`mtd_env.py`)
> "Using `ONOSClient`, create a Gymnasium environment `SDN_MTD_Env` as a two-player Markov Game (Eghtesad 2020). State vector size 24: flow stats + latency + ML threat score (obs[10]) + Attacker Knowledge Entropy H_A (obs[11]) + Path Entropy H_P (obs[12]) + Attacker Prior Belief (obs[13]). Reward is the Zero-Sum payoff from Eq. (6) of Eghtesad: R_d = W_rp*(1-belief) - W_dc*action - W_rs*belief - W_oh*(latency/1000)."

#### Prompt 3: The ML Detector (`threat_detector.py`)
> "Implement `EnsembleSwitchingDetector` per HybridMTD (Li et al. 2021). Three classifiers: RandomForest (Regime A: ≤500 flows), XGBoost (Regime B: 500–2000), IsolationForest (Regime C: >2000 or 3× spike). Train all on CIC-IDS2017 + InSDN. At inference select regime by live flow count. Add `explain_prediction()` using TreeSHAP (Lundberg 2017) to identify top contributing flow features."

#### Prompt 4: Double DQN + Dueling (`main.py`)
> "Configure SB3's DQN as a Double DQN (van Hasselt 2016) with `target_update_interval=500` and `tau=0.005`. Use `policy_kwargs={'net_arch': [512, 256]}` to approximate the Dueling architecture (Wang 2016) where the wider first layer encodes state value and the narrower second layer estimates action advantage. Log per-step SHAP top features to the evaluation CSV."

---

## 10. Research References

Full citations with code mappings: see [`docs/papers.md`](docs/papers.md)

| # | Paper | Technique | File |
|---|---|---|---|
| 1 | Eghtesad et al. (2020) GameSec | Markov Game, Attacker Entropy, Zero-Sum payoff | `src/mtd_env.py` |
| 2 | Li et al. (2021) IEEE TDSC | HybridMTD Ensemble Switching (RF/XGB/IsoForest) | `src/threat_detector.py` |
| 3 | van Hasselt et al. (2016) AAAI | Double DQN — decoupled Q-value estimation | `main.py` |
| 4 | Wang et al. (2016) ICML | Dueling Network V(s) + A(s,a) separation | `main.py` |
| 5 | Lundberg & Lee (2017) NeurIPS | SHAP explainability — TreeSHAP for RF/XGB | `src/threat_detector.py` |
| 6 | Sharafaldin et al. (2018) ICISSP | CIC-IDS2017 training dataset | `src/threat_detector.py` |
| 7 | Elsayed & Sahoo (2020) IEEE Access | InSDN — SDN-specific attack dataset | `src/threat_detector.py` |
| 8 | Chowdhary et al. (2020) IEEE CNS | Path Entropy as MTD metric formula | `src/mtd_env.py` |
