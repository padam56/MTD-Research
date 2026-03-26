# MTD-Playground: Full Roadmap & Code Analysis

## 1. WHAT WE HAVE DONE SO FAR (Completed)

### Phase 1: Literature Survey ✅
- Surveyed 15+ papers on MTD + SDN + AI/ML
- Verified every paper is real and published (found 2 bogus ones from old work)
- Selected **Eghtesad et al. (2020) GameSec** as the primary implementation paper
- Output: `docs/LITERATURE_SURVEY.md`

### Phase 2: Offline Simulation (reference-only/) ✅
Built a complete offline simulation of the MTD-Playground framework:
- Gymnasium environment simulating the paper's topology
- Multi-stage attacker with kill chain progression
- DQN agent trained to defend adaptively
- Evaluation pipeline comparing 4 strategies
- HTML interactive report with visualizations
- 6 publication-ready matplotlib plots

---

## 2. CODE ARCHITECTURE — WHAT EACH FILE DOES

```
MTD - NEW/
├── reference-only/          ← OFFLINE SIMULATION (no ONOS needed)
│   ├── mtd_env.py           ← The core: Gymnasium environment
│   ├── train.py             ← Trains the DQN agent
│   ├── evaluate.py          ← Runs all strategies, outputs CSV + plots
│   ├── simulate.py          ← Terminal-based live simulation
│   └── generate_report.py   ← Generates interactive HTML report
│
├── mininet-ready/           ← TODO: Real ONOS/Mininet integration
│
├── models/
│   ├── best_model.zip       ← Best DQN model (selected by eval callback)
│   └── dqn_mtd_final.zip   ← Final DQN model after full training
│
├── results/
│   ├── simulation_report.html  ← Interactive HTML report (open in browser)
│   ├── summary_comparison.csv  ← Strategy comparison table
│   ├── timeseries_*.csv        ← Per-step data for each strategy
│   └── fig*.png                ← Publication plots
│
├── docs/
│   ├── LITERATURE_SURVEY.md    ← Verified references
│   └── ROADMAP.md              ← This file
│
├── requirements.txt
└── README.md
```

---

## 3. DETAILED CODE BREAKDOWN

### 3a. mtd_env.py — The Gymnasium Environment (THE CORE)

**What it is:**
A custom OpenAI Gymnasium environment that simulates the MTD-Playground
enterprise network. This is where everything happens.

**Input (what the agent sees) — 14-dimensional observation vector:**
```
Index  Feature                    Range    Source
─────  ─────────────────────────  ───────  ──────────────────────────
[0]    Active flow count          [0, 1]   NetworkSim.flow_count / 500
[1]    Avg packets per switch     [0, 1]   flow_count / (switches * 200)
[2]    Avg bytes per switch       [0, 1]   Proxy from [1]
[3]    Controller latency         [0, 1]   latency_ms / 200
[4]    Threat level               [0, 1]   attacker.stage / 4 (proxy)
[5]    Attacker stage             [0, 1]   stage / 4 (0=Recon, 4=Exfil)
[6]    Attacker knowledge         [0, 1]   Fraction of topology mapped
[7]    Path entropy (H_P)         [0, 1]   Shannon entropy of path distribution
[8]    Attacker entropy (H_A)     [0, 1]   1.0 - attacker_knowledge
[9]    Time since last mutation   [0, 1]   steps_since_mutation / 20
[10]   Total mutation count       [0, 1]   mutation_count / 50
[11]   Recon accuracy             [0, 1]   How well attacker identifies hosts
[12]   Network load               [0, 1]   flow_count * latency / 50000
[13]   Service availability       [0, 1]   Fraction of successful requests
```

**Output (what the agent decides) — Discrete(3):**
```
Action  Name                   Effect
──────  ─────────────────────  ─────────────────────────────────────
0       No Mutation (HOLD)     Do nothing. Cheap but attacker keeps learning.
1       Moderate Shuffle       Partially re-randomize 1 of 3 paths.
                               Costs ~10ms latency, ~2% availability.
                               Reduces attacker knowledge by 10-25%.
2       Aggressive Randomize   Fully re-randomize all 3 paths.
                               Costs ~25ms latency, ~5% availability.
                               Reduces attacker knowledge by 25-50%.
                               35% chance of pushing attacker back a stage.
```

**Reward function (Eghtesad et al. 2020, Eq. 6 — zero-sum payoff):**
```
reward = security_gain          (+)  keeps attacker knowledge low
       + stage_penalty          (-)  penalty proportional to attacker stage
       + stage_bonus            (-)  big penalty when attacker advances
       + exfil_penalty          (-)  -10 if attacker reaches exfiltration
       + entropy_bonus          (+)  reward for high path entropy
       + availability_bonus     (+)  reward for keeping services up
       - mutation_cost          (-)  cost proportional to action level
       - overhead               (-)  cost from increased latency
```

**Key classes inside mtd_env.py:**
```
AttackerSim       Simulates multi-stage attacker (5-stage kill chain)
                  - Gains knowledge each step (faster when path entropy is low)
                  - MTD mutations reduce knowledge and can push stage back
                  - Advances stage probabilistically based on knowledge level

NetworkSim        Simulates SDN network state (3 switches, 3 paths)
                  - Tracks path probability distribution (used for entropy)
                  - Applies mutations (partial or full path re-randomization)
                  - Simulates latency overhead and availability degradation

MTDPlaygroundEnv  The Gymnasium env combining both
                  - reset() → start fresh episode
                  - step(action) → apply action, advance attacker, return obs/reward
                  - 200 steps per episode (truncated) or until exfiltration (terminated)
```

### 3b. train.py — DQN Training

**What it does:**
Trains a Double DQN agent (stable-baselines3) on the MTDPlaygroundEnv.

**Model architecture:**
```
Algorithm:    Double DQN (van Hasselt et al. 2016)
Network:      MLP with layers [256, 256]
Input:        14-dim observation vector
Output:       Q-values for 3 actions
```

**Training hyperparameters:**
```
learning_rate:         5e-4
buffer_size:           50,000 transitions
batch_size:            128
gamma (discount):      0.99
target_update:         every 250 steps (soft update, tau=0.01)
exploration:           epsilon 1.0 → 0.1 over 40% of training
total_timesteps:       200,000
```

**What it's trained ON:**
- NOT trained on any dataset (no CIC-IDS2017, no InSDN)
- Trained via **reinforcement learning** — the agent interacts with the
  simulated environment (mtd_env.py) and learns from rewards
- Each training step: agent sees network state → picks action → gets reward
  → updates Q-network to maximize future rewards
- The attacker is part of the environment (simulated opponent)

**What it produces:**
- `models/best_model.zip` — checkpoint with highest eval reward
- `models/dqn_mtd_final.zip` — final model after all training steps

### 3c. evaluate.py — Strategy Comparison

**What it does:**
Runs 4 strategies for N episodes each, computes paper metrics, generates plots.

**Strategies compared:**
```
1. No MTD          — always action=0 (baseline: proves the problem exists)
2. Periodic MTD    — action=2 every 10 steps (naive timer-based)
3. Random MTD      — random action each step (brute-force defense)
4. DQN Adaptive    — trained model picks actions (our contribution)
```

**Metrics computed (from paper Section 7.2):**
```
ASR       Attack Success Rate (% episodes where attacker exfiltrates)
ACT       Attack Completion Time (steps until exfiltration)
H_P       Mean path entropy (network unpredictability)
H_A       Mean attacker entropy (attacker confusion)
Latency   Mean end-to-end latency in ms
Avail     Mean service availability
Mutations Total path changes per episode
Reward    Cumulative RL reward (overall strategy quality)
```

**Outputs:**
- `results/summary_comparison.csv` — one row per strategy
- `results/timeseries_*.csv` — per-step data for each strategy
- `results/fig1-6.png` — publication-ready plots

### 3d. simulate.py — Terminal Live Visualization

**What it does:**
Real-time terminal animation using Rich library. Shows topology, kill chain,
metrics updating live. Good for demos in a terminal.

### 3e. generate_report.py — Interactive HTML Report

**What it does:**
Runs all strategies, captures every step, generates a self-contained HTML file
with interactive replay (play/pause/step), topology SVG, charts (Chart.js),
and the full Q&A/literature survey context.

---

## 4. MODEL INPUT/OUTPUT SUMMARY

```
┌─────────────────────────────────────────────────────────┐
│                    THE FULL PIPELINE                     │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Network     │───>│  14-dim      │───>│  DQN      │ │
│  │  State       │    │  Observation │    │  Agent    │ │
│  │  (flows,     │    │  Vector      │    │           │ │
│  │   latency,   │    │  [0.0-1.0]   │    │  Input:   │ │
│  │   paths)     │    │              │    │  14 floats│ │
│  └──────────────┘    └──────────────┘    │           │ │
│                                          │  Output:  │ │
│  ┌──────────────┐    ┌──────────────┐    │  Action   │ │
│  │  Attacker    │───>│  Knowledge,  │───>│  {0,1,2}  │ │
│  │  State       │    │  Stage,      │    └─────┬─────┘ │
│  │  (knowledge, │    │  Recon acc   │          │       │
│  │   stage)     │    │              │          ▼       │
│  └──────────────┘    └──────────────┘    ┌───────────┐ │
│                                          │  Execute  │ │
│  ┌──────────────┐                        │  Mutation │ │
│  │  Reward      │<───────────────────────│  on SDN   │ │
│  │  R_d = -R_a  │                        │  Network  │ │
│  │  (zero-sum)  │                        └───────────┘ │
│  └──────────────┘                                      │
└─────────────────────────────────────────────────────────┘

OFFLINE (reference-only/):
  Network State = simulated by NetworkSim + AttackerSim
  Execute Mutation = updates simulated path probabilities

ONLINE (mininet-ready/ — TODO):
  Network State = fetched from ONOS REST API (flows, port stats)
  Execute Mutation = POST flow rules to ONOS (real path change)
```

---

## 5. WHAT STILL NEEDS TO BE DONE

### Phase 3: ONOS/Mininet Integration (mininet-ready/) — NOT DONE YET

This is what you'll do on your friend's PC where ONOS + Mininet is set up.

**Files to create:**
```
mininet-ready/
├── onos_client.py        ← REST API wrapper for ONOS controller
├── mtd_env_live.py       ← Same env but reads from ONOS instead of simulation
├── topology.py           ← Mininet/Containernet topology setup (5 hosts, 3 switches)
├── run_live.py           ← Main script: loads trained model, runs on real network
└── attack_scripts/       ← nmap, hping3, curl scripts to simulate attacker
    ├── recon.sh           ← nmap scan from attacker host
    ├── ddos.sh            ← hping3 flood from attacker
    └── lateral.sh         ← curl/wget pivoting through hosts
```

**What onos_client.py needs to do:**
```python
class ONOSClient:
    def get_network_state() -> dict:
        # GET http://<onos>:8181/onos/v1/flows → flow rules
        # GET http://<onos>:8181/onos/v1/statistics/ports → packet counts
        # Returns same format as NetworkSim but with REAL data

    def execute_mutation(level: int) -> bool:
        # level=0: do nothing
        # level=1: POST new flow rules to change 1 path
        # level=2: POST new flow rules to randomize all paths
        # Uses ONOS Intents API or direct flow rule installation

    def get_controller_latency() -> float:
        # GET http://<onos>:8181/onos/v1/cluster
        # Measures round-trip time in ms
```

**What mtd_env_live.py needs to do:**
- Same interface as mtd_env.py (Gymnasium env)
- But _get_observation() calls onos_client instead of simulation
- And step() calls onos_client.execute_mutation() instead of NetworkSim

**What topology.py needs to do:**
```python
# Create Mininet topology matching MTD-Playground paper Figure 1:
#
#   Client (h1) ─── S0 ─── DMZ Web (h3)
#                    │
#                   S1 ─── Internal App (h4)
#                    │
#   Attacker (h2) ─ S2 ─── DB Server (h5)
#
# Using Containernet for Docker-based hosts
# Using Open vSwitch for switches
# Connected to ONOS controller
```

**What run_live.py needs to do:**
```python
# 1. Load the trained DQN model (models/best_model.zip)
# 2. Create live environment (mtd_env_live.py + onos_client.py)
# 3. Loop:
#    a. Read network state from ONOS
#    b. DQN agent picks action
#    c. Execute mutation on ONOS
#    d. Log metrics to CSV
# 4. Meanwhile, run attack scripts from attacker host
# 5. Compare results with offline simulation
```

### Phase 4: Real Attack Evaluation — NOT DONE YET

Run actual attacks on the Mininet topology while DQN defends:
```
Terminal 1: python3 run_live.py                    ← DQN defending
Terminal 2: mininet> h2 nmap -sS 10.0.0.30         ← Attacker scanning
Terminal 3: mininet> h2 hping3 --flood 10.0.0.30    ← DDoS attempt
```

Compare real results with simulation results to validate the model.

### Phase 5: Paper Results — NOT DONE YET

- Run Phase 4 experiments multiple times
- Collect real ASR, ACT, latency, availability from live testbed
- Compare with offline simulation results (should be similar)
- Generate final plots for the paper

---

## 6. CURRENT RESULTS SUMMARY

From 30-episode offline evaluation:

| Strategy         | ASR    | Avg Steps | Mutations | Latency  | Avail  | Reward |
|-----------------|--------|-----------|-----------|----------|--------|--------|
| No MTD          | 100%   | 31        | 0         | 11ms     | 100%   | 8.0    |
| Periodic MTD    | 97%    | 68        | 7         | 50ms     | 97.7%  | 50.2   |
| Random MTD      | 10%    | 194       | 129       | 221ms    | 82.5%  | 144.0  |
| **DQN Adaptive**| **47%**| **141**   | **58**    | **74ms** |**89%** |**155.3**|

**What this means:**
- DQN gets the **highest reward** (best overall trade-off)
- DQN uses **half the mutations** of Random (58 vs 129) = less overhead
- DQN maintains **89% availability** vs Random's 82.5%
- DQN has **74ms latency** vs Random's 221ms
- DQN still has 47% ASR → room for improvement with more training or better reward shaping

---

## 7. HOW TO IMPROVE THE DQN (If Needed)

Things that can improve the 47% ASR:

1. **Train longer** — 200k steps may not be enough, try 500k-1M
2. **Reward shaping** — increase exfiltration penalty from -10 to -20
3. **Better state features** — add rate-of-change of attacker knowledge
4. **PPO instead of DQN** — PPO often works better for these environments
5. **Curriculum learning** — start with weak attacker, gradually increase
6. **Prioritized replay** — focus learning on episodes where attack succeeded

---

## 8. DEPENDENCIES

```
numpy>=1.24          ← Array math
gymnasium>=0.29      ← RL environment framework
stable-baselines3    ← DQN implementation
matplotlib>=3.7      ← Publication plots
rich                 ← Terminal visualization
```

For ONOS integration (Phase 3), also need:
```
requests             ← ONOS REST API calls
mininet              ← Network emulation (on friend's PC)
containernet         ← Docker-in-Mininet (on friend's PC)
```
