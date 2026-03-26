# MTD-Playground: AI/ML Decision Engine for SDN Moving Target Defense

Adaptive Moving Target Defense using Deep Reinforcement Learning on Software-Defined Networks.

## Paper
**MTD-Playground: An Attacker-Aware Evaluation Framework for Network Moving Target Defense** (2026 CCS)

## Method
Based on **Eghtesad et al. (2020)** — "Adversarial Deep RL Based Adaptive Moving Target Defense" (GameSec 2020).
- Markov Game: Defender vs Attacker with DQN policy learning
- 14-dim observation (flow count, path entropy, attacker knowledge, latency, ...)
- 3 actions: Hold / Moderate Shuffle / Aggressive Randomize
- Zero-sum reward: R_defender = -R_attacker

## Results (30-episode evaluation)
| Strategy | ASR (lower=better) | Latency | Availability | Reward |
|----------|-------------------|---------|-------------|--------|
| No MTD | 100% | 11ms | 100% | 8.0 |
| Periodic | 97% | 50ms | 97.7% | 50.2 |
| Random | 10% | 221ms | 82.5% | 144.0 |
| **DQN Adaptive** | **47%** | **74ms** | **89%** | **155.3** |

**ASR** = Attack Success Rate (% of episodes where attacker exfiltrates data). Lower = better defense.

## Project Structure
```
├── src/                    # Core simulation code
│   ├── mtd_env.py            Gymnasium environment (5 hosts, 3 switches)
│   ├── train.py              Train DQN agent (200k steps)
│   ├── evaluate.py           Compare 4 strategies → CSV + plots
│   ├── simulate.py           Terminal live animation
│   ├── generate_report.py    Interactive HTML report
│   └── generate_flowchart.py Project flowchart
│
├── mininet/                # Real SDN integration (ONOS + Mininet)
│   ├── topology.py           Mininet topology (paper Figure 1)
│   ├── onos_client.py        ONOS REST API wrapper
│   ├── mtd_env_live.py       Live Gymnasium env (reads from ONOS)
│   ├── run_live.py           Run DQN agent on real network
│   ├── run_demo.py           Full demo: attacker + defender
│   └── attack_scripts/       recon.sh, ddos.sh, lateral.sh
│
├── models/                 # Trained DQN models
├── results/                # Outputs (HTML report, CSVs, flowchart)
├── docs/                   # Paper PDF, literature survey, roadmap
├── setup.sh                # One-command ONOS + Mininet installer
└── requirements.txt
```

## Quick Start

### Offline simulation (no ONOS needed)
```bash
pip install -r requirements.txt
cd src
python3 train.py --timesteps 200000          # Train DQN
python3 evaluate.py --episodes 50            # Compare strategies
python3 generate_report.py                   # HTML report
```

### Real SDN demo (ONOS + Mininet)
```bash
sudo bash setup.sh                           # Install everything
sudo python3 mininet/topology.py             # Start topology (Terminal 1)
python3 mininet/run_live.py --strategy dqn   # DQN agent (Terminal 2)
# Open http://localhost:8181/onos/ui (onos/rocks) to watch
```

## References
See [docs/LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md) for the full verified literature survey (14 papers).
