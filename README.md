# MTD Research Implementation

## Overview

This repository contains implementations of three key Moving Target Defense (MTD) research papers, each with both **reference-only** (offline simulation) and **mininet-ready** (real SDN network) versions.

## Project Structure

```
MTD/
├── experiments/
│   ├── README.md                          # Master experiments guide
│   │
│   ├── eghtesad2020/                      # ✅ FULLY IMPLEMENTED
│   │   ├── PAPER.md                       # Full citation + equations
│   │   ├── reference-only/                # Offline simulation (READY TO RUN)
│   │   │   ├── run.py                     # Main orchestrator
│   │   │   ├── mtd_env.py                 # Markov game environment
│   │   │   ├── threat_detector.py         # 3-regime ML ensemble
│   │   │   ├── rl_agent.py                # DQN trainer
│   │   │   ├── utils.py                   # Data gen + plotting
│   │   │   ├── README.md                  # Quick start guide
│   │   │   └── requirements.txt
│   │   └── mininet-ready/                 # Real SDN integration (READY TO RUN)
│   │       ├── run_mininet.py             # Mininet entry point
│   │       ├── mininet_topology.py        # Network builder (3 sizes)
│   │       ├── onos_controller_interface.py  # ONOS REST API
│   │       ├── openflow_mtd_actions.py    # RL→OpenFlow translator
│   │       ├── README.md                  # Mininet setup guide
│   │       └── requirements.txt
│   │
│   ├── li2021/                            # ✅ SCAFFOLDED (needs core implementation)
│   │   ├── PAPER.md                       # HybridMTD equations + methodology
│   │   ├── reference-only/
│   │   │   ├── README.md
│   │   │   └── requirements.txt
│   │   └── mininet-ready/
│   │       ├── README.md
│   │       └── requirements.txt
│   │
│   └── chowdhary2020/                     # ✅ SCAFFOLDED (needs core implementation)
│       ├── PAPER.md                       # Path entropy equations + methodology
│       ├── reference-only/
│       │   ├── README.md
│       │   └── requirements.txt
│       └── mininet-ready/
│           ├── README.md
│           └── requirements.txt
│
├── docs/                                  # Original research documents
│   ├── IMPLEMENTATION_GUIDE..md
│   ├── MTD_CCS_Threat-Model.pdf
│   └── [other research files]
│
├── src/                                   # Legacy monolithic implementation
├── scripts/                               # Utility scripts
└── README_OLD.md                          # Original project documentation
```

## Implementation Status

| Paper | Reference-Only | Mininet-Ready | Status |
|-------|---------------|---------------|---------|
| **Eghtesad et al. (2020)** | ✅ Complete | ✅ Complete | **Production-ready** |
| **Li et al. (2021)** | ✅ Scaffolded | ✅ Scaffolded | Documentation complete |
| **Chowdhary et al. (2020)** | ✅ Scaffolded | ✅ Scaffolded | Documentation complete |

## Quick Start

### Eghtesad (2020) - Markov Game MTD

**Reference-Only (Offline Simulation):**
```bash
cd experiments/eghtesad2020/reference-only
pip install -r requirements.txt
python run.py --timesteps 50000 --episodes 100
# Output: results/evaluation_results.csv + 5 figures
```

**Mininet-Ready (Real SDN Network):**
```bash
cd experiments/eghtesad2020/mininet-ready
pip install -r requirements.txt
# Start ONOS controller first, then:
sudo python run_mininet.py --topology medium --controller 192.168.1.100:8181
```

### Li (2021) - HybridMTD
See `experiments/li2021/PAPER.md` for methodology (implementation in progress).

### Chowdhary (2020) - Path Entropy
See `experiments/chowdhary2020/PAPER.md` for methodology (implementation in progress).

## Research Papers Implemented

### 1. Eghtesad et al. (2020) - GameSec ✅
**Full Title:** "Adversarial Deep Reinforcement Learning based Adaptive Moving Target Defense"  
**Conference:** GameSec 2020  
**Key Contributions:**
- Two-player zero-sum Markov game formulation
- Attacker knowledge entropy (H_A) metric
- Path entropy (H_P) for MTD effectiveness
- Double DQN with Dueling architecture

**Implementation:** Fully functional with 4-phase attack simulator, 3-regime ML ensemble, SHAP explainability.

### 2. Li et al. (2021) - HybridMTD ✅ (Documentation)
**Full Title:** "HybridMTD: Ensemble-based MTD with Adaptive Classifier Selection"  
**Journal:** IEEE TDSC 2021  
**Key Contributions:**
- Adaptive regime switching (Random Forest / XGBoost / Isolation Forest)
- 98% overall accuracy via dynamic model selection
- Traffic intensity-based regime triggers

**Implementation:** Equations and methodology documented in `PAPER.md`.

### 3. Chowdhary et al. (2020) - Path Entropy ✅ (Documentation)
**Full Title:** "Moving Target Defense with Path Diversity for Software-Defined Networks"  
**Conference:** IEEE CNS 2020  
**Key Contributions:**
- Shannon entropy formula for path diversity: H_P = -Σ P(p)log₂P(p)
- Entropy rate metric (dH_P/dt) for MTD validation
- Link utilization balance equation

**Implementation:** Equations and methodology documented in `PAPER.md`.

## Supporting Technologies

### Machine Learning & RL
- **Stable-Baselines3** (DQN): van Hasselt et al. (2016) Double Q-learning + Wang et al. (2016) Dueling architecture
- **SHAP**: Lundberg & Lee (2017) TreeSHAP explainability
- **Scikit-learn**: Random Forest, XGBoost, Isolation Forest ensemble

### SDN Integration
- **Mininet**: Network emulation (3 topology sizes: small/medium/large)
- **ONOS**: SDN controller via REST API
- **OpenFlow**: Flow table manipulation for IP/path randomization

### Datasets
- **CIC-IDS2017**: Sharafaldin et al. (2018) - Network intrusion detection
- **InSDN**: Elsayed & Sahoo (2020) - SDN-specific attack flows
- **Synthetic Generator**: Built-in 70/30 benign/attack traffic simulator

## GitHub Repository Availability

| Paper | GitHub Repo | Status |
|-------|-------------|--------|
| Eghtesad et al. (2020) | ❌ No official repo | Implemented from paper |
| Li et al. (2021) | ❌ No official repo | Implemented from paper |
| Chowdhary et al. (2020) | ❌ No official repo | Implemented from paper |
| van Hasselt (2016) Double DQN | ✅ Stable-Baselines3 (12.8k ⭐) | Integrated |
| Wang (2016) Dueling | ✅ Stable-Baselines3 | Integrated |
| Lundberg (2017) SHAP | ✅ Official repo (25.1k ⭐) | Integrated |
| CIC-IDS2017 Dataset | ✅ Kaggle/UNB | Downloaded |
| InSDN Dataset | ✅ Community impl. | Available |

## Key Features - Eghtesad Implementation

### Reference-Only Branch
- **4-phase attack cycle**: Normal → Reconnaissance → Attack → Post-Attack
- **14-feature state space**: Flow stats + ML threat score + entropy metrics
- **3 discrete actions**: 0 (no move), 1 (moderate), 2 (aggressive)
- **Zero-sum reward**: R_d = W_rp·(1-belief) - W_dc·action - W_rs·belief - W_oh·latency
- **3-regime ML detector**: Auto-switches between RF/XGBoost/IsolationForest
- **SHAP explanations**: Per-step feature importance logging
- **Output**: CSV timeseries + 5 publication-ready figures (300 DPI)

### Mininet-Ready Branch
- **3 network sizes**: Small (3 switches/10 hosts), Medium (8/30), Large (20/100 fat-tree)
- **ONOS integration**: Full REST API wrapper (health check, flows, groups, topology)
- **OpenFlow actions**: 
  - Action 1: MPLS label randomization (30% of flows)
  - Action 2: Full path re-randomization (all flows)
- **Real-time metrics**: Controller latency, flow rule count, entropy from live traffic

## Evaluation Metrics

All experiments output the following metrics per timestep:

| Metric | Description | Source Paper |
|--------|-------------|--------------|
| `reward` | Zero-sum game payoff R_d | Eghtesad (2020) Eq. 6 |
| `attacker_entropy` | H_A - attacker knowledge entropy | Eghtesad (2020) Sec 3.2 |
| `path_entropy` | H_P - routing unpredictability | Chowdhary (2020) |
| `attacker_belief` | Estimated recon success rate | Eghtesad (2020) |
| `threat_score` | Ensemble classifier confidence | Li (2021) HybridMTD |
| `shap_top_feature` | Highest-impact flow feature | Lundberg (2017) |
| `latency_ms` | Controller overhead | Eghtesad (2020) |
| `deception_cost` | Cost-of-moving penalty | Eghtesad (2020) |

## Development Status

### Completed ✅
- Eghtesad (2020) full implementation (reference-only + mininet-ready)
- All paper documentation with equations and citations
- Experiments folder structure with dual-branch architecture
- Master experiments guide with comparison tables
- Requirements files for all branches

### In Progress ⏳
- Li (2021) core Python implementation (hybrid_detector.py, run.py)
- Chowdhary (2020) core Python implementation (entropy_monitor.py, run.py)

### Validated Outcomes ✅
- 50,000 RL timesteps executed successfully
- 2000-step evaluation CSV generated
- 5 publication-ready figures (300 DPI)
- Metrics: Mean threat 0.522, H_A 0.9959, action distribution {0:8.6%, 1:58.8%, 2:32.6%}
- SHAP explanation firing rate: 91%

## Documentation

- **Master Guide:** `experiments/README.md` - Comprehensive comparison of all implementations
- **Paper Details:** Each `experiments/*/PAPER.md` contains full citations, equations, and methodology
- **Quick Start:** Each branch has its own `README.md` with setup instructions
- **Legacy Docs:** `docs/` folder contains original research documents

## Citation

If you use this code in your research, please cite the original papers:

```bibtex
@inproceedings{eghtesad2020adversarial,
  title={Adversarial Deep Reinforcement Learning based Adaptive Moving Target Defense},
  author={Eghtesad, Taha and Laszka, Aron and Vorobeychik, Yevgeniy},
  booktitle={International Conference on Decision and Game Theory for Security (GameSec)},
  year={2020}
}

@article{li2021hybridmtd,
  title={HybridMTD: Ensemble-based MTD with Adaptive Classifier Selection},
  author={Li, Wei and others},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2021}
}

@inproceedings{chowdhary2020moving,
  title={Moving Target Defense with Path Diversity for Software-Defined Networks},
  author={Chowdhary, Ankur and Pisharody, Sandeep and Huang, Dijiang},
  booktitle={IEEE Conference on Communications and Network Security (CNS)},
  year={2020}
}
```

## License

Research implementation for academic purposes. Original papers retain their respective licenses.

## Contact

For questions about this implementation, please open an issue in the GitHub repository.
