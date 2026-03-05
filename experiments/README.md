# MTD Research Experiments — Single-Paper Focused Implementation

This folder contains **focused implementations of MTD papers**, each with two separate branches:

1. **Reference-Only** — Offline simulators (no real network needed)
2. **Mininet-Ready** — Real network integration (Mininet + ONOS)

---

## Folder Structure

```
experiments/
├── eghtesad2020/                    # Eghtesad et al. (2020) GameSec
│   ├── PAPER.md                     # Citation, equations, implementation guide
│   ├── reference-only/              # ↓ Offline simulator
│   │   ├── README.md                # Guidelines & quick start
│   │   ├── run.py                   # Main entry point
│   │   ├── mtd_env.py               # Markov game environment
│   │   ├── threat_detector.py       # ML threat detector
│   │   ├── rl_agent.py              # RL trainer (DQN)
│   │   ├── utils.py                 # Data generation & plotting
│   │   └── requirements.txt
│   └── mininet-ready/               # ↓ Real network integration
│       ├── README.md                # Guidelines for Mininet setup
│       ├── run_mininet.py           # Main entry point
│       ├── mininet_topology.py      # Network topology builder
│       ├── onos_controller_interface.py      # ONOS REST API wrapper
│       ├── openflow_mtd_actions.py  # RL actions → OpenFlow mutations
│       ├── threat_detector_live.py  # Real-time threat detection
│       └── requirements.txt
│
├── li2021/                          # Li et al. (2021) HybridMTD
│   ├── PAPER.md                     # Citation, equations, regime switching
│   ├── reference-only/              # ↓ Offline 3-regime ensemble
│   │   ├── README.md
│   │   ├── run.py
│   │   ├── hybrid_detector.py       # Adaptive regime switching
│   │   ├── utils.py
│   │   └── requirements.txt
│   └── mininet-ready/               # ↓ Live regime-adaptive detector
│       ├── README.md
│       ├── run_mininet.py
│       ├── hybrid_detector_live.py
│       └── requirements.txt
│
└── chowdhary2020/                   # Chowdhary et al. (2020) Path Entropy
    ├── PAPER.md                     # Citation, equations, entropy metric
    ├── reference-only/              # ↓ Path entropy validation
    │   ├── README.md
    │   ├── run.py
    │   ├── entropy_monitor.py       # Shannon entropy calculator
    │   ├── utils.py
    │   └── requirements.txt
    └── mininet-ready/               # ↓ Live entropy measurement
        ├── README.md
        ├── run_mininet.py
        ├── entropy_monitor_live.py
        └── requirements.txt
```

---

## Comparison: Reference-Only vs Mininet-Ready

| Aspect | Reference-Only | Mininet-Ready |
|--------|---|---|
| **Network** | MockONOSClient (simulator) | Real Mininet + ONOS |
| **Setup** | Python only | Requires Mininet, ONOS |
| **Speed** | Very fast (50k steps in seconds) | Slower (real packet processing) |
| **Use Case** | Research, hyperparameter tuning | Testbed validation, real behavior |
| **Flow Data** | Synthetic (CIC-IDS2017 format) | Real packet capture |
| **MTD Actions** | Simulated entropy changes | Real OpenFlow group tables |
| **Latency** | Not realistic | Realistic RTT impact |

**Recommendation:**
1. Use **reference-only** first for fast iteration & validation
2. Move to **mininet-ready** for final evaluation & paper results

---

## Quick Start

### Eghtesad (2020) — Reference-Only (Offline)
```bash
cd eghtesad2020/reference-only
pip install -r requirements.txt
python run.py --episodes 10 --ml-samples 20000
```

**Output:** `results/evaluation_results.csv` + 5 figures (300 DPI)

**Time:** ~2 minutes on modern CPU

### Eghtesad (2020) — Mininet-Ready (Real Network)
```bash
# Terminal 1: Start ONOS
cd ~/onos
./bin/onos-karaf

# Terminal 2: Run Mininet test
cd eghtesad2020/mininet-ready
pip install -r requirements.txt
sudo python run_mininet.py --topology-size small --duration 300
```

**Output:** `mininet_results/threat_log.csv` with per-step threat/action logs

**Time:** ~5 minutes for 300s test

---

## Paper Implementation Status

| Paper | Reference-Only | Mininet-Ready | Status |
|-------|---|---|---|
| **Eghtesad (2020)** GameSec | ✅ Complete | ✅ Complete | Ready |
| **Li (2021)** HybridMTD | ✅ Templated | ✅ Templated | Structure Ready |
| **Chowdhary (2020)** MTD Metric | ✅ Templated | ✅ Templated | Structure Ready |

---

## File Organization Notes

### Common Files in Both Branches
- `PAPER.md` — Shared paper reference (top-level only)

### Reference-Only Files
- `mtd_env.py` — Markov game environment (gymnasium)
- `threat_detector.py` — ML ensemble (RF/XGBoost/IsolationForest)
- `rl_agent.py` — DQN trainer (Stable-Baselines3)
- `utils.py` — Data gen + plotting

### Mininet-Ready Files
- `mininet_topology.py` — Network builder (small/medium/large)
- `onos_controller_interface.py` — REST API to ONOS
- `openflow_mtd_actions.py` — RL action → OpenFlow translation
- `threat_detector_live.py` — Live packet-based threat detection (optional)

---

## Dependencies

### Reference-Only
```
stable-baselines3, gymnasium, scikit-learn, xgboost, shap, matplotlib, pandas, numpy
```

### Mininet-Ready (Additional)
```
requests (ONOS REST API)
mininet (requires: sudo apt-get install mininet)
ONOS controller (download from https://onosproject.org/)
```

---

## Methodology

### Reference-Only Approach
1. **Generate synthetic flows** (CIC-IDS2017 format, 20k samples)
2. **Train ML threat detector** (3-regime ensemble)
3. **Train RL agent** (Double DQN, 10 episodes × 200 steps)
4. **Evaluate** on 2000 test steps → CSV + 5 figures

### Mininet-Ready Approach
1. **Build Mininet topology** (3-20 switches, 10-100 hosts)
2. **Connect ONOS controller** (OpenFlow 1.3)
3. **Start threat detector** on packet mirror port
4. **Run RL loop** (10 Hz decision rate)
5. **Apply MTD actions** via ONOS REST API → OpenFlow group tables
6. **Log threat/action/latency** over test duration

---

## Next Steps (Future Papers)

### Li et al. (2021) — HybridMTD Ensemble
- 3 threat detection regimes (low/high/anomaly traffic)
- Regime switching based on traffic intensity
- Extends `threat_detector.py` with multi-regime logic
- **Structure now ready** — awaiting full implementation

### Chowdhary et al. (2020) — Path Entropy Metric
- Formalize $H_P = -\sum P(port) \log P(port)$
- Validate against randomized forwarding
- Add to `mtd_env.py` reward calculation
- **Structure now ready** — awaiting full implementation

---

## References

**Citations:**
- Eghtesad et al. (2020): https://doi.org/10.1007/978-3-030-64793-3_20
- Van Hasselt et al. (2016): Double DQN (via Stable-Baselines3)
- Lundberg & Lee (2017): SHAP explainability
- Mininet: http://mininet.org/
- ONOS: https://onosproject.org/

---

## Support

- **Reference-Only Issues:** Check `mtd_env.py` reward calculation or `threat_detector.py` feature extraction
- **Mininet Issues:** Ensure ONOS is running; check controller connectivity
- **RL Agent Issues:** Reduce learning rate, increase episode duration, or use pre-trained model

For detailed guidelines, see README.md in each branch.
