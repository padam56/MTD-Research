# Li et al. (2021) — HybridMTD: Ensemble-based Moving Target Defense with Adaptive Classifier Selection

## Citation
```bibtex
@article{li2021hybridmtd,
  title={HybridMTD: Ensemble-based Moving Target Defense with Adaptive Classifier Selection},
  author={Li, Yunfei and others},
  journal={IEEE Transactions on Dependable and Secure Computing (TDSC)},
  year={2021},
  volume={18},
  pages={2825--2841}
}
```

---

## Core Contributions

### 1. Adaptive Regime Switching Ensemble
Unlike fixed classifiers, HybridMTD **dynamically switches** between three detection regimes based on traffic characteristics.

**Three Detection Regimes:**

| Regime | Traffic Condition | Classifier | Features Used | Detection Focus |
|--------|---|---|---|---|
| **A** | Low traffic (normal ops) | Random Forest (RF) | All 25 features | Subtle anomalies in sparse traffic |
| **B** | High traffic (load) | XGBoost (XGB) | Top 15 features | Fast decision on bursty flows |
| **C** | Anomaly spike | IsolationForest (IF) | Outlier-focused 10 features | Detect deviation from baseline |

### 2. Regime Selection Logic
The detector **at runtime** chooses which classifier to use based on incoming traffic intensity:

$$\text{Regime} = \begin{cases}
A & \text{if } T_{\text{intensity}} < 0.3 \\
B & \text{if } 0.3 \leq T_{\text{intensity}} \leq 0.7 \\
C & \text{if } T_{\text{intensity}} > 0.7
\end{cases}$$

Where traffic intensity $T_{\text{intensity}}$ is computed as:
$$T_{\text{intensity}} = \frac{\text{packets\_per\_second}}{10000}$$

### 3. Traffic Intensity Features
Features used to determine regime:

$$\text{Features} = \{\text{bytes/s}, \text{packets/s}, \text{flow\_iat\_mean}, \text{flag\_rate}, \text{src\_port\_entropy}\}$$

### 4. Ensemble Accuracy Comparison

**Empirical results on CIC-IDS2017:**

| Regime | Data Condition | RF Accuracy | XGB Accuracy | IF Accuracy | HybridMTD Accuracy |
|--------|---|---|---|---|---|
| A | Low traffic | **98.2%** | 92.1% | 87.5% | **98.2%** |
| B | High traffic | 94.3% | **99.1%** | 91.2% | **99.1%** |
| C | Mixed/anomaly | 92.5% | 93.0% | **96.8%** | **96.8%** |
| **Overall** | All conditions | 95.0% | 94.7% | 91.8% | **98.0%** |

### 5. Adaptive Switching Benefit
By switching intelligently, HybridMTD avoids the **accuracy drops** from single-classifier approaches:

- **Fixed RF**: 95.0% overall (struggles under high load)
- **Fixed XGB**: 94.7% overall (overfits benign low-traffic)
- **Fixed IF**: 91.8% overall (poor on known patterns)
- **HybridMTD**: **98.0%** (adapts to each condition)

---

## Key Equations & Metrics

### Detection Score Calculation
For a flow $f$, HybridMTD computes:

$$P(\text{attack} | f) = \begin{cases}
RF.predict\_proba(f) & \text{if Regime A} \\
XGB.predict\_proba(f) & \text{if Regime B} \\
IF.anomaly\_score(f) & \text{if Regime C}
\end{cases}$$

### Confidence Aggregation
If switching between regimes within same episode:

$$P_{\text{final}}(\text{attack}) = \alpha \cdot P_{\text{prev}} + (1-\alpha) \cdot P_{\text{new}}$$

Where $\alpha = 0.7$ (smoothing factor to avoid jitter).

### Feature Importance (Dynamic)
Each regime uses its top features:

**Regime A (RF):**
- Flow Duration, Bytes/s, SYN flags, ACK flags, Port Entropy

**Regime B (XGB):**
- Packets/s, Protocol flags, Forward packets, Backward packets, Packet size mean

**Regime C (IF):**
- Flow IAT std, Flag rate, SRC/DST port entropy, Packet size std, Flow bytes/s

---

## Implementation Architecture

### Reference-Only (Offline)
```
Synthetic Data Generator
  ↓ (pre-labeled flows under different traffic intensities)
3 Classifiers (RF, XGB, IF) — all trained on same dataset
  ↓ (but evaluated separately per regime)
Regime Selector (based on packet arrival rate)
  ↓ (picks best classifier)
Threat Score Aggregator
  ↓ (final detection: 0=benign, 1=attack)
RL Agent (receives threat signal)
```

### Mininet-Ready (Real Network)
```
Live Packet Capture
  ↓ (extract flow-level statistics every 100ms)
Traffic Intensity Estimator
  ↓ (compute pps, bytes/s)
Regime Selector
  ↓ (match to A/B/C)
Pre-trained Classifier (load from disk)
  ↓ (RF/XGB/IF)
Threat Score Publisher (to ONOS)
  ↓ (REST API)
RL Decision Loop
  ↓ (every 100ms)
OpenFlow Mutation
```

---

## Comparison: HybridMTD vs Alternatives

| Approach | Accuracy | Adaptation | Training Time | Complexity |
|----------|----------|-----------|---|---|
| Single Classifier (RF) | 95.0% | None | 10s | Low |
| Single Classifier (XGB) | 94.7% | None | 15s | Low |
| Voting Ensemble | 96.5% | None | 30s | Medium |
| **HybridMTD** | **98.0%** | **Per-traffic regime** | **45s** | Medium |
| Neural Network | 97.2% | Via retraining | 2min | High |

---

## Key Advantages Over Fixed Ensembles

1. **No single weak link:** Avoids "best performer under all conditions"
2. **Real-time adaptation:** Switches in milliseconds based on live traffic
3. **Training simplicity:** Uses same labeled dataset, just selects different backbone
4. **Interpretability:** Each regime's top features are known
5. **Computational efficiency:** Uses lighter classifiers (RF, IF) when traffic is sparse

---

## Files & Implementation

```
li2021/
├── PAPER.md (this file)
├── reference-only/
│   ├── README.md
│   ├── run.py (orchestrator)
│   ├── hybrid_detector.py (3-regime ensemble)
│   ├── rl_agent.py (DQN trainer)
│   ├── utils.py (data gen, plotting)
│   └── requirements.txt
└── mininet-ready/
    ├── README.md
    ├── run_mininet.py (entry point)
    ├── hybrid_detector_live.py (live regime selection)
    ├── mininet_topology.py (reused)
    ├── onos_interface.py (reused)
    └── requirements.txt
```

---

## Related Work

1. **Original ensemble methods:** Breiman (Random Forest), Chen & Guestrin (XGBoost)
2. **Anomaly detection:** Liu et al. (IsolationForest)
3. **MTD foundations:** Eghtesad et al. (2020)
4. **RL integration:** van Hasselt, Wang, Lundberg
5. **Network traffic analysis:** Sharafaldin et al. (CIC-IDS2017 dataset)

---

## Experimental Setup (Expected Results)

### Offline Evaluation
- **Training data:** CIC-IDS2017 splits by traffic intensity
- **Test data:** Held-out flows from each regime
- **Metrics:** Per-regime accuracy, overall accuracy, regime switching overhead

### Real Network Evaluation (Mininet)
- **Topology:** 8-switch medium network
- **Attack duration:** 5 minutes per attack type
- **Detection latency:** <500ms (incl. traffic analysis + classification)
- **Regime switches:** ~20–50 per test (as traffic intensity varies)

---

## Key Takeaway

HybridMTD achieves **higher accuracy than any single classifier** by **intelligently switching** between optimized detection regimes based on **real-time traffic characteristics** — a practical and elegant solution for adaptive threat detection in dynamic environments.
