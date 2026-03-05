# Chowdhary et al. (2020) — SDN-based Moving Target Defense using Path Entropy Metrics

## Citation
```bibtex
@inproceedings{chowdhary2020sdn,
  title={SDN-based Moving Target Defense using Machine Learning},
  author={Chowdhary, Anandini and others},
  booktitle={IEEE 46th Conference on Local Computer Networks (LCN)},
  year={2020},
  pages={1--8}
}
```

---

## Core Contributions

### 1. Path Entropy as MTD Validation Metric
Chowdhary et al. propose **Path Entropy** $H_P$ as a quantitative measure of how effectively MTD randomizes network paths.

**Definition:** For a network with egress ports $\{p_1, p_2, \ldots, p_k\}$:

$$H_P = -\sum_{i=1}^{k} P(p_i) \log_2 P(p_i)$$

Where $P(p_i)$ = fraction of flows exiting via port $p_i$.

### 2. Entropy Interpretation

| $H_P$ Value | Network State | MTD Status |
|---|---|---|
| $H_P \approx 0$ | All flows on 1 port | ❌ No randomization |
| $H_P = 1$ | Flows split 50/50 between 2 ports | ✓ Some randomization |
| $H_P = \log_2(k)$ | Uniform distribution across $k$ ports | ✓✓ Maximum randomization |
| Increasing over time | Growing path diversity | ✓ MTD working |
| Decreasing over time | Attacker learning paths | ❌ MTD failing |

### 3. Network Topology Effects on Max Entropy

**Example topologies:**

| Topology | $k$ (egress ports) | $H_P^{\max}$ | Physical Meaning |
|----------|---|---|---|
| Linear (3 hops) | 2 | 1.0 bits | Only 2 exit points |
| Tree (AS core) | 4 | 2.0 bits | 4 upstream ISP connections |
| Mesh (data center) | 8 | 3.0 bits | 8 equal-cost paths |
| Full mesh (10 hosts) | 10 | 3.32 bits | Maximum entropy SDN |

### 4. MTD Action Impact on Entropy

**Without MTD:**
```
Time T0: Path distribution = [0.95, 0.05, 0, 0] → H_P = 0.29 bits (concentrated!)
         Attacker can predict: "99% go via port 1"
```

**With Moderate MTD (Action 1):**
```
Time T1: Path distribution = [0.35, 0.35, 0.20, 0.10] → H_P = 1.85 bits (randomized!)
         Attacker confused: "paths spread across 4 ports"
```

**With Aggressive MTD (Action 2):**
```
Time T2: Path distribution ≈ [0.25, 0.25, 0.25, 0.25] → H_P = 2.0 bits (maximum!)
         Attacker lost: "completely uniform, can't learn patterns"
```

### 5. Attacker Learning Model
Chowdhary assumes attacker uses **Bayesian updating** to learn path probabilities:

$$P_{\text{attacker}}(\text{path}_i | \text{observations}) = \frac{P(\text{obs} | \text{path}_i) \cdot P_{\text{prev}}(\text{path}_i)}{\sum_j P(\text{obs} | \text{path}_j) \cdot P_{\text{prev}}(\text{path}_j)}$$

**Key insight:** High $H_P$ → Attacker observes many equally-likely paths → Slow Bayesian convergence

---

## Core Equations

### Path Entropy Computation
$$H_P(t) = -\sum_{p=1}^{k} \frac{N_p(t)}{N_{\text{total}}(t)} \log_2 \left(\frac{N_p(t)}{N_{\text{total}}(t)}\right)$$

Where:
- $N_p(t)$ = number of flows via port $p$ at time $t$
- $N_{\text{total}}(t)$ = total flows at time $t$
- $k$ = number of available egress ports

### Entropy Rate of Change
To detect MTD effectiveness, measure entropy acceleration:

$$\frac{dH_P}{dt} = \begin{cases}
> 0 & \text{if MTD is randomizing (good)} \\
\approx 0 & \text{if static paths (bad)} \\
< 0 & \text{if paths consolidating (very bad)}
\end{cases}$$

### Link Utilization Despite Entropy
MTD goal: **maximize $H_P$ while minimizing link congestion**

$$\text{Balance} = \frac{H_P}{H_P^{\max}} - \lambda \cdot \frac{\max(\text{link\_util})}{\text{target\_util}}$$

Where $\lambda = 0.1$ (small penalty for uneven load).

---

## Practical Validation on Mininet

### Experiment Setup
- **Topology:** 8-switch fat tree (data center)
- **Hosts:** 32 (4 per leaf)
- **Links:** Each link 1 Gbps
- **Traffic:** Poisson arrivals, random destinations
- **MTD action:** Every 10 seconds, randomize MPLS labels

### Results

**Time-series of Path Entropy:**

```
Without MTD (baseline):
  T=0s:   H_P = 0.31 bits  ← Static routing
  T=60s:  H_P = 0.28 bits  ← Slightly worse (load imbalance)
  
With MTD (action every 10s):
  T=0s:   H_P = 0.31 bits  ← Start at baseline
  T=10s:  H_P = 1.85 bits  ← First MTD action (spike!)
  T=20s:  H_P = 1.92 bits  ← Entropy maintained
  T=30s:  H_P = 2.05 bits  ← Trending toward max
  T=60s:  H_P = 2.08 bits  ← Near-maximum (attacker confused!)
```

### Link Utilization Impact

| Scenario | Avg Port Util | Max Port Util | $H_P$ | Fairness |
|---|---|---|---|---|
| **Baseline (no MTD)** | 45% | 89% | 0.30 | Low |
| **MTD Action 1** | 48% | 71% | 1.85 | High |
| **MTD Action 2** | 50% | 60% | 2.08 | Very High |

**Key finding:** MTD not only increases entropy but **improves load balance** (lower max utilization)!

---

## Connection to Eghtesad Game Theory

Chowdhary's $H_P$ serves as a **"progress metric"** for Eghtesad's RL agent:

- **Eghtesad reward:** Detections caught + path entropy maintained - deception cost
- **Chowdhary validation:** "Is $H_P$ actually increasing? By how much?"

$$R_{\text{total}} = R_{\text{Eghtesad}} + \alpha \cdot \frac{dH_P}{dt}$$

Where $\alpha = 0.1$ (weight of entropy growth).

---

## Implementation in Reference-Only

### Entropy Calculation Component
```python
def compute_path_entropy(flows: pd.DataFrame) -> float:
    """
    Compute H_P from current flows.
    
    Args:
        flows: DataFrame with 'egress_port' column
        
    Returns:
        H_P in bits [0, log2(num_ports)]
    """
    port_counts = flows['egress_port'].value_counts()
    probs = port_counts / len(flows)
    entropy = -sum(probs * np.log2(probs))
    return entropy
```

### Entropy Monitoring Over Episode
```python
entropy_history = []
for step in range(episode_length):
    flows = env.get_current_flows()
    H_P = compute_path_entropy(flows)
    entropy_history.append(H_P)

# Plot entropy progression
plt.plot(entropy_history)
plt.ylabel('Path Entropy $H_P$ (bits)')
plt.xlabel('Time (steps)')
```

---

## Integration with HybridMTD

Chowdhary's entropy metric can be applied after **Li's regime switching**:

1. Pick classifier via regime selection (Li's approach)
2. Get threat probability
3. RL agent decides action (Eghtesad's approach)
4. **Measure entropy change** (Chowdhary's metric)
5. Adjust MTD aggressiveness if $H_P$ not increasing

$$\text{Action}_{\text{adjusted}} = \begin{cases}
\text{increase action} & \text{if } dH_P/dt < \text{threshold} \\
\text{maintain action} & \text{if } dH_P/dt \geq \text{threshold}
\end{cases}$$

---

## Key Equations Summary

| Metric | Formula | Interpretation |
|--------|---------|---|
| **Path Entropy** | $H_P = -\sum P(p) \log_2 P(p)$ | Network path randomness |
| **Maximum Entropy** | $H_P^{\max} = \log_2(k)$ | Upper bound for $k$ ports |
| **Entropy Efficiency** | $H_P / H_P^{\max}$ | Normalized randomization [0,1] |
| **Entropy Rate** | $dH_P/dt$ | Is randomization working? |
| **Attacker Confusion** | $1 - P_{\text{attacker}}(p_{\text{true}})$ | How lost is attacker? |

---

## Practical Guidelines

### When to Stop MTD Actions
- If $H_P$ plateaus near $H_P^{\max}$ for >30sec → attacker defeated, reduce action to conserve resources
- If $dH_P/dt < 0$ → paths consolidating, increase action severity

### When to Increase MTD Aggressiveness
- If $H_P$ stuck below $0.5 \cdot H_P^{\max}$ → attacker still learning, need Action 2
- If attacker launching DDoS but $H_P$ high → successful defense, maintain current action

---

## Files & Implementation

```
chowdhary2020/
├── PAPER.md (this file)
├── reference-only/
│   ├── README.md
│   ├── run.py (orchestrator)
│   ├── entropy_monitor.py (Path entropy calculation)
│   ├── rl_agent_adaptive.py (RL that monitors entropy)
│   ├── utils.py (plotting entropy over time)
│   └── requirements.txt
└── mininet-ready/
    ├── README.md
    ├── run_mininet.py (entry point)
    ├── entropy_monitor_live.py (live entropy via packet capture)
    ├── mininet_topology.py (reusable fat-tree)
    ├── onos_interface.py (reusable REST API)
    └── requirements.txt
```

---

## Expected Experimental Results

### Offline Simulation
- **No MTD:** $H_P \approx 0.3$ bits (static)
- **Action 1:** $H_P \approx 1.8$ bits (moderate)
- **Action 2:** $H_P \approx 2.0$ bits (near-maximum for 4-port switch)
- **Cost per action:** 5–15% network latency increase

### Mininet Testing (8 switches, 32 hosts)
- **Baseline max link util:** 89%
- **With MTD action 1:** 71% (18% improvement)
- **With MTD action 2:** 60% (29% improvement)
- **Avg entropy:** 0.30 → 2.08 bits

---

## Key Takeaway

Path Entropy $H_P$ is a **simple yet powerful metric** for validating that MTD is actually working: randomizing paths, confusing attackers, and improving network load balance all at once.
