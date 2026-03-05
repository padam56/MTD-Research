# Chowdhary et al. (2020) — Path Entropy MTD Metric: Reference-Only Implementation

## Overview

**Implementation of Path Entropy validation metric** for MTD effectiveness.

Measures how well MTD is randomizing network paths: $H_P = -\sum P(p) \log_2 P(p)$

**Perfect for:** Validating MTD success, understanding entropy dynamics, baseline comparisons.

---

## Quick Start

```bash
pip install -r requirements.txt
python run.py --episodes 10 --output-dir ./results
```

Outputs: CSV + entropy growth visualization in `results/`

---

## Key Components

### `entropy_monitor.py`
- **compute_path_entropy(flows)** — Shannon entropy of egress ports
- Tracks $dH_P/dt$ (entropy rate of change)
- Flags when entropy plateauing (attacker defeated) or declining (MTD failing)

### `run.py`
- Runs episode with MTD actions (0/1/2)
- Measures $H_P$ at each step
- Plots entropy progression
- Compares action impact: No MTD vs Moderate vs Aggressive

### `utils.py`
- Synthetic network flows
- Entropy calculation & visualization
- Per-action impact summary

---

## Files

See `../PAPER.md` for equations and practical guidelines.

---

## Expected Output

```
No MTD:        H_P = 0.30 ± 0.05 bits (static, concentrated paths)
Action 1:      H_P = 1.80 ± 0.15 bits (randomized)
Action 2:      H_P = 2.05 ± 0.08 bits (near-maximum!)

dH_P/dt:
  No MTD:      ≈ 0 (paths unchanged)
  Action 1:    > 0 (entropy growing for 10-20 steps, then plateaus)
  Action 2:    > 0 (faster entropy growth)
```

---

## Key Insight

Path Entropy validates that MTD is **actually working**: randomizing paths and confusing attackers. Without this metric, you can't prove paths are changing!

---

## Detailed README Coming Soon

Full implementation details + case studies will be added after core Eghtesad validation.
