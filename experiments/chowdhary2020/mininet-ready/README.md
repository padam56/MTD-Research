# Chowdhary et al. (2020) — Path Entropy MTD Metric: Mininet-Ready Implementation

## Overview

Real-network measurement of **path entropy evolution** using live Mininet traffic.

Captures actual port distributions → computes Shannon entropy → validates MTD effectiveness.

---

## Setup

Requires:
- ONOS controller running
- Mininet + packet capture tools
- Python 3.8+

## Quick Start

```bash
# Terminal 1: Start ONOS
~/onos/bin/onos-karaf

# Terminal 2: Run test
pip install -r requirements.txt
sudo python run_mininet.py --topology-size small --monitor-entropy --duration 300
```

---

## Key Components

### `entropy_monitor_live.py`
- Live packet sniffer + flow statistics
- Computes egress port distribution every 10 steps
- Calculates $H_P$ and $dH_P/dt$ in real-time
- Alerts when entropy stalls (attacker convergence)

### `run_mininet.py`
- Topology setup
- Packet capture initialization
- Entropy monitoring thread
- Logs port distributions over time

---

## Files

See `../PAPER.md` for equations and methodology.

---

## Expected Output

```
mininet_results/entropy_log.csv:
  timestamp, H_P, dH_P/dt, max_port_util, active_ports
  0.0,       0.30, 0.00,    0.89,         1
  10.0,      1.85, 0.155,   0.71,         4  ← Action 1!
  20.0,      1.92, 0.007,   0.70,         4  ← plateau
  30.0,      2.08, 0.016,   0.60,         4  ← Action 2!
```

**Expected**: $H_P$ jumps on MTD action, then slowly increases as attacker learns paths.

---

## Detailed README Coming Soon

Full setup guide + entropy interpretation will be added after core Eghtesad validation.
