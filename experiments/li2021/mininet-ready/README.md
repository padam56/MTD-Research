# Li et al. (2021) — HybridMTD: Mininet-Ready Implementation

## Overview

Real-network deployment of **HybridMTD adaptive ensemble** with **live traffic-based regime selection**.

Streams packets → regime classifier → threat decision → ONOS via REST API.

---

## Setup

Requires:
- ONOS controller running on 127.0.0.1:8181
- Mininet installed (`sudo apt-get install mininet`)
- Python 3.8+

## Quick Start

```bash
# Terminal 1: Start ONOS
~/onos/bin/onos-karaf

# Terminal 2: Run test
pip install -r requirements.txt
sudo python run_mininet.py --topology-size small --duration 300
```

---

## Key Components

### `hybrid_detector_live.py`
- Live packet capture from mirrored port
- Dynamic traffic intensity calculation
- Real-time regime selector
- Pre-trained 3 classifiers (loaded from disk)

### `run_mininet.py`
- Starts Mininet topology
- Initializes packet sniffer
- Runs RL decision loop (10 Hz)
- Logs threat/action/latency

---

## Files

See `../PAPER.md` for equations and methodology.

---

## Expected Output

```
mininet_results/threat_log.csv:
  timestamp, threat_prob, regime, action, latency_ms
  0.0,       0.32,        A,      1,      45
  0.1,       0.41,        A,      1,      52
  1.5,       0.85,        B,      2,      38      ← Regime switched!
```

**Regime switches:** 5–10 per minute (as traffic fluctuates)

---

## Detailed README Coming Soon

Full setup guide + troubleshooting will be added after core Eghtesad validation.
