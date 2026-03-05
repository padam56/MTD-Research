# Eghtesad (2020) — Mininet-Ready Implementation (ONOS Integration)

## Overview

This is a **real-network deployment** of the Eghtesad game adapted for **Mininet + ONOS controller**. It bridges the offline RL agent with live OpenFlow switches, enabling:
- Real packet capture → threat detection
- RL decisions → ONOS OpenFlow group table mutations
- Live topology learning attacks against the MTD

**Perfect for:** Testbed validation, controller integration, real-time performance evaluation.

---

## Prerequisites

### Software Stack
1. **Mininet** (2.3+): Network simulator with OpenFlow support
   ```bash
   sudo apt-get install mininet
   ```

2. **ONOS** (1.15+): SDN controller with OpenFlow 1.3+ support
   - Download: https://onosproject.org/download/
   - Start: `onos-karaf` (requires Java 11+)

3. **Python packages:**
   ```bash
   pip install -r requirements.txt
   # Additional: scapy (packet capture)
   pip install scapy
   ```

### Network Connectivity
- Mininet hosts: 10.0.0.1–10.0.0.N (default OpenFlow 1.3)
- ONOS controller: 127.0.0.1:6653 (OpenFlow connection)
- Threat detector: Listens on traffic mirror port

---

## Architecture

```
Mininet Topology (5–50 switches, 10–100 hosts)
         ↓
OpenFlow Packets
         ↓ (mirrored to packet sniffer)
Threat Detector Live (packet capture analysis)
         ↓ (threat probability)
RL Agent (trained offline or online)
         ↓ (action 0/1/2)
ONOS Controller REST API
         ↓ (POST /onos/v1/flows)
OpenFlow Group Table Update
         ↓ (MPLS/IP rewrite, path mutation)
Switch Port Randomization (IP/port remapping)
         ↓
New packet forwarding via randomized paths
```

---

## File Descriptions

### `run_mininet.py` (MAIN)
**Entry point for Mininet testbed.**

Orchestrates:
1. Spawn Mininet topology (10 hosts, 3 switches)
2. Connect to ONOS controller
3. Start threat detector packet sniffer thread
4. Run RL agent decision loop (10 Hz, 100ms per decision)
5. Log results: threat level, actions, path mutations, latency

**Usage:**
```bash
# Must be run with sudo (Mininet requirement)
sudo python run_mininet.py \
  --topology-size small \    # {small, medium, large}
  --controller-ip 127.0.0.1 \
  --controller-port 6653 \
  --attack-type portscanning \
  --duration 300              # Run for 5 minutes
```

**Outputs:**
```
mininet_results/
├── threat_log.csv (timestamp, threat_prob, action, latency_ms)
├── flow_mutations.log
└── topology_snapshots/
    ├── initial.json (pre-attack topology)
    ├── t=60s.json (topology after 1 min MTD actions)
    └── t=300s.json (final topology)
```

### `mininet_topology.py`
**Mininet network builder.**

**Classes:**
- `MTDTopology` — Generates multi-switch, multi-host topology
  - Small: 3 switches, 10 hosts (test)
  - Medium: 8 switches, 30 hosts (evaluation)
  - Large: 20 switches, 100 hosts (stress test)

**Key methods:**
- `build_topology(size)` — Create switches, hosts, links
- `setup_openflow_rules()` — Initial switch rules
- `get_host_by_ip(ip)` — Find host object
- `get_path(src_ip, dst_ip)` — Current path in topology

**Example:**
```python
topo = MTDTopology()
topo.build_topology(size='small')
net = Mininet(topo=topo, controller=RemoteController('onos', ip='127.0.0.1'))
net.start()
```

### `onos_controller_interface.py`
**ONOS REST API wrapper.**

Communicates with ONOS to:
- Query network topology
- Retrieve flow statistics
- Push new OpenFlow group tables for path mutations

**Classes:**
- `ONOSController` — REST client for ONOS
  - `get_devices()` — List all switches
  - `get_hosts()` — List all hosts
  - `get_flows()` — Retrieve current flows
  - `post_flow_group(switch_id, group_table)` — Mutate paths
  - `get_topology()` — Get topology JSON

**Example:**
```python
onos = ONOSController(ip='127.0.0.1', port=8181, username='karaf', password='karaf')
devices = onos.get_devices()
print(f"Found {len(devices)} switches")

# Mutate path: push new MPLS label or IP rewrite
group_table = {
    "switch_id": "of:0000000000000001",
    "action": "IP_REWRITE",
    "group_rules": [
        {"port": 1, "vlan_id": 100},
        {"port": 2, "vlan_id": 101}
    ]
}
onos.post_flow_group(**group_table)
```

### `openflow_mtd_actions.py`
**Translates RL actions to OpenFlow mutations.**

**Classes:**
- `OpenFlowMTD` — Converts action {0,1,2} to group table updates
  - Action 0: No-op (keep current flows)
  - Action 1: Moderate mutation (swap 30% of egress ports)
  - Action 2: Aggressive mutation (randomize all MPLS labels)

**MTD Methods:**
```python
class OpenFlowMTD:
    def apply_action_0(self):
        # No change
        pass
    
    def apply_action_1(self):
        # Moderate: MPLS label swap for 30% of flows
        affected_flows = random.sample(self.current_flows, 0.3 * len(self.current_flows))
        for flow in affected_flows:
            new_label = random.randint(100, 200)
            self.group_table.append({
                "match": flow,
                "action": f"MPLS_LABEL={new_label}"
            })
        self.onos.push_group_table(self.group_table)
    
    def apply_action_2(self):
        # Aggressive: Full network path re-randomization
        # Change routing so packets take different switch paths
        for switch_id in self.switches:
            new_route = self.compute_disjoint_path()
            self.onos.update_switch_routing(switch_id, new_route)
```

### `threat_detector_live.py`
**Real-time threat detection from live packets.**

Differs from reference-only version:
- Sniffs live packets (via Scapy or tcpdump)
- Extracts per-flow statistics in real-time **without** MockONOSClient
- Computes threat probability from ML model (pre-trained)

**Classes:**
- `LiveThreatDetector` — Packet sniffer + threat scorer
  - `start_sniffer()` — Begin capture on interface
  - `extract_flow_features()` — Parse packets → CIC-IDS2017 format
  - `predict_threat()` — ML ensemble prediction + SHAP
  - `get_current_threat()` — Return latest threat probability

**Example:**
```python
detector = LiveThreatDetector(
    interface='eth0',  # Mirrored port
    ml_model_path='./threat_detector.pkl',
    window_size=10  # Aggregate statistics over last 10 packets
)
detector.start_sniffer()

while True:
    threat_prob = detector.get_current_threat()
    print(f"Threat level: {threat_prob:.2f}")
    time.sleep(1)
```

### `rl_agent_mininet.py` (OPTIONAL)
**Real-time RL decision loop for live network.**

Options:
1. **Load pre-trained agent:** Use model trained offline in `reference-only/`
2. **Online training:** Adapt agent in real-time (requires slow exploration)

**Key loop:**
```python
while running:
    state = {
        'threat': detector.get_current_threat(),
        'H_A': compute_attacker_knowledge(),
        'H_P': compute_path_entropy(),
        'traffic': stats.bytes_per_sec
    }
    action = agent.predict(state)  # 0, 1, or 2
    mtd.apply_action(action)
    time.sleep(0.1)  # 10 Hz decision rate
```

---

## Step-by-Step Deployment

### 1. Start ONOS Controller
```bash
cd ~/onos
./bin/onos-karaf  # Starts ONOS on 127.0.0.1:8181
```

### 2. Check ONOS Web UI
- URL: http://localhost:8181/onos/ui/
- Default credentials: admin/onos

### 3. Train Offline Agent (Optional)
```bash
cd ../reference-only
python run.py --episodes 50 --timesteps-per-episode 500
# Generates: ./models/dqn_agent.zip
```

### 4. Start Mininet + MTD
```bash
sudo python run_mininet.py \
  --topology-size small \
  --controller-ip 127.0.0.1 \
  --controller-port 6653 \
  --model-path ../reference-only/models/dqn_agent.zip \
  --attack-type portscanning \
  --duration 600  # 10 minutes
```

### 5. Monitor Results (in another terminal)
```bash
tail -f mininet_results/threat_log.csv
# Output:
# timestamp,threat_prob,action,latency_ms
# 0.0,0.32,1,45
# 0.1,0.41,1,52
# 0.2,0.89,2,38
```

---

## Attack Simulation Options

### Passive Reconnaissance
```bash
python run_mininet.py --attack-type portscanning
# Host 10.0.0.5 attempts to scan all ports on 10.0.0.1–10.0.0.10
# MTD should increase H_A (paths keep changing)
```

### Active DDoS
```bash
python run_mininet.py --attack-type ddos
# Host 10.0.0.5 sends 1000pkt/s to 10.0.0.1
# MTD should (a) detect threat, (b) apply action 2, (c) drop/reroute traffic
```

### Exfiltration
```bash
python run_mininet.py --attack-type exfiltration
# Host 10.0.0.10 (compromised) sends data to 192.168.1.100 (attacker)
# MTD should detect unusual egress patterns → block or reroute
```

---

## Expected Results

### RTT Impact
- No MTD: RTT ≈50 ms
- Moderate (Action 1): RTT ≈55 ms (+10%)
- Aggressive (Action 2): RTT ≈65 ms (+30%)

### Detection Performance
- Detection latency: 200–500 ms (threat detector window)
- False positive rate: ~2–5% (tunable on ML threshold)
- False negative rate: ~1–3%

### RL Agent Behavior
- **Early episodes:** Explores random actions (H_A high)
- **Later episodes:** Biases toward action 1 (balanced cost-benefit)
- **Under attack:** Shifts toward action 2 (aggressive MTD)

---

## Troubleshooting

### ONOS Controller Not Responding
```bash
# Check if ONOS is running:
curl http://127.0.0.1:8181/onos/v1/devices

# If 404, restart ONOS:
./bin/onos-karaf
karaf@root> app activate org.onosproject.openflow
```

### Mininet Switches Connect but No Flow Rules
- ONOS may require explicit flow installation
- Add to `run_mininet.py`:
  ```python
  onos.post_intent(
      intent_type="PointToPointIntent",
      src="10.0.0.1", dst="10.0.0.10"
  )
  ```

### Packet Sniffer Missing Traffic
- Verify mirror port is configured on switch
- Check tcpdump: `sudo tcpdump -i eth0 "tcp or udp"`

### RL Agent Unstable / Diverges
- Use pre-trained model instead of online training
- Reduce learning rate if online: `learning_rate=1e-5`

---

## Integration with Reference-Only

**Recommended workflow:**
1. **Phase 1:** Train in `reference-only/` (50 episodes, 10k steps)
2. **Phase 2:** Export trained model: `agent.save("../mininet-ready/models/agent")`
3. **Phase 3:** Load in `mininet-ready/` and test on real topology
4. **Phase 4:** (Optional) Fine-tune online via `rl_agent_mininet.py`

---

## Paper Reference
See `../PAPER.md` for equations and related work.
