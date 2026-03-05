"""
mock_onos_client.py
-------------------
Drop-in replacement for ONOSClient that requires NO live ONOS controller.

Simulates four realistic network phases in a repeating cycle:
    Phase 0 — NORMAL      (~60s)  : benign background traffic
    Phase 1 — RECON       (~30s)  : attacker port-scanning, rising flow counts
    Phase 2 — ATTACK      (~40s)  : DDoS/intrusion, anomalous byte spikes
    Phase 3 — POST-ATTACK (~20s)  : traffic subsides, paths still hot

The mock produces all the same dict structures as the real ONOSClient so
mtd_env.py, threat_detector.py, and main.py work without any changes.

Usage:
    from src.mock_onos_client import MockONOSClient
    client = MockONOSClient()                  # no config file needed
"""

import time
import random
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MockONOS] %(message)s")


# ── Phase durations (in STEPS, not seconds) ───────────────────────────
# Using steps instead of wall-clock time ensures phases cycle correctly
# whether the simulation runs fast (offline) or slow (live testbed).
PHASE_DURATIONS = {
    "normal":      60,   # 60 steps of calm background traffic
    "recon":       30,   # 30 steps of port-scan activity
    "attack":      40,   # 40 steps of active DDoS / intrusion
    "post_attack": 20,   # 20 steps of traffic subsiding
}
PHASE_ORDER = ["normal", "recon", "attack", "post_attack"]

# ── Realistic device IDs ───────────────────────────────────────────────
DEVICE_IDS = [
    "of:0000000000000001",
    "of:0000000000000002",
    "of:0000000000000003",
    "of:0000000000000004",
]

# ── IP pools (source diversity = attacker entropy signal) ──────────────
NORMAL_SRC_IPS = [f"10.0.0.{i}" for i in range(1, 20)]
ATTACK_SRC_IPS = [f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
                  for _ in range(200)]   # large scanning pool


class MockONOSClient:
    """
    Stateful simulation of the ONOS Northbound REST API.
    Cycles through network phases autonomously.
    No network connection required.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._step_count   = 0           # step-based phase counter
        self._phase_index  = 0
        self._phase_step_start = 0       # step at which current phase began
        self._mutations    = 0
        self._last_action  = 0
        self.mutation_log: list = []
        self._last_state: dict = {}

        logging.info("MockONOSClient initialised — no controller required.")

    # ------------------------------------------------------------------
    # Reset — called at the start of every RL episode
    # ------------------------------------------------------------------

    def reset(self):
        """Restart the phase cycle from scratch. Called by env.reset() between episodes."""
        self._step_count       = 0
        self._phase_index      = 0
        self._phase_step_start = 0
        self._mutations        = 0
        self._last_action      = 0
        self.mutation_log      = []
        logging.info("MockONOSClient reset — phase cycle restarted.")

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _current_phase(self) -> str:
        """Returns the current phase name without advancing the step counter."""
        return PHASE_ORDER[self._phase_index]

    def _advance_step(self):
        """Advance one simulation step and transition phases when due."""
        self._step_count += 1
        steps_in_phase = self._step_count - self._phase_step_start
        current = PHASE_ORDER[self._phase_index]
        if steps_in_phase >= PHASE_DURATIONS[current]:
            self._phase_index = (self._phase_index + 1) % len(PHASE_ORDER)
            self._phase_step_start = self._step_count
            new_phase = PHASE_ORDER[self._phase_index]
            logging.info(f"Phase transition → {new_phase.upper()} (step {self._step_count})")

    def _phase_factor(self) -> float:
        """
        Returns 0.0 (calm) → 1.0 (peak attack) based on current phase.
        Used to scale all synthetic metrics.
        """
        phase = self._current_phase()
        steps_in_phase = self._step_count - self._phase_step_start
        t = steps_in_phase / max(PHASE_DURATIONS[phase], 1)  # 0→1 within phase

        if phase == "normal":
            return 0.05 + random.gauss(0, 0.02)           # low, steady
        elif phase == "recon":
            return 0.2 + 0.3 * t + random.gauss(0, 0.05) # rising
        elif phase == "attack":
            return 0.7 + 0.3 * math.sin(math.pi * t)     # peak then taper
        else:  # post_attack
            return 0.4 - 0.3 * t                          # falling

    # ------------------------------------------------------------------
    # Observation — mirrors ONOSClient.get_network_state()
    # ------------------------------------------------------------------

    def get_network_state(self) -> dict:
        self._advance_step()   # one step per observation, not per sub-call
        self._last_state = {
            "flows":      self._generate_flows(),
            "port_stats": self._generate_port_stats(),
            "timestamp":  time.time(),
        }
        return self._last_state

    def _generate_flows(self) -> list:
        """
        Generates synthetic flow rules.
        During attack phases: many flows from diverse IPs (high entropy).
        During normal phases: few flows from stable known IPs (low entropy).
        """
        phase  = self._current_phase()
        pf     = max(0.0, min(1.0, self._phase_factor()))

        # Base flow count: 50 normal → up to 3000 under attack
        base_count = int(50 + 2950 * pf)
        # Add Gaussian noise
        flow_count = max(10, int(random.gauss(base_count, base_count * 0.1)))

        src_pool = ATTACK_SRC_IPS if phase in ("attack", "recon") else NORMAL_SRC_IPS

        flows = []
        for i in range(min(flow_count, 2000)):   # cap list size for speed
            src = random.choice(src_pool)
            dst = f"10.0.1.{random.randint(1, 30)}"
            protocol = random.choice(["TCP", "UDP", "ICMP"])
            flows.append({
                "id":       f"flow_{i:05d}",
                "deviceId": random.choice(DEVICE_IDS),
                "priority": random.randint(100, 40000),
                "timeout":  random.randint(10, 300),
                "treatment": {"instructions": [{"type": "OUTPUT", "port": str(random.randint(1, 4))}]},
                "selector":  {
                    "criteria": [
                        {"type": "ETH_TYPE",  "ethType": "0x800"},
                        {"type": "IPV4_SRC",  "ip": f"{src}/32"},
                        {"type": "IPV4_DST",  "ip": f"{dst}/32"},
                        {"type": "IP_PROTO",  "protocol": protocol},
                        # SYN flag count — high under SYN-flood attacks
                        {"type": "TCP_FLAGS", "flags": random.randint(
                            0, 900 if phase == "attack" else 10
                        )},
                    ]
                },
                # CIC-IDS2017-style flow-level features (used by ThreatDetector).
                # Values are per-flow to match the per-row training scale from
                # generate_synthetic_data.py — use _mean() not _sum() at inference.
                "features": {
                    # flow_bytes_per_sec: DoS 50k-500k, BENIGN 100-5000
                    "flow_bytes_per_sec":  random.gauss(
                        50000 * pf, 10000) if phase == "attack" else random.gauss(500, 100),
                    "flow_pkts_per_sec":   random.gauss(100 * pf + 1, 20),
                    # Tiny packets in DoS (40-80 B), large in BENIGN (200-1200 B)
                    "fwd_pkt_len_mean":    random.gauss(
                        60  if phase == "attack" else
                        50  if phase == "recon"  else 800, 50),
                    # Per-flow flag counts — match training: DoS syn=400-1000, BENIGN=0-3
                    "syn_flag_count":      random.randint(
                        500 if phase == "attack" else 1 if phase == "recon" else 0,
                        999 if phase == "attack" else 3 if phase == "recon" else 2),
                    "rst_flag_count":      random.randint(
                        100 if phase == "attack" else 0,
                        299 if phase == "attack" else 3),
                    "psh_flag_count":      random.randint(0, 4 if phase == "attack" else 1),
                    "ack_flag_count":      random.randint(
                        0, 9 if phase == "attack" else 50),
                    # Short duration during SYN-flood, long for background BENIGN
                    "flow_duration":       random.uniform(0.001, 1.0) if phase == "attack" else
                                           random.uniform(0.0001, 0.5) if phase == "recon" else
                                           random.uniform(5.0, 60.0),
                    # Idle/active times
                    "idle_mean":           random.gauss(
                        0.001 if phase == "attack" else
                        1.5   if phase == "recon"  else 0.5, 0.01),
                    "active_mean":         random.gauss(
                        0.01  if phase == "attack" else
                        0.001 if phase == "recon"  else 1.0, 0.001),
                    # Subflow bytes: bulk flood in attack, light in BENIGN
                    "subflow_fwd_bytes":   max(0, int(random.gauss(
                        2000 if phase == "attack" else 500, 200))),
                },
            })

        return flows

    def _generate_port_stats(self) -> list:
        """Port byte/packet counters — scale with attack phase."""
        pf  = max(0.0, min(1.0, self._phase_factor()))
        stats = []
        for device in DEVICE_IDS:
            ports = []
            for port in range(1, 5):
                ports.append({
                    "port":      str(port),
                    "bytesReceived":     int(random.gauss(1e6 * (1 + 10 * pf), 1e5)),
                    "bytesSent":         int(random.gauss(1e6 * (1 + 8  * pf), 1e5)),
                    "packetsReceived":   int(random.gauss(5000 * (1 + 10 * pf), 500)),
                    "packetsSent":       int(random.gauss(5000 * (1 + 8  * pf), 500)),
                    "packetsRxDropped":  int(100 * pf + random.expovariate(1)),
                    "packetsTxDropped":  int(50  * pf + random.expovariate(1)),
                })
            stats.append({"device": device, "ports": ports})
        return stats

    def get_controller_latency(self) -> float:
        """
        Simulated latency in ms.
        Normal: 5–25ms. Under attack: spikes to 200ms (resource exhaustion).
        Ref: Eghtesad (2020) overhead cost term W_overhead * latency.
        """
        pf = max(0.0, min(1.0, self._phase_factor()))
        base = random.gauss(15, 5)          # normal latency
        spike = random.gauss(200 * pf, 20) if pf > 0.5 else 0
        return round(max(1.0, base + spike), 2)

    # ------------------------------------------------------------------
    # Actions — mirrors ONOSClient.execute_mutation()
    # ------------------------------------------------------------------

    def execute_mutation(self, mutation_level: int) -> bool:
        """
        Simulates pushing a mutation to ONOS.
        Also applies side-effects: mutations during attack phases reduce
        the phase factor (simulating the MTD working), which naturally
        lowers threat scores in subsequent observations.
        """
        if mutation_level == 0:
            return True

        self._mutations += 1
        self._last_action = mutation_level

        # Aggressive mutation fast-forwards out of the attack phase by 15 steps
        if mutation_level == 2 and self._current_phase() == "attack":
            self._phase_step_start -= 15
            logging.info("MTD action 2: accelerated attack phase exit.")

        self.mutation_log.append({
            "level":     mutation_level,
            "timestamp": time.time(),
            "phase":     self._current_phase(),
        })
        # Simulate small ACK delay
        time.sleep(0.001)
        return True

    def trigger_high_alert_mutation(self) -> bool:
        """ML high-alert → force level-2 mutation."""
        logging.info("HIGH ALERT: forcing aggressive mutation.")
        return self.execute_mutation(2)
