"""
onos_client.py
--------------
Modular wrapper for the ONOS Northbound REST API.
This module is the pure observation and action bridge between the
AI/ML Decision Engine and the SDN Controller — it has no intelligence
of its own. All decision logic lives in mtd_env.py and threat_detector.py.

Role in the AI pipeline:
  - Provides the RL agent (Eghtesad 2020) with its state observations:
    flow counts, port stats, controller latency.
  - Executes the discrete actions chosen by the agent (mutate level 0/1/2).
  - Surfaces live flow data to the ThreatDetector (Li 2021 HybridMTD)
    for ensemble-switched classification.
"""

import requests
import json
import time
import logging
from requests.auth import HTTPBasicAuth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ONOSClient] %(message)s")


class ONOSClient:
    """
    REST API wrapper for the ONOS SDN Controller (Northbound interface).
    Handles authentication, state observation, and mutation execution.
    """

    def __init__(self, config_path: str = "onos_config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)

        self.base_url = f"http://{config['controller_ip']}:{config['port']}/onos/v1"
        self.auth     = HTTPBasicAuth(config["username"], config["password"])
        self.headers  = {"Content-Type": "application/json", "Accept": "application/json"}
        self.mutation_log: list = []

        logging.info(f"ONOSClient initialised. Target: {self.base_url}")

    # ------------------------------------------------------------------
    # Observation — feeds the RL state vector (Eghtesad 2020, Sec 3.2)
    # ------------------------------------------------------------------

    def get_network_state(self) -> dict:
        """
        Fetches current network state from ONOS.
        Returns a dict with flows and port_stats consumed by SDN_MTD_Env
        to build the Markov Game state vector S = (s_net, s_attacker).

        Ref: Eghtesad et al. (2020) — state space observation interface.
        """
        return {
            "flows":      self._get_flows(),
            "port_stats": self._get_port_stats(),
            "timestamp":  time.time(),
        }

    def _get_flows(self) -> list:
        """Fetches all installed flow rules from ONOS."""
        try:
            resp = requests.get(
                f"{self.base_url}/flows",
                auth=self.auth, headers=self.headers, timeout=5,
            )
            resp.raise_for_status()
            return resp.json().get("flows", [])
        except requests.RequestException as e:
            logging.error(f"Failed to fetch flows: {e}")
            return []

    def _get_port_stats(self) -> list:
        """Fetches per-port packet/byte counters from all ONOS devices."""
        try:
            resp = requests.get(
                f"{self.base_url}/statistics/ports",
                auth=self.auth, headers=self.headers, timeout=5,
            )
            resp.raise_for_status()
            return resp.json().get("statistics", [])
        except requests.RequestException as e:
            logging.error(f"Failed to fetch port stats: {e}")
            return []

    def get_controller_latency(self) -> float:
        """
        Measures round-trip latency to the ONOS controller in milliseconds.

        Used as the network overhead component in the Zero-Sum payoff:
            R_d = ... - W_overhead * (latency_ms / 1000)
        Ref: Eghtesad et al. (2020) — cost-of-moving term c_m in Eq. (6).
        """
        start = time.time()
        try:
            requests.get(
                f"{self.base_url}/cluster",
                auth=self.auth, headers=self.headers, timeout=5,
            )
        except requests.RequestException:
            return 9999.0  # High penalty when controller is unreachable
        return round((time.time() - start) * 1000, 2)

    # ------------------------------------------------------------------
    # Actions — executes discrete moves chosen by the RL agent
    # ------------------------------------------------------------------

    def execute_mutation(self, mutation_level: int) -> bool:
        """
        Pushes an IP/path mutation Intent to ONOS.
        Called by SDN_MTD_Env.step() on each RL timestep.

        Args:
            mutation_level:
                0 = No Move       (agent chose to hold)
                1 = Moderate      (partial path shuffle)
                2 = Aggressive    (full IP + path randomisation)

        Returns:
            bool: True if ONOS acknowledged the mutation.

        Ref: Eghtesad et al. (2020) — discrete action set A = {0, 1, 2}.
        """
        if mutation_level == 0:
            logging.info("Action 0: No mutation.")
            return True

        payload = self._build_intent_payload(mutation_level)
        try:
            resp = requests.post(
                f"{self.base_url}/intents",
                auth=self.auth, headers=self.headers,
                data=json.dumps(payload), timeout=5,
            )
            resp.raise_for_status()
            logging.info(f"Mutation level {mutation_level} applied. HTTP {resp.status_code}")
            self.mutation_log.append({
                "level":     mutation_level,
                "timestamp": time.time(),
                "status":    resp.status_code,
            })
            return True
        except requests.RequestException as e:
            logging.error(f"Mutation failed: {e}")
            return False

    def _build_intent_payload(self, level: int) -> dict:
        """
        Constructs a PointToPoint Intent payload for ONOS.
        In production, ingress/egress device IDs are resolved from the
        live topology API. Priority scales with mutation intensity.
        """
        return {
            "type":         "PointToPointIntent",
            "appId":        "org.onosproject.mtd",
            "priority":     40000 if level == 2 else 30000,
            "ingressPoint": {"device": "of:0000000000000001", "port": "1"},
            "egressPoint":  {"device": "of:0000000000000002", "port": "2"},
        }

    def trigger_high_alert_mutation(self) -> bool:
        """
        Called by ThreatDetector when ensemble classifier confidence >= 90%
        (HybridMTD Ref: Li et al. 2021). Forces immediate aggressive mutation
        (level 2), bypassing the RL agent's scheduled action.
        """
        logging.warning("HIGH ALERT from ML Detector — forcing aggressive mutation.")
        return self.execute_mutation(mutation_level=2)
