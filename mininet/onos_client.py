"""
onos_client.py — ONOS REST API Client for MTD-Playground
=========================================================
Connects to the ONOS SDN controller to:
  1. READ network state (flows, port stats, latency)
  2. WRITE path mutations (install/modify flow rules)

ONOS REST API docs: http://localhost:8181/onos/v1/docs

Default credentials: onos / rocks
"""

import requests
import json
import time
import random
import logging
from requests.auth import HTTPBasicAuth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ONOS] %(message)s")

# MTD-Playground topology device IDs (set by topology.py)
SWITCH_IDS = [
    "of:0000000000000001",  # S0
    "of:0000000000000002",  # S1
    "of:0000000000000003",  # S2
]

# Host IPs and MACs
HOSTS = {
    "client":   {"ip": "10.0.0.10", "mac": "00:00:00:00:00:01"},
    "attacker": {"ip": "10.0.0.20", "mac": "00:00:00:00:00:02"},
    "web":      {"ip": "10.0.0.30", "mac": "00:00:00:00:00:03"},
    "app":      {"ip": "10.0.0.100", "mac": "00:00:00:00:00:04"},
    "db":       {"ip": "10.0.0.40", "mac": "00:00:00:00:00:05"},
}


class ONOSClient:
    """REST API client for ONOS SDN controller."""

    def __init__(self, host="127.0.0.1", port=8181, username="onos", password="rocks"):
        self.base_url = f"http://{host}:{port}/onos/v1"
        self.auth = HTTPBasicAuth(username, password)
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        self.mutation_count = 0
        self._verify_connection()

    def _verify_connection(self):
        """Check ONOS is reachable."""
        try:
            r = self._get("/cluster")
            nodes = r.get("nodes", [])
            logging.info(f"Connected to ONOS. Nodes: {len(nodes)}")
        except Exception as e:
            logging.error(f"Cannot connect to ONOS at {self.base_url}: {e}")
            raise

    def _get(self, path):
        """GET request to ONOS API."""
        get_headers = {"Accept": "application/json"}
        r = requests.get(f"{self.base_url}{path}", auth=self.auth, headers=get_headers, timeout=5)
        r.raise_for_status()
        return r.json()

    def _post(self, path, data):
        """POST request to ONOS API."""
        r = requests.post(f"{self.base_url}{path}", auth=self.auth, headers=self.headers,
                          data=json.dumps(data), timeout=5)
        r.raise_for_status()
        return r

    def _delete(self, path):
        """DELETE request to ONOS API."""
        r = requests.delete(f"{self.base_url}{path}", auth=self.auth, headers=self.headers, timeout=5)
        return r

    # ==================================================================
    # READ — Network State Observation
    # ==================================================================

    def get_network_state(self) -> dict:
        """
        Fetch full network state from ONOS.
        Returns dict with flows, port_stats, devices, hosts.
        """
        return {
            "flows": self.get_flows(),
            "port_stats": self.get_port_stats(),
            "devices": self.get_devices(),
            "hosts": self.get_hosts(),
            "timestamp": time.time(),
        }

    def get_flows(self) -> list:
        """Get all installed flow rules."""
        try:
            data = self._get("/flows")
            return data.get("flows", [])
        except Exception as e:
            logging.error(f"Failed to get flows: {e}")
            return []

    def get_flows_for_device(self, device_id: str) -> list:
        """Get flow rules for a specific switch."""
        try:
            data = self._get(f"/flows/{device_id}")
            return data.get("flows", [])
        except Exception as e:
            logging.error(f"Failed to get flows for {device_id}: {e}")
            return []

    def get_port_stats(self) -> list:
        """Get per-port packet/byte counters for all devices."""
        try:
            data = self._get("/statistics/ports")
            return data.get("statistics", [])
        except Exception as e:
            logging.error(f"Failed to get port stats: {e}")
            return []

    def get_devices(self) -> list:
        """Get list of connected switches."""
        try:
            data = self._get("/devices")
            return data.get("devices", [])
        except Exception as e:
            logging.error(f"Failed to get devices: {e}")
            return []

    def get_hosts(self) -> list:
        """Get discovered hosts."""
        try:
            data = self._get("/hosts")
            return data.get("hosts", [])
        except Exception as e:
            logging.error(f"Failed to get hosts: {e}")
            return []

    def get_controller_latency(self) -> float:
        """Measure round-trip latency to ONOS in ms."""
        start = time.time()
        try:
            self._get("/cluster")
        except Exception:
            return 9999.0
        return round((time.time() - start) * 1000, 2)

    def get_flow_count(self) -> int:
        """Get total number of flow rules across all switches."""
        return len(self.get_flows())

    def get_link_count(self) -> int:
        """Get number of active links."""
        try:
            data = self._get("/links")
            return len(data.get("links", []))
        except Exception:
            return 0

    # ==================================================================
    # WRITE — Path Mutation Actions
    # ==================================================================

    def execute_mutation(self, level: int) -> bool:
        """
        Execute a path mutation on the SDN network.

        Level 0: No mutation (hold)
        Level 1: Moderate — change flow rules on 1 switch (partial path shuffle)
        Level 2: Aggressive — change flow rules on all switches (full randomization)

        Implements path randomization by modifying forwarding rules so that
        traffic between hosts takes different paths through the switch fabric.
        """
        if level == 0:
            return True

        try:
            if level == 1:
                # Moderate: modify flows on one random switch
                target_switch = random.choice(SWITCH_IDS)
                self._randomize_paths_on_switch(target_switch)
                self.mutation_count += 1
                logging.info(f"Moderate mutation on {target_switch} (total: {self.mutation_count})")

            elif level == 2:
                # Aggressive: modify flows on all switches
                for switch_id in SWITCH_IDS:
                    self._randomize_paths_on_switch(switch_id)
                self.mutation_count += 1
                logging.info(f"Aggressive mutation on ALL switches (total: {self.mutation_count})")

            return True

        except Exception as e:
            logging.error(f"Mutation failed: {e}")
            return False

    def _randomize_paths_on_switch(self, device_id: str):
        """
        Randomize forwarding paths on a specific switch.

        Strategy: Delete existing flow rules installed by our app,
        then install new rules with randomized output ports.
        ONOS reactive forwarding will also create new paths as
        traffic arrives on the cleared switch.
        """
        # Remove our custom MTD flow rules (keep ONOS system rules)
        flows = self.get_flows_for_device(device_id)
        for flow in flows:
            app_id = flow.get("appId", "")
            # Only remove flows installed by our MTD app or reactive fwd
            if app_id in ("org.onosproject.fwd", "org.onosproject.mtd"):
                flow_id = flow.get("id", "")
                if flow_id:
                    self._delete(f"/flows/{device_id}/{flow_id}")

        # Install new flow rules with randomized priorities
        # This forces traffic to take different paths
        self._install_randomized_rules(device_id)

    def _install_randomized_rules(self, device_id: str):
        """
        Install flow rules with randomized output ports to shuffle paths.

        For each host pair, we install a flow rule that forwards traffic
        to a randomly selected output port, effectively changing the path.
        """
        # Get available ports on this switch
        try:
            ports_data = self._get(f"/devices/{device_id}/ports")
            ports = [p["port"] for p in ports_data.get("ports", [])
                     if p.get("isEnabled", False) and p["port"] != "local"]
        except Exception:
            return

        if len(ports) < 2:
            return

        # For each pair of ports, install a forwarding rule with random priority
        # This changes which path gets preferred for traffic between hosts
        priority = random.randint(30000, 40000)

        for src_host, src_info in HOSTS.items():
            for dst_host, dst_info in HOSTS.items():
                if src_host == dst_host:
                    continue

                # Pick a random output port (this is the path randomization)
                out_port = random.choice(ports)

                flow_rule = {
                    "priority": priority,
                    "timeout": 30,  # rules expire after 30s (forces re-randomization)
                    "isPermanent": False,
                    "deviceId": device_id,
                    "treatment": {
                        "instructions": [
                            {"type": "OUTPUT", "port": out_port}
                        ]
                    },
                    "selector": {
                        "criteria": [
                            {"type": "ETH_TYPE", "ethType": "0x0800"},
                            {"type": "IPV4_SRC", "ip": f"{src_info['ip']}/32"},
                            {"type": "IPV4_DST", "ip": f"{dst_info['ip']}/32"},
                        ]
                    }
                }

                try:
                    self._post(f"/flows/{device_id}", flow_rule)
                except Exception:
                    pass  # Some rules may conflict, that's OK

    # ==================================================================
    # UTILITY
    # ==================================================================

    def reset(self):
        """Reset mutation counter."""
        self.mutation_count = 0

    def get_topology_summary(self) -> dict:
        """Get a quick summary of the network state."""
        devices = self.get_devices()
        hosts = self.get_hosts()
        flows = self.get_flows()
        latency = self.get_controller_latency()

        return {
            "switches": len(devices),
            "hosts": len(hosts),
            "flows": len(flows),
            "links": self.get_link_count(),
            "latency_ms": latency,
            "mutations": self.mutation_count,
        }
