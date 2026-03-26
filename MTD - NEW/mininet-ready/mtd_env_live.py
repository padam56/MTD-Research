"""
mtd_env_live.py — Live ONOS Gymnasium Environment
===================================================
Same interface as reference-only/mtd_env.py but reads from
a real ONOS controller instead of simulation.

The trained DQN model (models/best_model.zip) works directly
with this environment — same observation space, same actions.

Usage:
    from mtd_env_live import MTDLiveEnv
    env = MTDLiveEnv()
    obs, info = env.reset()
    action = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import logging

from onos_client import ONOSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LiveEnv] %(message)s")

OBS_SIZE = 14
MAX_STEPS = 200


class MTDLiveEnv(gym.Env):
    """
    Gymnasium environment connected to a real ONOS SDN controller.

    Observation: same 14-dim vector as offline env
    Actions: same Discrete(3) — {hold, moderate, aggressive}
    Reward: computed from real network metrics
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, onos_host="127.0.0.1", onos_port=8181,
                 username="onos", password="rocks", render_mode=None):
        super().__init__()

        self.client = ONOSClient(onos_host, onos_port, username, password)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        self.current_step = 0
        self.episode_rewards = []
        self.prev_flow_count = 0
        self.prev_latency = 0
        self.baseline_latency = None

        # Track attacker knowledge estimate (inferred from flow stability)
        self.knowledge_estimate = 0.0
        self.prev_flow_hash = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_rewards = []
        self.client.reset()
        self.knowledge_estimate = 0.0
        self.prev_flow_hash = None

        # Measure baseline latency
        self.baseline_latency = self.client.get_controller_latency()
        logging.info(f"Environment reset. Baseline latency: {self.baseline_latency:.1f}ms")

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action)

        # 1. Execute mutation on real network
        mutation_ok = self.client.execute_mutation(action)

        # Small delay to let ONOS process the flow changes
        if action > 0:
            time.sleep(0.5)

        # 2. Read new network state
        obs = self._get_observation()

        # 3. Compute reward
        latency = self.client.get_controller_latency()
        reward = self._compute_reward(action, obs, latency, mutation_ok)

        self.episode_rewards.append(reward)
        self.current_step += 1

        terminated = False  # No simulated attacker in live mode
        truncated = self.current_step >= MAX_STEPS

        info = {
            "step": self.current_step,
            "latency_ms": latency,
            "mutation_ok": mutation_ok,
            "mutation_count": self.client.mutation_count,
            "flow_count": self.client.get_flow_count(),
            "path_entropy": float(obs[7]),
            "attacker_entropy": float(obs[8]),
            "knowledge_estimate": float(obs[6]),
            "service_availability": float(obs[13]),
            "cumulative_reward": sum(self.episode_rewards),
        }

        if self.render_mode == "human":
            self._render_step(info)

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Build 14-dim observation from real ONOS data.
        Matches the same indices as the offline simulation.
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        state = self.client.get_network_state()
        flows = state["flows"]
        port_stats = state["port_stats"]
        latency = self.client.get_controller_latency()

        flow_count = len(flows)

        # [0] Active flow count (normalised)
        obs[0] = min(flow_count / 500.0, 1.0)

        # [1] Avg packets per switch
        total_pkts = 0
        total_bytes = 0
        for device_stats in port_stats:
            for port in device_stats.get("statistics", []):
                total_pkts += port.get("packetsReceived", 0) + port.get("packetsSent", 0)
                total_bytes += port.get("bytesReceived", 0) + port.get("bytesSent", 0)
        obs[1] = min(total_pkts / 1e6, 1.0)

        # [2] Avg bytes per switch
        obs[2] = min(total_bytes / 1e9, 1.0)

        # [3] Controller latency (normalised)
        obs[3] = min(latency / 200.0, 1.0)

        # [4] Threat level — estimated from flow patterns
        # High flow count spike or unusual patterns indicate scanning/attack
        if self.prev_flow_count > 0:
            flow_change_rate = abs(flow_count - self.prev_flow_count) / max(self.prev_flow_count, 1)
            obs[4] = min(flow_change_rate, 1.0)
        self.prev_flow_count = flow_count

        # [5] Attacker stage estimate (0-1)
        # In live mode, we don't know the exact stage — estimate from flow patterns
        # For now, use a heuristic based on flow diversity
        obs[5] = obs[4]  # proxy: threat level

        # [6] Attacker knowledge estimate
        # Inferred from flow stability: if flow rules haven't changed,
        # the attacker can build a stable map of the network
        flow_hash = hash(str(sorted([f.get("id", "") for f in flows[:50]])))
        if self.prev_flow_hash is not None and flow_hash == self.prev_flow_hash:
            self.knowledge_estimate = min(1.0, self.knowledge_estimate + 0.05)
        else:
            self.knowledge_estimate = max(0.0, self.knowledge_estimate - 0.1)
        self.prev_flow_hash = flow_hash
        obs[6] = self.knowledge_estimate

        # [7] Path entropy — computed from flow rule output port distribution
        obs[7] = self._compute_path_entropy(flows)

        # [8] Attacker entropy
        obs[8] = 1.0 - self.knowledge_estimate

        # [9] Time since last mutation
        obs[9] = min(self.current_step / 20.0, 1.0) if self.client.mutation_count == 0 else 0.0

        # [10] Mutation count
        obs[10] = min(self.client.mutation_count / 50.0, 1.0)

        # [11] Recon accuracy estimate
        obs[11] = self.knowledge_estimate * 0.8

        # [12] Network load
        obs[12] = min(flow_count * latency / 50000.0, 1.0)

        # [13] Service availability — based on latency ratio to baseline
        if self.baseline_latency and self.baseline_latency > 0:
            ratio = latency / self.baseline_latency
            obs[13] = max(0.0, min(1.0, 2.0 - ratio))  # 1.0 when normal, drops with high latency
        else:
            obs[13] = 1.0

        return obs

    def _compute_path_entropy(self, flows: list) -> float:
        """Compute path entropy from real flow rules."""
        if not flows:
            return 0.5

        port_counts = {}
        for f in flows:
            treatment = f.get("treatment", {})
            for instr in treatment.get("instructions", []):
                if instr.get("type") == "OUTPUT":
                    port_key = f"{f.get('deviceId', 'unk')}_{instr.get('port', '0')}"
                    port_counts[port_key] = port_counts.get(port_key, 0) + 1

        if not port_counts:
            return 0.5

        counts = np.array(list(port_counts.values()), dtype=float)
        probs = counts / counts.sum()
        h = -np.sum(probs * np.log2(probs + 1e-10))
        h_max = np.log2(max(len(port_counts), 2))
        return float(np.clip(h / h_max, 0.0, 1.0))

    def _compute_reward(self, action, obs, latency, mutation_ok):
        """
        Reward function — same structure as offline env.
        In live mode, we can't know the attacker's true state,
        so we use proxy metrics from network observations.
        """
        knowledge = float(obs[6])
        path_entropy = float(obs[7])
        availability = float(obs[13])

        security_gain = 1.0 * (1.0 - knowledge)
        entropy_bonus = 0.3 * path_entropy
        avail_bonus = 0.2 * availability
        mutation_cost = 0.3 * action
        overhead = 0.2 * min(latency / 200.0, 1.0)

        if not mutation_ok:
            mutation_cost += 1.0

        reward = security_gain + entropy_bonus + avail_bonus - mutation_cost - overhead
        return float(reward)

    def _render_step(self, info):
        print(f"[Step {info['step']}] "
              f"Flows={info['flow_count']} "
              f"Latency={info['latency_ms']:.1f}ms "
              f"H_P={info['path_entropy']:.3f} "
              f"Knowledge≈{info['knowledge_estimate']:.2f} "
              f"Mutations={info['mutation_count']} "
              f"Reward={info['cumulative_reward']:.2f}")
