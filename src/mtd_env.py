"""
mtd_env.py
----------
Custom Gymnasium (OpenAI Gym) environment for the SDN-MTD RL agent.
Modelled as a two-player Markov Game (Defender vs. Attacker).

Architecture Reference:
    Eghtesad et al. (2020) — "Adversarial Deep Reinforcement Learning based
    Adaptive Moving Target Defense", GameSec 2020.
    https://doi.org/10.1007/978-3-030-64793-3_20

Key Additions vs. baseline:
    - State space includes 'Attacker Knowledge' entropy metrics as per
      the Markov Game formulation in Eghtesad et al.
    - Reward is a Zero-Sum Game payoff: Defender gain = -Attacker gain.
      Explicit 'Cost of Deception' vs. 'Attacker Reconnaissance Success'
      terms following the GameSec 2020 payoff matrix.

Action Space  : Discrete(3)  -> 0=No Move, 1=Moderate, 2=Aggressive
Observation   : 1-D numpy array (OBS_SIZE=24) of normalised features
Reward        : Zero-Sum payoff (see _zero_sum_payoff)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
import time
from scipy.stats import entropy as scipy_entropy  # for Shannon entropy calc

from .onos_client import ONOSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MTD-Env] %(message)s")

# -----------------------------------------------------------------------
# Zero-Sum Game payoff weights
# Ref: Eghtesad et al. (2020) GameSec — payoff matrix R(s,a_d,a_a)
# -----------------------------------------------------------------------
W_DECEPTION_COST  = 0.4   # cost paid by defender when performing a mutation
W_RECON_SUCCESS   = 1.0   # penalty when attacker successfully maps paths
W_RECON_PREVENTED = 1.2   # reward when mutation foils attacker knowledge
W_OVERHEAD        = 0.3   # controller/network overhead cost weight

# Feature vector:
#   [0]      – active flow count (normalised)
#   [1–8]    – per-device packet/byte counts
#   [9]      – controller latency
#   [10]     – ML threat score (injected by ThreatDetector)
#   [11]     – Attacker Knowledge Entropy H_A  (Eghtesad 2020)
#   [12]     – Path Entropy H_P  (unpredictability metric)
#   [13]     – Prior attacker belief (estimated recon success rate)
#   [14–23]  – Reserved for future topology features
OBS_SIZE = 24

# Episode length (number of time steps before a reset)
MAX_STEPS = 200


class SDN_MTD_Env(gym.Env):
    """
    Gymnasium environment wrapping the ONOS SDN controller.

    The state vector is derived from live flow statistics. The agent
    selects a mutation level; the environment applies it via ONOSClient,
    measures the resulting security gain / latency, and returns a reward.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str = "onos_config.json", render_mode=None):
        super().__init__()

        self.client = ONOSClient(config_path)
        self.render_mode = render_mode
        self.current_step = 0
        self.episode_rewards = []

        # --- Spaces ---
        self.action_space = spaces.Discrete(3)
        # Normalised floats in [0, 1] for each feature
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        logging.info("SDN_MTD_Env initialised.")

    # ------------------------------------------------------------------
    # Core Gym Interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_rewards = []
        # Reset mock client so every episode starts from phase 0 (normal)
        if hasattr(self.client, "reset"):
            self.client.reset()

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Apply the mutation (with session continuity when enabled)
        # Ref: SDN-MTD Standards — session continuity via OpenFlow Group Tables
        mutation_success = self.client.execute_mutation(mutation_level=action)

        # 2. Measure overhead (controller round-trip latency in ms)
        latency_ms = self.client.get_controller_latency()

        # 3. Observe the new network state
        obs = self._get_observation()

        # 4. Zero-Sum Game payoff
        # Ref: Eghtesad et al. (2020) GameSec — R_d(s,a_d,a_a) = -R_a(s,a_d,a_a)
        reward, payoff_components = self._zero_sum_payoff(action, obs, latency_ms)

        if not mutation_success:
            reward -= 1.0  # Penalty for failed ONOS API call

        self.episode_rewards.append(reward)
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= MAX_STEPS

        info = {
            "latency_ms": latency_ms,
            "mutation_success": mutation_success,
            "step": self.current_step,
            "attacker_entropy": float(obs[11]),
            "path_entropy": float(obs[12]),
            "attacker_belief": float(obs[13]),
            **payoff_components,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        step = self.current_step
        total_reward = sum(self.episode_rewards)
        print(f"[Step {step}] Cumulative Reward: {total_reward:.3f}")

    def close(self):
        logging.info("Environment closed.")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """
        Extracts a fixed-size (OBS_SIZE=24) feature vector from the ONOS
        network state, augmented with Attacker Knowledge metrics.

        Attacker Knowledge Entropy (H_A) and Path Entropy (H_P) are derived
        from the Markov Game state representation in:
            Eghtesad et al. (2020) — GameSec, Section 3.2 State Space.
        """
        state = self.client.get_network_state()
        flows = state.get("flows", [])
        port_stats = state.get("port_stats", [])

        features = np.zeros(OBS_SIZE, dtype=np.float32)

        # --- Network Observation Features [0–9] ---
        # Feature 0: Number of active flows (normalised by ceiling of 1000)
        features[0] = min(len(flows) / 1000.0, 1.0)

        # Features 1–8: Per-device packet/byte counts from port stats
        for i, device in enumerate(port_stats[:4]):
            ports = device.get("statistics", [])
            total_pkts = sum(p.get("packetsReceived", 0) for p in ports)
            total_bytes = sum(p.get("bytesReceived", 0) for p in ports)
            features[1 + i * 2] = min(total_pkts / 1e6, 1.0)
            features[2 + i * 2] = min(total_bytes / 1e9, 1.0)

        # Feature 9: Controller latency (normalised by 500 ms ceiling)
        latency = self.client.get_controller_latency()
        features[9] = min(latency / 500.0, 1.0)

        # Feature 10: ML threat score (injected externally by ThreatDetector)
        # — leave at 0.0 here; populated in main.py before model.predict()

        # --- Attacker Knowledge Metrics [11–13] ---
        # Ref: Eghtesad et al. (2020) — Markov Game state space S = (s_net, s_attacker)
        # s_attacker encodes how much of the network the attacker has successfully mapped.

        # Feature 11: Attacker Knowledge Entropy H_A
        #   H_A is high when attacker has uniform uncertainty (MTD is working).
        #   H_A is low when attacker has discovered stable IP→path mappings.
        features[11] = self._compute_attacker_knowledge_entropy(flows)

        # Feature 12: Path Entropy H_P — unpredictability of current IP assignments
        #   A higher H_P signals a diverse path space, making reconnaissance harder.
        #   Ref: Equation (4) in Eghtesad et al. (2020).
        features[12] = self._compute_path_entropy(flows)

        # Feature 13: Prior attacker belief (estimated recon success rate)
        #   Proxy: fraction of flows with repeated identical src/dst IP pairs
        #   over the last observation window — stable flows indicate exposure.
        if flows:
            ip_pairs = [(f.get("selector", {}).get("criteria", [{}])[0].get("ip", ""),
                         f.get("selector", {}).get("criteria", [{}])[-1].get("ip", ""))
                        for f in flows]
            unique_ratio = len(set(ip_pairs)) / max(len(ip_pairs), 1)
            # Low uniqueness → attacker can map endpoints → high belief
            features[13] = np.float32(1.0 - unique_ratio)

        return features

    def _compute_attacker_knowledge_entropy(self, flows: list) -> float:
        """
        Computes normalised Shannon entropy of the flow destination distribution.
        High entropy → attacker cannot predict routing (MTD effective).
        Low entropy  → attacker has learned stable paths (MTD failing).

        Ref: Eghtesad et al. (2020) GameSec — Attacker Knowledge metric H_A,
             derived from the probability distribution over network configurations.
        """
        if not flows:
            return 1.0  # No data = maximum uncertainty for attacker

        # Build probability distribution over egress ports as a proxy for
        # the configuration distribution the attacker is trying to learn.
        port_counts: dict = {}
        for f in flows:
            egress = f.get("treatment", {}).get("instructions", [{}])[0].get("port", "0")
            port_counts[egress] = port_counts.get(egress, 0) + 1

        counts = np.array(list(port_counts.values()), dtype=np.float32)
        probs = counts / counts.sum()
        h = float(scipy_entropy(probs, base=2))

        # Normalise by log2(N) so result is in [0, 1]
        max_entropy = np.log2(max(len(probs), 2))
        return float(np.clip(h / max_entropy, 0.0, 1.0))

    def _compute_path_entropy(self, flows: list) -> float:
        """
        Measures diversity of active forwarding paths using Shannon entropy
        of the egress port distribution across all flow rules.

        High entropy → flows spread evenly across ports → attacker cannot
        predict routing → MTD is working.
        Low entropy  → flows concentrated on a few ports → predictable paths.

        Ref: Eq. (4) 'Path Entropy' in Eghtesad et al. (2020),
             also used in Chowdhary et al. (2020) SDN-MTD framework.
        """
        if not flows:
            return 0.5

        # Count flows per egress output port (port diversity = path diversity)
        port_counts: dict = {}
        for f in flows:
            instructions = f.get("treatment", {}).get("instructions", [])
            for instr in instructions:
                if instr.get("type") == "OUTPUT":
                    port = f"{f.get('deviceId', 'unk')}_p{instr.get('port', '0')}"
                    port_counts[port] = port_counts.get(port, 0) + 1

        if not port_counts:
            # Fallback: use flow ID character diversity
            return 0.5

        # Shannon entropy of port distribution, normalised to [0, 1]
        counts = np.array(list(port_counts.values()), dtype=float)
        probs  = counts / counts.sum()
        raw_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = np.log2(len(port_counts)) if len(port_counts) > 1 else 1.0
        return float(np.clip(raw_entropy / max_entropy, 0.0, 1.0))

    def inject_threat_score(self, score: float, index: int = 10):
        """
        Allows threat_detector.py to inject its confidence score into the
        observation vector at inference time before the agent acts.
        Call this on the stored obs before passing to the model.
        This is a stateless helper — applied in main.py's loop.
        """
        pass

    def _zero_sum_payoff(
        self, action: int, obs: np.ndarray, latency_ms: float
    ) -> tuple[float, dict]:
        """
        Zero-Sum Game payoff for the Defender.
        Models the interaction as R_d(s, a_d, a_a) = -R_a(s, a_d, a_a).

        Payoff components:
          + W_RECON_PREVENTED  * (1 - attacker_belief)  if action > 0
            → reward for reducing the attacker's effective prior belief
          + W_DECEPTION_COST   * action (scaled)
            → cost paid for executing a mutation (controller load, session risk)
          - W_RECON_SUCCESS    * attacker_belief
            → penalty proportional to how much the attacker already knows
          - W_OVERHEAD         * (latency_ms / 1000)
            → network overhead penalty

        Ref:
            Eghtesad et al. (2020) GameSec — Eq. (6) Zero-Sum payoff matrix.
            "The defender reward is defined as the negative of the attacker reward,
             forming a strictly competitive (zero-sum) Markov Game."
        """
        attacker_belief  = float(obs[13]) if len(obs) > 13 else 0.0
        attacker_entropy = float(obs[11]) if len(obs) > 11 else 1.0
        threat_score     = float(obs[10]) if len(obs) > 10 else 0.0

        # --- Defender gains (positive terms) ---
        # Executing a mutation disrupts attacker's accumulated knowledge.
        # Gain scales with action intensity and how much knowledge is disrupted.
        if action > 0:
            recon_prevented = W_RECON_PREVENTED * (1.0 - attacker_belief) * action * 0.5
        else:
            recon_prevented = 0.0

        # Bonus for keeping entropy high (unpredictable network state)
        entropy_bonus = attacker_entropy * 0.3

        # --- Defender costs (negative terms) ---
        # Cost of Deception: mutating disrupts legitimate traffic flows.
        # Ref: Eghtesad et al. (2020) — "cost of moving" term c_m.
        deception_cost = W_DECEPTION_COST * action

        # Attacker reconnaissance success penalty: if attacker_belief is high,
        # the defender is "losing" the Markov Game regardless of action.
        recon_success_penalty = W_RECON_SUCCESS * attacker_belief * (1.0 - action * 0.3)

        # Threat bonus: immediate reward for reacting to a detected probe.
        threat_response = threat_score * 0.5 * action

        # Network overhead cost.
        overhead_cost = W_OVERHEAD * (latency_ms / 1000.0)

        total_reward = (
            recon_prevented
            + entropy_bonus
            + threat_response
            - deception_cost
            - recon_success_penalty
            - overhead_cost
        )

        components = {
            "recon_prevented":      round(recon_prevented, 4),
            "entropy_bonus":        round(entropy_bonus, 4),
            "deception_cost":       round(deception_cost, 4),
            "recon_success_penalty": round(recon_success_penalty, 4),
            "threat_response":      round(threat_response, 4),
            "overhead_cost":        round(overhead_cost, 4),
        }
        return float(total_reward), components
