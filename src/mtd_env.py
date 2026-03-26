"""
mtd_env.py — Offline MTD-Playground Gymnasium Environment
==========================================================
Simulates the enterprise network topology from the MTD-Playground paper
(5 hosts, 3 switches) without requiring ONOS or Mininet.

Formulation: Two-player Markov Game (Defender vs Attacker)
    Ref: Eghtesad, Vorobeychik & Laszka (2020)
         "Adversarial Deep RL Based Adaptive Moving Target Defense"
         GameSec 2020, Springer LNCS 12513, pp. 58-79.
         DOI: 10.1007/978-3-030-64793-3_4

Topology (from MTD-Playground paper, Figure 1):
    Client (10.0.0.10) --- S0 --- DMZ Web (10.0.0.30)
                            |
                           S1 --- Internal App (10.0.0.100)
                            |
                           S2 --- DB (10.0.0.40)
    Attacker (10.0.0.20) ---S2

    3 switches (S0, S1, S2), 5 hosts.
    3 possible forwarding paths between any pair of hosts.

Attack Model (MTD-Playground paper, Section 4):
    Multi-stage: Recon -> Initial Access -> Lateral Movement -> Exploitation -> Exfiltration
    Attacker progresses through stages; MTD mutations disrupt progress.

State Space (14 features):
    [0]  active_flow_count (normalised)
    [1]  avg_packets_per_switch (normalised)
    [2]  avg_bytes_per_switch (normalised)
    [3]  controller_latency (normalised)
    [4]  threat_level (simulated detector output)
    [5]  attacker_stage (0-4, normalised to 0-1)
    [6]  attacker_knowledge (fraction of topology mapped)
    [7]  path_entropy H_P (Shannon entropy of path distribution)
    [8]  attacker_entropy H_A (attacker's uncertainty about config)
    [9]  time_since_last_mutation (normalised)
    [10] mutation_count (normalised)
    [11] recon_accuracy (how accurately attacker maps hosts)
    [12] network_load (simulated QoS metric)
    [13] service_availability (fraction of successful client requests)

Action Space: Discrete(3)
    0 = No mutation (hold)
    1 = Moderate (partial path shuffle — randomise 1 of 3 paths)
    2 = Aggressive (full path randomisation — randomise all 3 paths)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

OBS_SIZE = 14
MAX_STEPS = 200

# Payoff weights (Eghtesad et al. 2020, Eq. 6)
W_SECURITY_GAIN = 1.0
W_MUTATION_COST = 0.3
W_OVERHEAD = 0.2


class AttackerSim:
    """
    Simulates a multi-stage attacker against the MTD-Playground topology.

    Attack stages (MTD-Playground paper Section 4.1):
        0: Reconnaissance / Scanning
        1: Initial Access (exploit DMZ web server)
        2: Lateral Movement (pivot to internal app)
        3: Privilege Escalation + DB exploitation
        4: Data Exfiltration (terminal — attacker wins)

    The attacker accumulates knowledge each step. MTD mutations reduce
    this knowledge, forcing the attacker to restart reconnaissance.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.reset()

    def reset(self):
        self.stage = 0              # current attack stage
        self.knowledge = 0.0        # fraction of topology mapped [0,1]
        self.recon_accuracy = 0.0   # how well attacker can identify real hosts
        self.steps_in_stage = 0

    def step(self, mutation_level: int, path_entropy: float):
        """
        Advance attacker by one timestep.
        Returns: (stage_changed, exfiltrated)
        """
        self.steps_in_stage += 1

        # Knowledge gain depends on path entropy (low entropy = easy to map)
        # Stronger attacker: gains knowledge faster, especially with low entropy
        base_gain = self.rng.uniform(0.04, 0.12)
        entropy_factor = (1.0 - path_entropy) * 1.5 + 0.3  # never zero
        knowledge_gain = base_gain * entropy_factor
        self.knowledge = min(1.0, self.knowledge + knowledge_gain)

        # Recon accuracy tracks with knowledge but has noise
        self.recon_accuracy = self.knowledge * self.rng.uniform(0.7, 1.0)

        # MTD mutation disrupts attacker knowledge
        if mutation_level == 1:
            disruption = self.rng.uniform(0.1, 0.25)
            self.knowledge = max(0.0, self.knowledge - disruption)
            if self.rng.random() < 0.15 and self.stage > 0:
                self.stage -= 1
                self.steps_in_stage = 0
        elif mutation_level == 2:
            disruption = self.rng.uniform(0.25, 0.5)
            self.knowledge = max(0.0, self.knowledge - disruption)
            if self.rng.random() < 0.35 and self.stage > 0:
                self.stage = max(0, self.stage - self.rng.integers(1, 3))
                self.steps_in_stage = 0

        # Attacker tries to advance stage based on knowledge
        # More aggressive: higher base probability, cumulative with time in stage
        advance_prob = self.knowledge * 0.2 + self.steps_in_stage * 0.01
        if self.stage == 0:
            advance_prob *= 1.3  # recon is easier
        elif self.stage >= 2:
            advance_prob *= 0.8  # lateral movement is harder

        stage_changed = False
        if self.rng.random() < advance_prob and self.stage < 4:
            self.stage += 1
            self.steps_in_stage = 0
            stage_changed = True

        exfiltrated = self.stage >= 4
        return stage_changed, exfiltrated


class NetworkSim:
    """
    Simulates the SDN network state for the MTD-Playground topology.
    Tracks path configurations, flow counts, and QoS metrics.
    """

    NUM_SWITCHES = 3
    NUM_PATHS = 3  # 3 possible paths between host pairs

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.reset()

    def reset(self):
        # Path distribution: probability of each path being active
        # Start with uniform (maximum entropy)
        self.path_probs = np.ones(self.NUM_PATHS) / self.NUM_PATHS
        self.flow_count = self.rng.integers(50, 150)
        self.latency_ms = self.rng.uniform(5, 20)
        self.mutation_count = 0
        self.steps_since_mutation = 0
        self.service_availability = 1.0

    def apply_mutation(self, level: int):
        """Apply path randomisation based on mutation level."""
        if level == 0:
            self.steps_since_mutation += 1
            # Paths drift toward non-uniform over time (attacker learns)
            drift = self.rng.uniform(0.0, 0.03, size=self.NUM_PATHS)
            self.path_probs += drift
            self.path_probs /= self.path_probs.sum()
            return

        self.mutation_count += 1
        self.steps_since_mutation = 0

        if level == 1:
            # Moderate: partially re-randomise paths
            noise = self.rng.uniform(0.1, 0.3, size=self.NUM_PATHS)
            self.path_probs = 0.5 * self.path_probs + 0.5 * (noise / noise.sum())
            self.latency_ms += self.rng.uniform(5, 15)  # overhead
            self.service_availability = max(0.85, self.service_availability - 0.02)
        elif level == 2:
            # Aggressive: fully re-randomise
            noise = self.rng.uniform(0.01, 1.0, size=self.NUM_PATHS)
            self.path_probs = noise / noise.sum()
            self.latency_ms += self.rng.uniform(15, 40)  # higher overhead
            self.service_availability = max(0.7, self.service_availability - 0.05)

        self.path_probs /= self.path_probs.sum()

        # Flow count changes with mutation
        self.flow_count = self.rng.integers(80, 200)

    def step_passive(self):
        """Passive network evolution each timestep."""
        # Latency naturally recovers
        self.latency_ms = max(5.0, self.latency_ms * 0.95 + self.rng.uniform(-1, 2))
        # Service availability recovers
        self.service_availability = min(1.0, self.service_availability + 0.005)
        # Flow count fluctuates
        self.flow_count = max(10, self.flow_count + self.rng.integers(-10, 10))

    def path_entropy(self) -> float:
        """Shannon entropy of path distribution, normalised to [0,1]."""
        p = self.path_probs
        p = p[p > 0]
        h = -np.sum(p * np.log2(p))
        h_max = np.log2(self.NUM_PATHS)
        return float(np.clip(h / h_max, 0.0, 1.0))


class MTDPlaygroundEnv(gym.Env):
    """
    Offline simulation of the MTD-Playground evaluation framework.

    This environment runs without ONOS/Mininet and produces the same
    evaluation metrics defined in MTD-Playground paper Section 7.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, seed=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.network = NetworkSim(self.rng)
        self.attacker = AttackerSim(self.rng)
        self.current_step = 0
        self.episode_rewards = []
        # Per-episode tracking for metrics
        self.attack_succeeded = False
        self.mutations_applied = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.network.rng = self.rng
            self.attacker.rng = self.rng
        self.network.reset()
        self.attacker.reset()
        self.current_step = 0
        self.episode_rewards = []
        self.attack_succeeded = False
        self.mutations_applied = 0
        return self._get_obs(), {}

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action)

        # 1. Apply mutation to network
        self.network.apply_mutation(action)
        if action > 0:
            self.mutations_applied += 1

        # 2. Advance attacker
        path_ent = self.network.path_entropy()
        stage_changed, exfiltrated = self.attacker.step(action, path_ent)

        # 3. Passive network evolution
        self.network.step_passive()

        # 4. Compute reward (Eghtesad et al. 2020 zero-sum payoff)
        reward, info = self._compute_reward(action, path_ent, stage_changed, exfiltrated)

        self.episode_rewards.append(reward)
        self.current_step += 1

        terminated = exfiltrated  # episode ends if attacker exfiltrates
        truncated = self.current_step >= MAX_STEPS
        self.attack_succeeded = exfiltrated

        obs = self._get_obs()
        info.update({
            "step": self.current_step,
            "attacker_stage": self.attacker.stage,
            "attacker_knowledge": self.attacker.knowledge,
            "path_entropy": path_ent,
            "attacker_entropy": 1.0 - self.attacker.knowledge,
            "recon_accuracy": self.attacker.recon_accuracy,
            "latency_ms": self.network.latency_ms,
            "service_availability": self.network.service_availability,
            "mutation_count": self.mutations_applied,
            "exfiltrated": exfiltrated,
            "cumulative_reward": sum(self.episode_rewards),
        })

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0] = min(self.network.flow_count / 500.0, 1.0)
        obs[1] = min(self.network.flow_count / (self.network.NUM_SWITCHES * 200.0), 1.0)
        obs[2] = obs[1] * 0.8  # bytes proxy
        obs[3] = min(self.network.latency_ms / 200.0, 1.0)
        obs[4] = min(self.attacker.stage / 4.0, 1.0)  # threat level proxy
        obs[5] = self.attacker.stage / 4.0
        obs[6] = self.attacker.knowledge
        obs[7] = self.network.path_entropy()
        obs[8] = 1.0 - self.attacker.knowledge  # attacker entropy
        obs[9] = min(self.network.steps_since_mutation / 20.0, 1.0)
        obs[10] = min(self.network.mutation_count / 50.0, 1.0)
        obs[11] = self.attacker.recon_accuracy
        obs[12] = min(self.network.flow_count * self.network.latency_ms / 50000.0, 1.0)
        obs[13] = self.network.service_availability
        return obs

    def _compute_reward(self, action, path_entropy, stage_changed, exfiltrated):
        """
        Zero-sum payoff: R_defender = -R_attacker
        Ref: Eghtesad et al. (2020) GameSec, Eq. 6
        """
        # Security gain: reward for keeping attacker entropy high
        security_gain = W_SECURITY_GAIN * (1.0 - self.attacker.knowledge)

        # Penalty proportional to attacker stage (progressive threat)
        stage_penalty = -0.15 * self.attacker.stage

        # Bonus for preventing stage advancement
        stage_bonus = 0.0
        if stage_changed:
            stage_bonus = -1.5  # penalty: attacker advanced

        # Big penalty if exfiltrated
        exfil_penalty = -10.0 if exfiltrated else 0.0

        # Cost of mutation (Eghtesad: cost-of-moving c_m)
        mutation_cost = W_MUTATION_COST * action

        # Overhead cost
        overhead = W_OVERHEAD * min(self.network.latency_ms / 200.0, 1.0)

        # Service availability bonus
        avail_bonus = 0.2 * self.network.service_availability

        # Entropy bonus (high path entropy = good defence)
        entropy_bonus = 0.3 * path_entropy

        reward = (
            security_gain
            + stage_penalty
            + stage_bonus
            + exfil_penalty
            + entropy_bonus
            + avail_bonus
            - mutation_cost
            - overhead
        )

        components = {
            "security_gain": round(security_gain, 4),
            "stage_penalty": round(stage_penalty, 4),
            "stage_bonus": round(stage_bonus, 4),
            "exfil_penalty": round(exfil_penalty, 4),
            "mutation_cost": round(mutation_cost, 4),
            "overhead": round(overhead, 4),
            "entropy_bonus": round(entropy_bonus, 4),
            "avail_bonus": round(avail_bonus, 4),
        }
        return float(reward), components

    def render(self):
        if self.render_mode == "human":
            print(
                f"[Step {self.current_step}] "
                f"Stage={self.attacker.stage} "
                f"Knowledge={self.attacker.knowledge:.2f} "
                f"PathEnt={self.network.path_entropy():.2f} "
                f"Reward={sum(self.episode_rewards):.2f}"
            )
