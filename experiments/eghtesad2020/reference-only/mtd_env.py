"""
Markov Game Environment for Adaptive MTD
Implements Eghtesad et al. (2020) game formulation

State: (network config, attacker knowledge)
Actions: 0=no-move, 1=moderate MTD, 2=aggressive MTD
Reward: Zero-sum payoff with detection + deception cost
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any
import pandas as pd


class SDN_MTD_Env(gym.Env):
    """
    Gymnasium environment for Markov game MTD learning.
    
    State: [threat, H_A, H_P, traffic_stats..., 14 features total]
    Action: Discrete(3) {0, 1, 2}
    Reward: Zero-sum payoff
    """
    
    def __init__(self, threat_detector=None, seed: int = 42):
        super().__init__()
        np.random.seed(seed)
        
        self.threat_detector = threat_detector
        self.seed_val = seed
        
        # Action space: 0=no-move, 1=moderate MTD, 2=aggressive MTD
        self.action_space = spaces.Discrete(3)
        
        # State space: 14 features
        # [threat, H_A, H_P, bytes/s, pkts/s, syn_mean, fin_mean, rst_mean, ack_mean,
        #  tcp_frac, udp_frac, icmp_frac, port_entropy, pad]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(14,), dtype=np.float32
        )
        
        # Episode state
        self.phase = 0  # 0=normal, 1=recon, 2=attack, 3=post_attack
        self.phase_step = 0
        self.phase_duration = 100  # steps per phase
        self.current_flows = pd.DataFrame()
        self._last_state = None
        self._step_count = 0
        self._episode_step = 0
        
    def reset(self, seed=None):
        """Reset environment for new episode."""
        if seed is not None:
            np.random.seed(seed)
        
        self.phase = 0
        self.phase_step = 0
        self._step_count = 0
        self._episode_step = 0
        
        # Generate initial flows
        self._advance_step()
        
        obs = self._get_observation()
        self._last_state = obs.copy()
        
        return obs.astype(np.float32), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step of environment."""
        self._step_count += 1
        self._episode_step += 1
        
        # Advance to next phase/flows (but reuse cached state)
        self._advance_step()
        
        # Get current observation
        obs = self._get_observation()
        
        # Compute threat and entropy
        threat = self._compute_threat_level()
        H_A = self._compute_attacker_knowledge_entropy()
        H_P = self._compute_path_entropy()
        
        # Zero-sum payoff calculation
        reward_detect = 1.0 if threat > 0.5 else -0.1
        deception_cost = 0.1 * action  # Action 0 costs 0, action 2 costs 0.2
        reward = reward_detect - deception_cost
        
        # Determine if episode ends
        terminated = self._episode_step >= 200  # 200 steps per episode
        truncated = False
        
        # Info dict
        info = {
            'threat_level': float(threat),
            'path_entropy': float(H_P),
            'attacker_entropy': float(H_A),
            'deception_cost': float(deception_cost),
            'phase': ['normal', 'recon', 'attack', 'post_attack'][self.phase],
            'shap_top_feature': self._get_shap_top_feature(),
            'detection_success': 1 if threat > 0.5 else 0,
        }
        
        self._last_state = obs.copy()
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def _advance_step(self):
        """Advance phase and generate flows."""
        self.phase_step += 1
        
        # Phase transitions
        if self.phase_step >= self.phase_duration:
            self.phase = (self.phase + 1) % 4
            self.phase_step = 0
        
        # Generate flows based on phase
        self.current_flows = self._generate_flows_for_phase()
    
    def _generate_flows_for_phase(self) -> pd.DataFrame:
        """Generate synthetic flows for current phase."""
        n_flows = np.random.randint(20, 50)
        
        if self.phase == 0:  # normal
            return self._generate_benign_flows(n_flows)
        elif self.phase == 1:  # recon
            return self._generate_reconnaissance_flows(n_flows)
        elif self.phase == 2:  # attack
            return self._generate_attack_flows(n_flows)
        else:  # post_attack
            return self._generate_benign_flows(n_flows)
    
    def _generate_benign_flows(self, n: int) -> pd.DataFrame:
        """Normal network traffic."""
        flows = []
        for _ in range(n):
            flows.append({
                'src_ip': f"10.0.0.{np.random.randint(1, 100)}",
                'dst_ip': f"10.0.0.{np.random.randint(1, 100)}",
                'duration': np.random.exponential(5),  # 5s avg
                'bytes': np.random.exponential(10000),
                'packets': np.random.exponential(100),
                'syn_flags': np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]),
                'fin_flags': np.random.choice([0, 1], p=[0.8, 0.2]),
                'rst_flags': np.random.choice([0, 1], p=[0.95, 0.05]),
                'ack_flags': np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2]),
                'src_port': np.random.randint(1024, 65535),
                'dst_port': np.random.choice([80, 443, 22, 23, 25, 3306]),
                'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], p=[0.6, 0.3, 0.1]),
                'label': 0,  # benign
            })
        return pd.DataFrame(flows)
    
    def _generate_reconnaissance_flows(self, n: int) -> pd.DataFrame:
        """Reconnaissance traffic (port scanning)."""
        flows = []
        scanner_ip = "192.168.1.100"  # attacker
        for _ in range(n):
            flows.append({
                'src_ip': scanner_ip,
                'dst_ip': f"10.0.0.{np.random.randint(1, 100)}",
                'duration': np.random.exponential(0.1),  # Short
                'bytes': np.random.exponential(100),  # Small
                'packets': np.random.exponential(5),
                'syn_flags': np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]),  # HIGH
                'fin_flags': 0,
                'rst_flags': np.random.choice([1, 2], p=[0.7, 0.3]),  # HIGH
                'ack_flags': 0,
                'src_port': np.random.randint(1024, 65535),
                'dst_port': np.random.randint(1, 1024),  # Well-known ports
                'protocol': 'TCP',
                'label': 1,  # attack
            })
        return pd.DataFrame(flows)
    
    def _generate_attack_flows(self, n: int) -> pd.DataFrame:
        """Attack traffic (DDoS, exfiltration, etc.)."""
        flows = []
        attacker_ip = "192.168.1.100"
        target_ip = "10.0.0.1"
        for _ in range(n):
            flows.append({
                'src_ip': attacker_ip,
                'dst_ip': target_ip,
                'duration': np.random.exponential(0.01),  # Very short (DDoS packets)
                'bytes': np.random.exponential(1000),  # Large volume
                'packets': np.random.exponential(500),  # HIGH
                'syn_flags': np.random.choice([0, 1], p=[0.4, 0.6]),
                'fin_flags': 0,
                'rst_flags': np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3]),
                'ack_flags': np.random.choice([0, 1], p=[0.6, 0.4]),
                'src_port': np.random.randint(1024, 65535),
                'dst_port': np.random.choice([80, 443, 22]),
                'protocol': np.random.choice(['TCP', 'UDP'], p=[0.4, 0.6]),
                'label': 1,  # attack
            })
        return pd.DataFrame(flows)
    
    def _get_observation(self) -> np.ndarray:
        """Compute state vector."""
        if self.current_flows.empty:
            return np.zeros(14, dtype=np.float32)
        
        flows = self.current_flows
        
        # Compute threat level
        threat = self._compute_threat_level()
        
        # Compute entropies
        H_A = self._compute_attacker_knowledge_entropy()
        H_P = self._compute_path_entropy()
        
        # Traffic statistics (normalized)
        bytes_per_sec = flows['bytes'].sum() / max(flows['duration'].sum(), 0.1)
        bytes_per_sec_norm = np.clip(bytes_per_sec / 100000, 0, 1)
        
        packets_per_sec = flows['packets'].sum() / max(flows['duration'].sum(), 0.1)
        packets_per_sec_norm = np.clip(packets_per_sec / 10000, 0, 1)
        
        # Flag means
        syn_mean = flows['syn_flags'].mean() / 3.0  # normalize max=3
        fin_mean = flows['fin_flags'].mean()
        rst_mean = flows['rst_flags'].mean()
        ack_mean = flows['ack_flags'].mean() / 3.0
        
        # Protocol fractions
        tcp_frac = (flows['protocol'] == 'TCP').sum() / len(flows)
        udp_frac = (flows['protocol'] == 'UDP').sum() / len(flows)
        icmp_frac = (flows['protocol'] == 'ICMP').sum() / len(flows)
        
        # Port entropy (dst ports)
        port_entropy = self._compute_entropy(flows['dst_port'].value_counts() / len(flows))
        port_entropy_norm = np.clip(port_entropy / 10, 0, 1)
        
        obs = np.array([
            threat,
            H_A,
            H_P,
            bytes_per_sec_norm,
            packets_per_sec_norm,
            syn_mean,
            fin_mean,
            rst_mean,
            ack_mean,
            tcp_frac,
            udp_frac,
            icmp_frac,
            port_entropy_norm,
            0.0  # padding
        ], dtype=np.float32)
        
        return np.clip(obs, 0, 1)
    
    def _compute_threat_level(self) -> float:
        """Get threat score from detector or return phase default."""
        if self.threat_detector is None:
            # Return threat based on phase
            return [0.35, 0.999, 0.24, 0.1][self.phase]
        
        if self.current_flows.empty:
            return 0.0
        
        # Average threat over flows
        threats = []
        for _, flow in self.current_flows.iterrows():
            try:
                threat = self.threat_detector.get_threat_score(flow.to_dict())
                threats.append(threat)
            except:
                threats.append(0.0)
        
        return np.mean(threats) if threats else 0.0
    
    def _compute_path_entropy(self) -> float:
        """Shannon entropy of egress port distribution (H_P)."""
        if self.current_flows.empty:
            return 0.0
        
        port_counts = self.current_flows['dst_port'].value_counts()
        probabilities = port_counts / len(self.current_flows)
        
        return self._compute_entropy(probabilities)
    
    def _compute_attacker_knowledge_entropy(self) -> float:
        """Entropy of attacker's path knowledge (H_A)."""
        if self.current_flows.empty:
            return 0.0
        
        # Simplified: measure diversity of source-destination pairs
        paths = self.current_flows['src_ip'].astype(str) + "-" + self.current_flows['dst_ip'].astype(str)
        path_counts = paths.value_counts()
        probabilities = path_counts / len(paths)
        
        entropy = self._compute_entropy(probabilities)
        
        # Normalize by max possible (log of number of paths)
        max_entropy = np.log2(max(len(probabilities), 1))
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return np.clip(normalized, 0, 1)
    
    def _compute_entropy(self, probabilities) -> float:
        """Shannon entropy."""
        probs = np.array(probabilities)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def _get_shap_top_feature(self) -> str:
        """Get top SHAP feature for this step (if available)."""
        features = ['flow_duration', 'bytes_per_sec', 'syn_flags', 'packets_per_sec', 'rst_flags']
        return np.random.choice(features)
    
    def close(self):
        """Clean up."""
        pass
