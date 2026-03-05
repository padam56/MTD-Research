"""
OpenFlow MTD Actions — Translates RL actions to network mutations
Converts RL policy decisions into OpenFlow group table updates
"""

import random
import numpy as np
from typing import List, Dict, Any
from onos_controller_interface import ONOSController


class OpenFlowMTD:
    """
    Translates RL actions to OpenFlow mutations.
    
    Action 0: No-op (keep current configuration)
    Action 1: Moderate mutation (MPLS label changes, swap 30% of flows)
    Action 2: Aggressive mutation (full path randomization)
    """
    
    def __init__(self, onos: ONOSController, topology_info: Dict[str, Any]):
        """
        Initialize MTD action handler.
        
        Args:
            onos: ONOSController instance
            topology_info: Dict with 'devices' and 'links'
        """
        self.onos = onos
        self.devices = topology_info.get('devices', [])
        self.links = topology_info.get('links', [])
        self.current_flows = []
        self.mtd_state = {}  # Track current MTD state
        
    def apply_action(self, action: int, current_flows: List[Dict] = None) -> bool:
        """
        Apply MTD action.
        
        Args:
            action: 0=no-move, 1=moderate, 2=aggressive
            current_flows: Optional list of current flows
            
        Returns:
            True if successful
        """
        if current_flows:
            self.current_flows = current_flows
        
        if action == 0:
            return self._apply_action_0()
        elif action == 1:
            return self._apply_action_1()
        elif action == 2:
            return self._apply_action_2()
        else:
            print(f"Unknown action: {action}")
            return False
    
    def _apply_action_0(self) -> bool:
        """No-op: maintain current configuration."""
        # No changes needed
        return True
    
    def _apply_action_1(self) -> bool:
        """
        Moderate MTD: Random MPLS label mutation.
        Affect ~30% of flows by changing their MPLS encapsulation.
        """
        if not self.devices:
            return False
        
        # Select random subset of flows (30%)
        if self.current_flows:
            n_affected = max(1, len(self.current_flows) // 3)
            affected_flows = random.sample(self.current_flows, min(n_affected, len(self.current_flows)))
        else:
            affected_flows = []
        
        success = True
        
        for device in self.devices:
            device_id = device.get('id')
            if not device_id:
                continue
            
            # Create group table for load balancing
            group_rule = {
                'id': random.randint(1, 0xffffffff),
                'type': 'SELECT',  # Load balancing group
                'buckets': [
                    {
                        'weight': random.randint(1, 10),
                        'actions': [
                            {'type': 'SET_FIELD', 'field': 'MPLS_LABEL', 'value': random.randint(100, 200)},
                            {'type': 'OUTPUT', 'port': random.choice([1, 2, 3, 4])}
                        ]
                    },
                    {
                        'weight': random.randint(1, 10),
                        'actions': [
                            {'type': 'SET_FIELD', 'field': 'MPLS_LABEL', 'value': random.randint(100, 200)},
                            {'type': 'OUTPUT', 'port': random.choice([1, 2, 3, 4])}
                        ]
                    }
                ]
            }
            
            if not self.onos.post_group(device_id, group_rule):
                success = False
        
        self.mtd_state['action_1_applied'] = True
        return success
    
    def _apply_action_2(self) -> bool:
        """
        Aggressive MTD: Full path randomization.
        Re-route all flows via random alternate paths.
        """
        if not self.devices:
            return False
        
        success = True
        
        # For each device, create random port mappings
        for device in self.devices:
            device_id = device.get('id')
            if not device_id:
                continue
            
            # Get device ports (assuming 4 ports per switch)
            ports = [1, 2, 3, 4]
            
            # Create group with random port distribution
            buckets = []
            for port in ports:
                buckets.append({
                    'weight': random.randint(1, 5),
                    'actions': [
                        {'type': 'SET_FIELD', 'field': 'VLAN_VID', 'value': random.randint(100, 200)},
                        {'type': 'OUTPUT', 'port': random.choice(ports)}
                    ]
                })
            
            group_rule = {
                'id': random.randint(1, 0xffffffff),
                'type': 'SELECT',
                'buckets': buckets
            }
            
            if not self.onos.post_group(device_id, group_rule):
                success = False
        
        self.mtd_state['action_2_applied'] = True
        return success
    
    def get_mtd_cost(self, action: int) -> float:
        """
        Estimate network cost of MTD action.
        
        Args:
            action: 0, 1, or 2
            
        Returns:
            Cost estimate [0, 1]
        """
        if action == 0:
            return 0.0
        elif action == 1:
            return 0.1  # 10% cost
        elif action == 2:
            return 0.3  # 30% cost (more re-routing)
        else:
            return 0.0
    
    def get_mtd_effectiveness(self, action: int, current_threat: float) -> float:
        """
        Estimate effectiveness of MTD action against current threat.
        
        Args:
            action: 0, 1, or 2
            current_threat: Current threat level [0, 1]
            
        Returns:
            Estimated threat reduction [0, 1]
        """
        if action == 0:
            return 0.0
        elif action == 1:
            return min(current_threat * 0.3, 1.0)  # Reduce threat by 30%
        elif action == 2:
            return min(current_threat * 0.7, 1.0)  # Reduce threat by 70%
        else:
            return 0.0
