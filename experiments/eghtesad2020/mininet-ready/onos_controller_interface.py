"""
ONOS Controller REST API Interface
Communicates with ONOS via REST to manage flows and topology
"""

import requests
import json
from typing import Dict, List, Any
import time


class ONOSController:
    """
    REST client for ONOS SDN Controller.
    Handles flow management, topology queries, and group table operations.
    """
    
    def __init__(self, ip: str = "127.0.0.1", port: int = 8181, 
                 username: str = "karaf", password: str = "karaf"):
        """
        Initialize ONOS controller connection.
        
        Args:
            ip: ONOS controller IP (default: 127.0.0.1)
            port: ONOS REST API port (default: 8181)
            username: ONOS credentials (default: karaf)
            password: ONOS credentials (default: karaf)
        """
        self.base_url = f"http://{ip}:{port}/onos/v1"
        self.auth = (username, password)
        self.timeout = 5
        
    def health_check(self) -> bool:
        """Check if ONOS is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/devices", auth=self.auth, timeout=self.timeout)
            return resp.status_code == 200
        except:
            return False
    
    def get_devices(self) -> List[Dict[str, Any]]:
        """Get all OpenFlow switches (devices)."""
        try:
            resp = requests.get(f"{self.base_url}/devices", auth=self.auth, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get('devices', [])
        except Exception as e:
            print(f"Error fetching devices: {e}")
        return []
    
    def get_hosts(self) -> List[Dict[str, Any]]:
        """Get all hosts."""
        try:
            resp = requests.get(f"{self.base_url}/hosts", auth=self.auth, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get('hosts', [])
        except Exception as e:
            print(f"Error fetching hosts: {e}")
        return []
    
    def get_flows(self, device_id: str = None) -> List[Dict[str, Any]]:
        """
        Get flows.
        
        Args:
            device_id: Optional—get flows on specific device
            
        Returns:
            List of flow dictionaries
        """
        try:
            if device_id:
                url = f"{self.base_url}/flows/{device_id}"
            else:
                url = f"{self.base_url}/flows"
            
            resp = requests.get(url, auth=self.auth, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get('flows', [])
        except Exception as e:
            print(f"Error fetching flows: {e}")
        return []
    
    def get_links(self) -> List[Dict[str, Any]]:
        """Get all network links."""
        try:
            resp = requests.get(f"{self.base_url}/links", auth=self.auth, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get('links', [])
        except Exception as e:
            print(f"Error fetching links: {e}")
        return []
    
    def post_flow(self, device_id: str, flow_rule: Dict) -> bool:
        """
        Install a flow rule on a switch.
        
        Args:
            device_id: Switch ID (e.g., "of:0000000000000001")
            flow_rule: Flow rule dictionary (OpenFlow format)
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/flows/{device_id}"
            headers = {'Content-Type': 'application/json'}
            payload = {'flows': [flow_rule]}
            
            resp = requests.post(url, json=payload, auth=self.auth, 
                               headers=headers, timeout=self.timeout)
            return resp.status_code in [200, 201]
        except Exception as e:
            print(f"Error posting flow: {e}")
        return False
    
    def post_group(self, device_id: str, group_rule: Dict) -> bool:
        """
        Install a group table entry (for multi-path forwarding).
        
        Args:
            device_id: Switch ID
            group_rule: Group rule dictionary
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/groups/{device_id}"
            headers = {'Content-Type': 'application/json'}
            payload = {'groups': [group_rule]}
            
            resp = requests.post(url, json=payload, auth=self.auth,
                               headers=headers, timeout=self.timeout)
            return resp.status_code in [200, 201]
        except Exception as e:
            print(f"Error posting group: {e}")
        return False
    
    def post_intent(self, intent: Dict) -> bool:
        """
        Install an intent (higher-level routing policy).
        
        Args:
            intent: Intent dictionary
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/intents"
            headers = {'Content-Type': 'application/json'}
            payload = intent
            
            resp = requests.post(url, json=payload, auth=self.auth,
                               headers=headers, timeout=self.timeout)
            return resp.status_code in [200, 201]
        except Exception as e:
            print(f"Error posting intent: {e}")
        return False
    
    def delete_flow(self, device_id: str, flow_id: str) -> bool:
        """Delete a flow rule."""
        try:
            url = f"{self.base_url}/flows/{device_id}/{flow_id}"
            resp = requests.delete(url, auth=self.auth, timeout=self.timeout)
            return resp.status_code in [200, 204]
        except Exception as e:
            print(f"Error deleting flow: {e}")
        return False
    
    def get_topology(self) -> Dict[str, Any]:
        """Get network topology (devices + links)."""
        topology = {
            'devices': self.get_devices(),
            'links': self.get_links(),
            'hosts': self.get_hosts()
        }
        return topology
    
    def compute_path(self, src_host_ip: str, dst_host_ip: str) -> List[str]:
        """
        Compute path between two hosts (placeholder).
        In real ONOS, this would use the Topology service.
        
        Args:
            src_host_ip: Source host IP
            dst_host_ip: Destination host IP
            
        Returns:
            List of switch IDs in path
        """
        # Placeholder—would need to implement actual path computation
        # In practice, ONOS uses the topology service and dijkstra
        return []
    
    def install_path_rule(self, path_switches: List[str], src_ip: str, dst_ip: str,
                         actions: List[Dict] = None) -> List[bool]:
        """
        Install forwarding rules along a path.
        
        Args:
            path_switches: List of switch IDs in path
            src_ip: Source IP
            dst_ip: Destination IP
            actions: Optional custom OpenFlow actions
            
        Returns:
            List of booleans indicating success per switch
        """
        results = []
        
        for i, switch_id in enumerate(path_switches):
            # Determine ingress/egress port (placeholder logic)
            if i == 0:
                # First switch: ingress from external port
                in_port = 1
            else:
                in_port = 2  # From previous switch
            
            if i == len(path_switches) - 1:
                # Last switch: egress to external port
                out_port = 3
            else:
                out_port = 2  # To next switch
            
            # Create flow rule
            flow_rule = {
                'tableId': 0,
                'priority': 100,
                'selector': {
                    'criteria': [
                        {'type': 'IN_PORT', 'port': in_port},
                        {'type': 'ETH_TYPE', 'ethType': '0x0800'},
                        {'type': 'IPV4_SRC', 'ip': src_ip + '/32'},
                        {'type': 'IPV4_DST', 'ip': dst_ip + '/32'}
                    ]
                },
                'treatment': {
                    'instructions': [
                        {'type': 'OUTPUT', 'port': out_port}
                    ]
                },
                'timeout': 300
            }
            
            success = self.post_flow(switch_id, flow_rule)
            results.append(success)
            
            if not success:
                print(f"Failed to install flow on {switch_id}")
        
        return results
