"""
Mininet Topology Builder
Creates network topologies for MTD testing
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.util import quietRun
import time


class MTDTopology(Topo):
    """
    Custom topology for MTD testing.
    Supports multiple sizes: small, medium, large
    """
    
    def __init__(self, size='small'):
        """
        Initialize topology.
        
        Args:
            size: 'small' (3 switches, 10 hosts),
                  'medium' (8 switches, 30 hosts),
                  'large' (20 switches, 100 hosts)
        """
        Topo.__init__(self)
        
        self.size = size
        self.switches = []
        self.hosts = []
        
        if size == 'small':
            self._build_small()
        elif size == 'medium':
            self._build_medium()
        elif size == 'large':
            self._build_large()
        else:
            raise ValueError(f"Unknown size: {size}")
    
    def _build_small(self):
        """Build small topology: 3 switches, 10 hosts."""
        # Switches (S1, S2, S3)
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow13')
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow13')
        s3 = self.addSwitch('s3', cls=OVSSwitch, protocols='OpenFlow13')
        
        self.switches = [s1, s2, s3]
        
        # Hosts (h1-h10)
        hosts = []
        for i in range(1, 11):
            h = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            hosts.append(h)
        
        self.hosts = hosts
        
        # Connect hosts to switches
        for i, h in enumerate(hosts):
            if i < 4:
                self.addLink(h, s1, bw=100)
            elif i < 7:
                self.addLink(h, s2, bw=100)
            else:
                self.addLink(h, s3, bw=100)
        
        # Switch-to-switch links
        self.addLink(s1, s2, bw=1000, delay='10ms')
        self.addLink(s2, s3, bw=1000, delay='10ms')
        self.addLink(s3, s1, bw=1000, delay='10ms')  # Triangle topology
    
    def _build_medium(self):
        """Build medium topology: 8 switches, 30 hosts (tree)."""
        # Root switches (S1, S2)
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow13')
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow13')
        
        # Aggregate switches (S3-S6)
        agg_switches = []
        for i in range(3, 7):
            s = self.addSwitch(f's{i}', cls=OVSSwitch, protocols='OpenFlow13')
            agg_switches.append(s)
        
        # Edge switches (S7-S8)
        edge_switches = []
        for i in range(7, 9):
            s = self.addSwitch(f's{i}', cls=OVSSwitch, protocols='OpenFlow13')
            edge_switches.append(s)
        
        self.switches = [s1, s2] + agg_switches + edge_switches
        
        # Hosts (h1-h30)
        hosts = []
        for i in range(1, 31):
            h = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            hosts.append(h)
        
        self.hosts = hosts
        
        # Connect hosts to edge switches
        for i, h in enumerate(hosts):
            switch = edge_switches[i % len(edge_switches)]
            self.addLink(h, switch, bw=100)
        
        # Connect edge switches to aggregate
        for edge_sw in edge_switches:
            for agg_sw in agg_switches:
                self.addLink(edge_sw, agg_sw, bw=1000, delay='5ms')
        
        # Connect aggregate to root
        for agg_sw in agg_switches:
            self.addLink(agg_sw, s1, bw=1000, delay='5ms')
            self.addLink(agg_sw, s2, bw=1000, delay='5ms')
        
        # Root connection
        self.addLink(s1, s2, bw=10000, delay='1ms')
    
    def _build_large(self):
        """Build large topology: 20 switches, 100 hosts."""
        # Create switches in layers
        root_switches = []
        for i in range(1, 3):
            s = self.addSwitch(f's{i}', cls=OVSSwitch, protocols='OpenFlow13')
            root_switches.append(s)
        
        agg_switches = []
        for i in range(3, 10):
            s = self.addSwitch(f's{i}', cls=OVSSwitch, protocols='OpenFlow13')
            agg_switches.append(s)
        
        edge_switches = []
        for i in range(10, 21):
            s = self.addSwitch(f's{i}', cls=OVSSwitch, protocols='OpenFlow13')
            edge_switches.append(s)
        
        self.switches = root_switches + agg_switches + edge_switches
        
        # Hosts (h1-h100)
        hosts = []
        for i in range(1, 101):
            h = self.addHost(f'h{i}', ip=f'10.0.{i//256}.{i%256}/24', defaultRoute='via 10.0.0.1')
            hosts.append(h)
        
        self.hosts = hosts
        
        # Connect hosts to edge switches
        hosts_per_edge = len(hosts) // len(edge_switches)
        for edge_idx, edge_sw in enumerate(edge_switches):
            start = edge_idx * hosts_per_edge
            end = start + hosts_per_edge if edge_idx < len(edge_switches) - 1 else len(hosts)
            for h_idx in range(start, end):
                self.addLink(hosts[h_idx], edge_sw, bw=100)
        
        # Connect edge to aggregate (6:1 oversubscription)
        for edge_sw in edge_switches:
            for agg_sw in agg_switches:
                self.addLink(edge_sw, agg_sw, bw=1000, delay='5ms')
        
        # Connect aggregate to root (each agg to all roots)
        for agg_sw in agg_switches:
            for root_sw in root_switches:
                self.addLink(agg_sw, root_sw, bw=10000, delay='2ms')
        
        # Root-to-root
        for i in range(len(root_switches) - 1):
            self.addLink(root_switches[i], root_switches[i+1], bw=40000, delay='1ms')


def start_mininet(topo_size='small', controller_ip='127.0.0.1', controller_port=6653):
    """
    Start Mininet network with ONOS controller.
    
    Args:
        topo_size: 'small', 'medium', or 'large'
        controller_ip: ONOS controller IP
        controller_port: ONOS OpenFlow port
        
    Returns:
        Mininet network object
    """
    print(f"[Mininet] Building {topo_size} topology...")
    
    topo = MTDTopology(size=topo_size)
    
    print(f"[Mininet] Starting network with controller {controller_ip}:{controller_port}...")
    
    net = Mininet(
        topo=topo,
        controller=RemoteController(
            'onos',
            ip=controller_ip,
            port=controller_port,
            protocols='OpenFlow13'
        ),
        switch=OVSSwitch,
        link=TCLink
    )
    
    net.start()
    
    # Wait for switches to connect to controller
    time.sleep(3)
    
    print(f"[Mininet] Network started!")
    print(f"  Switches: {len(topo.switches)}")
    print(f"  Hosts: {len(topo.hosts)}")
    
    return net


def stop_mininet(net):
    """Stop Mininet network."""
    print("[Mininet] Stopping network...")
    net.stop()
    # Clean up
    import os
    os.system("sudo mn -c > /dev/null 2>&1")


if __name__ == '__main__':
    # Test: start small topology
    net = start_mininet('small')
    
    # Add some ping tests
    print("\n[Test] h1 → h5 ping test:")
    result = net.pingFull(tolerate=True)
    
    # Cleanup
    stop_mininet(net)
