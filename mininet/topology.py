#!/usr/bin/env python3
"""
topology.py — MTD-Playground Mininet Topology
===============================================
Creates the enterprise network from MTD-Playground paper (Figure 1):

    Client (h1, 10.0.0.10)  ─── S0 ─── DMZ Web (h3, 10.0.0.30)
                                 │
    Attacker (h2, 10.0.0.20) ── S1 ─── Internal App (h4, 10.0.0.100)
                                 │
                                S2 ─── DB Server (h5, 10.0.0.40)

    3 Open vSwitch switches, 5 hosts
    Connected to ONOS controller at 127.0.0.1:6653

Usage:
    sudo python3 topology.py
    (then in mininet CLI: pingall, h2 nmap 10.0.0.30, etc.)
"""

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink


def create_topology():
    """Create the MTD-Playground enterprise topology."""

    info("*** Creating MTD-Playground topology\n")

    net = Mininet(
        controller=RemoteController,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    # --- Controller (ONOS) ---
    info("*** Adding ONOS controller\n")
    c0 = net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6653,
    )

    # --- Switches (3 OVS switches) ---
    info("*** Adding switches\n")
    s0 = net.addSwitch("s1", dpid="0000000000000001", protocols="OpenFlow13")
    s1 = net.addSwitch("s2", dpid="0000000000000002", protocols="OpenFlow13")
    s2 = net.addSwitch("s3", dpid="0000000000000003", protocols="OpenFlow13")

    # --- Hosts (5 hosts matching paper topology) ---
    info("*** Adding hosts\n")
    h_client   = net.addHost("client",   ip="10.0.0.10/24",  mac="00:00:00:00:00:01")
    h_attacker = net.addHost("attacker", ip="10.0.0.20/24",  mac="00:00:00:00:00:02")
    h_web      = net.addHost("web",      ip="10.0.0.30/24",  mac="00:00:00:00:00:03")
    h_app      = net.addHost("app",      ip="10.0.0.100/24", mac="00:00:00:00:00:04")
    h_db       = net.addHost("db",       ip="10.0.0.40/24",  mac="00:00:00:00:00:05")

    # --- Links ---
    # Switch-to-switch links (3 paths, with bandwidth limits for realism)
    info("*** Adding links\n")
    net.addLink(s0, s1, bw=100, delay="2ms")   # Path A: S0-S1
    net.addLink(s0, s2, bw=100, delay="2ms")   # Path B: S0-S2
    net.addLink(s1, s2, bw=100, delay="2ms")   # Path C: S1-S2

    # Host-to-switch links
    net.addLink(h_client,   s0, bw=100, delay="1ms")  # Client → S0
    net.addLink(h_web,      s0, bw=100, delay="1ms")  # DMZ Web → S0
    net.addLink(h_attacker, s1, bw=100, delay="1ms")  # Attacker → S1
    net.addLink(h_app,      s1, bw=100, delay="1ms")  # App → S1
    net.addLink(h_db,       s2, bw=100, delay="1ms")  # DB → S2

    # --- Start ---
    info("*** Starting network\n")
    net.start()

    info("*** Network started!\n")
    info("*** Topology:\n")
    info("    Client (10.0.0.10)   --- S0 --- DMZ Web (10.0.0.30)\n")
    info("                              |                         \n")
    info("    Attacker (10.0.0.20) --- S1 --- App (10.0.0.100)    \n")
    info("                              |                         \n")
    info("                             S2 --- DB (10.0.0.40)      \n")
    info("\n")
    info("*** ONOS GUI: http://localhost:8181/onos/ui\n")
    info("*** Useful commands:\n")
    info("    pingall                        - test connectivity\n")
    info("    attacker ping web              - ping from attacker to web\n")
    info("    attacker nmap -sS 10.0.0.30    - scan web server\n")
    info("    client ping db                 - test client to DB path\n")
    info("\n")

    # Run CLI for interactive use
    CLI(net)

    # Cleanup
    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    create_topology()
