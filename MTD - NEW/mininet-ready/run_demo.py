#!/usr/bin/env python3
"""
run_demo.py — Full MTD Demo: Topology + Attacker + DQN Defender
================================================================
Single script that:
  1. Creates the Mininet topology (5 hosts, 3 switches)
  2. Runs attacker actions (ping sweeps, port scans, flooding)
  3. Runs the DQN defender reacting to attacks in real time
  4. Logs everything to CSV and prints live to terminal

This is the "show everything" demo for the MTD-Playground paper.

Usage:
    sudo python3 run_demo.py
    sudo python3 run_demo.py --strategy none      # No defense (watch attacker win)
    sudo python3 run_demo.py --strategy dqn        # DQN defending (default)
    sudo python3 run_demo.py --strategy random     # Random defense
"""

import argparse
import os
import sys
import time
import threading
import csv
import subprocess
import numpy as np

# Add parent dir so we can import models
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference-only"))

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel

from onos_client import ONOSClient


def create_network():
    """Create MTD-Playground topology."""
    net = Mininet(controller=RemoteController, switch=OVSSwitch, link=TCLink, autoSetMacs=True)

    c0 = net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)

    s0 = net.addSwitch("s1", dpid="0000000000000001", protocols="OpenFlow13")
    s1 = net.addSwitch("s2", dpid="0000000000000002", protocols="OpenFlow13")
    s2 = net.addSwitch("s3", dpid="0000000000000003", protocols="OpenFlow13")

    h_client   = net.addHost("client",   ip="10.0.0.10/24",  mac="00:00:00:00:00:01")
    h_attacker = net.addHost("attacker", ip="10.0.0.20/24",  mac="00:00:00:00:00:02")
    h_web      = net.addHost("web",      ip="10.0.0.30/24",  mac="00:00:00:00:00:03")
    h_app      = net.addHost("app",      ip="10.0.0.100/24", mac="00:00:00:00:00:04")
    h_db       = net.addHost("db",       ip="10.0.0.40/24",  mac="00:00:00:00:00:05")

    net.addLink(s0, s1, bw=100, delay="2ms")
    net.addLink(s0, s2, bw=100, delay="2ms")
    net.addLink(s1, s2, bw=100, delay="2ms")
    net.addLink(h_client, s0, bw=100, delay="1ms")
    net.addLink(h_web, s0, bw=100, delay="1ms")
    net.addLink(h_attacker, s1, bw=100, delay="1ms")
    net.addLink(h_app, s1, bw=100, delay="1ms")
    net.addLink(h_db, s2, bw=100, delay="1ms")

    net.start()
    return net


STAGE_NAMES = ["Reconnaissance", "Initial Access", "Lateral Movement", "Priv Escalation", "Exfiltration"]


def attacker_thread(net, attack_log, stop_event):
    """
    Simulate multi-stage attack from the attacker host.
    Each stage runs real network commands.
    """
    attacker = net.get("attacker")
    stage = 0

    while not stop_event.is_set() and stage < 5:
        timestamp = time.strftime("%H:%M:%S")

        if stage == 0:
            # RECON: ping sweep to discover hosts
            attack_log.append({"time": timestamp, "stage": stage, "action": "Ping sweep 10.0.0.0/24"})
            print(f"  [ATTACKER] Stage 0: Reconnaissance — scanning network...")
            for ip in ["10.0.0.10", "10.0.0.30", "10.0.0.40", "10.0.0.100"]:
                result = attacker.cmd(f"ping -c 1 -W 1 {ip}")
                alive = "1 received" in result
                if alive:
                    print(f"  [ATTACKER]   Found host: {ip}")
                    attack_log.append({"time": timestamp, "stage": stage, "action": f"Discovered {ip}"})
            time.sleep(2)
            stage = 1

        elif stage == 1:
            # INITIAL ACCESS: attempt to reach DMZ web server
            attack_log.append({"time": timestamp, "stage": stage, "action": "Targeting DMZ Web 10.0.0.30"})
            print(f"  [ATTACKER] Stage 1: Initial Access — targeting DMZ Web Server...")
            result = attacker.cmd("ping -c 5 -W 1 10.0.0.30")
            loss = "100% packet loss" in result
            if loss:
                print(f"  [ATTACKER]   Web server unreachable! MTD may be active.")
                attack_log.append({"time": timestamp, "stage": stage, "action": "BLOCKED — web server unreachable"})
                time.sleep(3)
                # Retry
                continue
            else:
                print(f"  [ATTACKER]   Web server reached! Attempting exploitation...")
                attack_log.append({"time": timestamp, "stage": stage, "action": "Reached web server"})
                time.sleep(2)
                stage = 2

        elif stage == 2:
            # LATERAL MOVEMENT: try to reach internal app server
            attack_log.append({"time": timestamp, "stage": stage, "action": "Pivoting to App Server 10.0.0.100"})
            print(f"  [ATTACKER] Stage 2: Lateral Movement — pivoting to App Server...")
            result = attacker.cmd("ping -c 5 -W 1 10.0.0.100")
            # Also try traceroute to see path
            trace = attacker.cmd("traceroute -n -m 5 -w 1 10.0.0.100 2>/dev/null || echo 'no traceroute'")
            loss = "100% packet loss" in result
            if loss:
                print(f"  [ATTACKER]   App server unreachable! Path may have changed.")
                attack_log.append({"time": timestamp, "stage": stage, "action": "BLOCKED — path disrupted"})
                time.sleep(3)
                continue
            else:
                print(f"  [ATTACKER]   App server reached!")
                attack_log.append({"time": timestamp, "stage": stage, "action": "Reached app server"})
                time.sleep(2)
                stage = 3

        elif stage == 3:
            # PRIV ESCALATION: try to reach DB server
            attack_log.append({"time": timestamp, "stage": stage, "action": "Targeting DB Server 10.0.0.40"})
            print(f"  [ATTACKER] Stage 3: Privilege Escalation — targeting DB Server...")
            result = attacker.cmd("ping -c 5 -W 1 10.0.0.40")
            loss = "100% packet loss" in result
            if loss:
                print(f"  [ATTACKER]   DB server unreachable! MTD blocking lateral path.")
                attack_log.append({"time": timestamp, "stage": stage, "action": "BLOCKED — DB unreachable"})
                time.sleep(3)
                continue
            else:
                print(f"  [ATTACKER]   DB server reached! Attempting data extraction...")
                attack_log.append({"time": timestamp, "stage": stage, "action": "Reached DB server"})
                time.sleep(2)
                stage = 4

        elif stage == 4:
            # EXFILTRATION: large data transfer from DB
            attack_log.append({"time": timestamp, "stage": stage, "action": "DATA EXFILTRATION!"})
            print(f"  [ATTACKER] Stage 4: *** DATA EXFILTRATION — ATTACK SUCCEEDED ***")
            # Simulate large transfer
            attacker.cmd("ping -c 10 -s 1400 10.0.0.40")
            break

        if stop_event.is_set():
            break

    if stage < 4:
        print(f"  [ATTACKER] Attack stopped at Stage {stage}: {STAGE_NAMES[stage]}")
    return stage


def defender_thread(client, strategy, model, results_log, stop_event, step_interval=2.0):
    """
    Run the MTD defender — reads ONOS state, picks actions, mutates paths.
    """
    step = 0
    cumulative_reward = 0
    prev_flow_hash = None
    knowledge_estimate = 0.0
    baseline_latency = client.get_controller_latency()

    while not stop_event.is_set():
        step += 1
        timestamp = time.strftime("%H:%M:%S")

        # Read state from ONOS
        flows = client.get_flows()
        flow_count = len(flows)
        latency = client.get_controller_latency()

        # Compute path entropy
        port_counts = {}
        for f in flows:
            for instr in f.get("treatment", {}).get("instructions", []):
                if instr.get("type") == "OUTPUT":
                    key = f"{f.get('deviceId', '')}_{instr.get('port', '')}"
                    port_counts[key] = port_counts.get(key, 0) + 1
        if port_counts:
            counts = np.array(list(port_counts.values()), dtype=float)
            probs = counts / counts.sum()
            h = -np.sum(probs * np.log2(probs + 1e-10))
            h_max = np.log2(max(len(port_counts), 2))
            path_entropy = float(np.clip(h / h_max, 0.0, 1.0))
        else:
            path_entropy = 0.5

        # Knowledge estimate
        flow_hash = hash(str(sorted([f.get("id", "") for f in flows[:50]])))
        if prev_flow_hash is not None and flow_hash == prev_flow_hash:
            knowledge_estimate = min(1.0, knowledge_estimate + 0.05)
        else:
            knowledge_estimate = max(0.0, knowledge_estimate - 0.1)
        prev_flow_hash = flow_hash

        # Build observation (14-dim, same as offline env)
        obs = np.zeros(14, dtype=np.float32)
        obs[0] = min(flow_count / 500.0, 1.0)
        obs[3] = min(latency / 200.0, 1.0)
        obs[6] = knowledge_estimate
        obs[7] = path_entropy
        obs[8] = 1.0 - knowledge_estimate
        obs[13] = max(0.0, min(1.0, 2.0 - latency / max(baseline_latency, 1)))

        # Pick action
        if strategy == "dqn" and model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif strategy == "random":
            action = np.random.randint(0, 3)
        elif strategy == "periodic":
            action = 2 if step % 5 == 0 else 0
        else:
            action = 0

        # Execute
        action_names = {0: "HOLD", 1: "MODERATE", 2: "AGGRESSIVE"}
        if action > 0:
            client.execute_mutation(action)

        # Reward
        security = 1.0 * (1.0 - knowledge_estimate)
        cost = 0.3 * action
        overhead = 0.2 * min(latency / 200.0, 1.0)
        reward = security + 0.3 * path_entropy - cost - overhead
        cumulative_reward += reward

        # Display
        action_color = "\033[90m" if action == 0 else "\033[93m" if action == 1 else "\033[91m"
        print(f"  [DEFENDER] Step {step:>3d} | {action_color}{action_names[action]}\033[0m | "
              f"Flows={flow_count:<4d} Latency={latency:.1f}ms H_P={path_entropy:.3f} "
              f"Knowledge={knowledge_estimate:.0%} Mutations={client.mutation_count} "
              f"Reward={cumulative_reward:.1f}")

        results_log.append({
            "step": step,
            "time": timestamp,
            "action": action,
            "action_name": action_names[action],
            "flow_count": flow_count,
            "latency_ms": round(latency, 2),
            "path_entropy": round(path_entropy, 4),
            "knowledge_estimate": round(knowledge_estimate, 4),
            "mutations": client.mutation_count,
            "reward": round(reward, 4),
            "cumulative_reward": round(cumulative_reward, 4),
        })

        time.sleep(step_interval)

    return cumulative_reward


def main():
    parser = argparse.ArgumentParser(description="MTD-Playground Full Demo")
    parser.add_argument("--strategy", default="dqn", choices=["none", "random", "periodic", "dqn"])
    parser.add_argument("--duration", type=int, default=60, help="Demo duration in seconds")
    parser.add_argument("--model-path", default=os.path.join(os.path.dirname(__file__), "..", "models", "best_model.zip"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "results", "live_demo_results.csv"))
    args = parser.parse_args()

    setLogLevel("warning")

    # Load DQN model
    model = None
    if args.strategy == "dqn":
        try:
            from stable_baselines3 import DQN
            from mtd_env import MTDPlaygroundEnv
            if os.path.exists(args.model_path):
                dummy_env = MTDPlaygroundEnv(seed=0)
                model = DQN.load(args.model_path, env=dummy_env)
                print(f"[+] DQN model loaded from {args.model_path}")
            else:
                print(f"[!] Model not found: {args.model_path}, falling back to random")
                args.strategy = "random"
        except ImportError:
            print("[!] stable-baselines3 not installed, falling back to random")
            args.strategy = "random"

    # Create network
    print("\n" + "=" * 70)
    print("  MTD-PLAYGROUND LIVE DEMO")
    print(f"  Strategy: {args.strategy.upper()}")
    print(f"  Duration: {args.duration}s")
    print("=" * 70)

    print("\n[+] Creating Mininet topology...")
    net = create_network()

    # Wait for ONOS to discover topology
    print("[+] Waiting for ONOS to discover hosts...")
    time.sleep(5)

    # Generate initial traffic so ONOS discovers all hosts
    for host_name in ["client", "web", "app", "db", "attacker"]:
        host = net.get(host_name)
        host.cmd("ping -c 1 -W 1 10.0.0.10 &")
    time.sleep(3)

    # Connect to ONOS
    client = ONOSClient()
    summary = client.get_topology_summary()
    print(f"[+] ONOS: {summary['switches']} switches, {summary['hosts']} hosts, "
          f"{summary['flows']} flows, {summary['links']} links")

    print(f"\n[+] Starting demo... (Ctrl+C to stop)")
    print(f"    Watch ONOS GUI: http://localhost:8181/onos/ui (onos/rocks)")
    print()

    # Shared state
    attack_log = []
    results_log = []
    stop_event = threading.Event()

    # Start defender thread
    def_thread = threading.Thread(
        target=defender_thread,
        args=(client, args.strategy, model, results_log, stop_event, 2.0),
        daemon=True,
    )
    def_thread.start()

    # Run attacker (main thread)
    try:
        final_stage = attacker_thread(net, attack_log, stop_event)
    except KeyboardInterrupt:
        print("\n[!] Stopped by user")
        final_stage = -1

    # Stop defender
    stop_event.set()
    def_thread.join(timeout=5)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if results_log:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
            writer.writeheader()
            writer.writerows(results_log)

    # Final summary
    print()
    print("=" * 70)
    print("  DEMO RESULTS")
    print("=" * 70)
    result = "BREACHED" if final_stage >= 4 else "DEFENDED"
    color = "\033[91m" if final_stage >= 4 else "\033[92m"
    print(f"  Result:     {color}{result}\033[0m")
    print(f"  Strategy:   {args.strategy.upper()}")
    print(f"  Attacker reached: Stage {final_stage} ({STAGE_NAMES[min(final_stage, 4)] if final_stage >= 0 else 'N/A'})")
    print(f"  Mutations:  {client.mutation_count}")
    if results_log:
        print(f"  Avg Latency: {np.mean([r['latency_ms'] for r in results_log]):.1f}ms")
        print(f"  Avg H_P:     {np.mean([r['path_entropy'] for r in results_log]):.3f}")
        print(f"  Total Reward: {results_log[-1]['cumulative_reward']:.1f}")
    print(f"  Results saved: {args.output}")
    print("=" * 70)

    # Cleanup
    print("\n[+] Stopping Mininet...")
    net.stop()


if __name__ == "__main__":
    main()
