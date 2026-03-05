#!/usr/bin/env python3
"""
Eghtesad (2020) — Mininet Integration
Real-time adaptive MTD learning with ONOS controller

Usage:
    sudo python run_mininet.py --topology-size small --duration 300 --model-path ../reference-only/models/agent.zip
"""

import argparse
import time
import os
import csv
from pathlib import Path

from mininet_topology import start_mininet, stop_mininet
from onos_controller_interface import ONOSController
from openflow_mtd_actions import OpenFlowMTD


def main():
    parser = argparse.ArgumentParser(description="Eghtesad MTD — Mininet + ONOS")
    parser.add_argument('--topology-size', default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--controller-ip', default='127.0.0.1')
    parser.add_argument('--controller-port', type=int, default=6653)
    parser.add_argument('--model-path', default=None, help='Pre-trained RL model')
    parser.add_argument('--attack-type', default='none', 
                       choices=['none', 'portscanning', 'ddos', 'exfiltration'])
    parser.add_argument('--duration', type=int, default=300, help='Test duration (seconds)')
    parser.add_argument('--output-dir', default='./mininet_results')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Eghtesad (2020) — Mininet + ONOS Integration")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize ONOS connection
        print(f"\n[ONOS] Connecting to {args.controller_ip}:{args.controller_port}...")
        onos = ONOSController(ip=args.controller_ip, port=args.controller_port)
        
        if not onos.health_check():
            print("  ✗ ONOS not responding!")
            print("  Please ensure ONOS is running: ./bin/onos-karaf")
            return
        print("  ✓ ONOS connected")
        
        # Start Mininet
        print(f"\n[Mininet] Starting {args.topology_size} topology...")
        net = start_mininet(args.topology_size, args.controller_ip, args.controller_port)
        time.sleep(2)
        
        # Check topology
        topology = onos.get_topology()
        print(f"  Devices: {len(topology['devices'])}")
        print(f"  Hosts: {len(topology['hosts'])}")
        
        # Initialize MTD
        print(f"\n[MTD] Initializing OpenFlow MTD...")
        mtd = OpenFlowMTD(onos, topology)
        
        # Load RL model if provided
        rl_agent = None
        if args.model_path:
            from stable_baselines3 import DQN
            print(f"  Loading pre-trained model: {args.model_path}")
            rl_agent = DQN.load(args.model_path)
        
        # Main loop
        print(f"\n[Test] Running for {args.duration} seconds (attack type: {args.attack_type})...")
        
        results_file = os.path.join(args.output_dir, 'threat_log.csv')
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'threat_prob', 'action', 'latency_ms'])
            
            start_time = time.time()
            step = 0
            
            while time.time() - start_time < args.duration:
                step += 1
                t = time.time() - start_time
                
                # Get current threat level (placeholder)
                threat_prob = 0.3 + 0.2 * (0.5 + 0.5 * (step % 100) / 50)  # Simulated
                
                # RL decision
                if rl_agent:
                    # Use trained agent
                    obs = [threat_prob, 0.5, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.6, 0.3, 0.1, 0.0, 0.0]
                    action, _ = rl_agent.predict(obs)
                else:
                    # Random action
                    action = step % 3
                
                # Apply MTD action
                t_mtd_start = time.time()
                mtd.apply_action(int(action))
                latency_ms = (time.time() - t_mtd_start) * 1000
                
                # Log
                writer.writerow([t, threat_prob, int(action), latency_ms])
                
                # Print progress
                if step % 10 == 0:
                    print(f"  t={t:.1f}s: threat={threat_prob:.3f}, action={action}, latency={latency_ms:.1f}ms")
                
                time.sleep(0.1)  # 10 Hz decision rate
        
        print(f"\n✓ Results saved to {results_file}")
        
    except KeyboardInterrupt:
        print("\nInterrupt received")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[Cleanup] Stopping Mininet...")
        try:
            stop_mininet(net)
        except:
            pass
        
        print("=" * 70)


if __name__ == '__main__':
    main()
