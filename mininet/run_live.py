#!/usr/bin/env python3
"""
run_live.py — Run MTD Agent on Real ONOS/Mininet Network
==========================================================
Loads the trained DQN model and runs it against the live
ONOS SDN controller, making real path mutations.

Setup (run these first):
    1. sudo bash setup.sh                    ← Install ONOS, Mininet, OVS
    2. sudo python3 topology.py              ← Start Mininet topology (keep running)
    3. python3 run_live.py                   ← Run this in another terminal

Then in the Mininet CLI (terminal 2), run attacks:
    attacker ping -c 5 web                   ← Basic connectivity test
    attacker nmap -sS 10.0.0.30              ← Port scan (recon)
    attacker hping3 --flood -S 10.0.0.30     ← SYN flood (DDoS)

Watch the DQN agent react to the attacks by mutating paths.

Usage:
    python3 run_live.py                      ← DQN adaptive defense
    python3 run_live.py --strategy random    ← Random baseline
    python3 run_live.py --strategy none      ← No defense baseline
    python3 run_live.py --strategy periodic  ← Periodic baseline
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MTD] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run MTD agent on live ONOS network")
    parser.add_argument("--strategy", type=str, default="dqn",
                        choices=["none", "random", "periodic", "dqn"])
    parser.add_argument("--onos-host", type=str, default="127.0.0.1")
    parser.add_argument("--onos-port", type=int, default=8181)
    parser.add_argument("--model-path", type=str, default="../models/best_model.zip")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between each decision step")
    parser.add_argument("--output", type=str, default="../results/live_results.csv")
    args = parser.parse_args()

    # Import after argparse so --help works without deps
    from mtd_env_live import MTDLiveEnv

    # Connect to ONOS
    logging.info(f"Connecting to ONOS at {args.onos_host}:{args.onos_port}...")
    env = MTDLiveEnv(
        onos_host=args.onos_host,
        onos_port=args.onos_port,
        render_mode="human",
    )

    # Print network summary
    from onos_client import ONOSClient
    client = ONOSClient(args.onos_host, args.onos_port)
    summary = client.get_topology_summary()
    logging.info(f"Network: {summary['switches']} switches, {summary['hosts']} hosts, "
                 f"{summary['flows']} flows, {summary['links']} links")

    # Load strategy
    model = None
    if args.strategy == "dqn":
        try:
            from stable_baselines3 import DQN
            if os.path.exists(args.model_path):
                model = DQN.load(args.model_path, env=env)
                logging.info(f"Loaded DQN model from {args.model_path}")
            else:
                logging.error(f"Model not found: {args.model_path}")
                logging.info("Falling back to random strategy")
                args.strategy = "random"
        except ImportError:
            logging.error("stable-baselines3 not installed")
            args.strategy = "random"

    action_names = {0: "HOLD", 1: "MODERATE", 2: "AGGRESSIVE"}

    # CSV output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = [
        "step", "timestamp", "action", "action_name", "reward",
        "flow_count", "latency_ms", "path_entropy", "knowledge_estimate",
        "mutation_count", "service_availability", "cumulative_reward",
    ]
    csvfile = open(args.output, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Run
    logging.info(f"Starting MTD loop: strategy={args.strategy}, "
                 f"steps={args.steps}, interval={args.interval}s")
    logging.info("=" * 60)

    obs, _ = env.reset()
    cumulative_reward = 0.0

    try:
        for step in range(args.steps):
            # Pick action
            if args.strategy == "dqn" and model is not None:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            elif args.strategy == "random":
                action = np.random.randint(0, 3)
            elif args.strategy == "periodic":
                action = 2 if step % 10 == 0 else 0
            else:
                action = 0

            # Execute
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward

            # Log
            writer.writerow({
                "step": step + 1,
                "timestamp": time.strftime("%H:%M:%S"),
                "action": action,
                "action_name": action_names[action],
                "reward": round(reward, 4),
                "flow_count": info.get("flow_count", 0),
                "latency_ms": round(info.get("latency_ms", 0), 2),
                "path_entropy": round(info.get("path_entropy", 0), 4),
                "knowledge_estimate": round(info.get("knowledge_estimate", 0), 4),
                "mutation_count": info.get("mutation_count", 0),
                "service_availability": round(info.get("service_availability", 1.0), 4),
                "cumulative_reward": round(cumulative_reward, 4),
            })
            csvfile.flush()

            if terminated or truncated:
                break

            # Wait before next step
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logging.info("Stopped by user (Ctrl+C)")

    finally:
        csvfile.close()
        logging.info("=" * 60)
        logging.info(f"Done. {step + 1} steps completed.")
        logging.info(f"Total mutations: {env.client.mutation_count}")
        logging.info(f"Cumulative reward: {cumulative_reward:.2f}")
        logging.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
