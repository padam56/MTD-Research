#!/usr/bin/env python3
"""
simulate.py — Live Visual MTD-Playground Simulation
=====================================================
Real-time terminal simulation showing:
  - Enterprise network topology (5 hosts, 3 switches)
  - Multi-stage attacker progressing through kill chain
  - DQN defender deciding when to mutate paths
  - Path entropy, attacker knowledge, metrics updating live

Based on MTD-Playground paper topology (Figure 1) and
Eghtesad et al. (2020) GameSec Markov Game formulation.

Usage:
    python3 simulate.py                    # DQN adaptive defense
    python3 simulate.py --strategy none    # No defense (attacker wins fast)
    python3 simulate.py --strategy random  # Random mutations
    python3 simulate.py --strategy periodic --speed 0.3
    python3 simulate.py --side-by-side     # Compare all strategies
"""

import argparse
import os
import sys
import time
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich.columns import Columns
from rich.progress_bar import ProgressBar

from mtd_env import MTDPlaygroundEnv

console = Console()

STAGE_NAMES = [
    "0: Reconnaissance",
    "1: Initial Access",
    "2: Lateral Movement",
    "3: Priv Escalation",
    "4: DATA EXFILTRATED!",
]

STAGE_COLORS = ["green", "yellow", "rgb(255,165,0)", "red", "bold white on red"]

PATH_LABELS = ["S0->S1 (via Core)", "S0->S2 (via Edge)", "S1->S2 (Direct)"]


def build_topology_display(env, action, step):
    """Build ASCII topology with live path highlighting."""
    net = env.network
    atk = env.attacker
    path_probs = net.path_probs

    # Determine which path is "active" (highest probability)
    active_path = int(np.argmax(path_probs))

    # Color paths by probability
    def path_bar(idx):
        p = path_probs[idx]
        filled = int(p * 20)
        bar = "█" * filled + "░" * (20 - filled)
        color = "green" if p > 0.4 else "yellow" if p > 0.25 else "red"
        marker = " ◄ ACTIVE" if idx == active_path else ""
        return f"[{color}]{bar}[/{color}] {p:.0%}{marker}"

    # Attacker position indicator
    atk_positions = {
        0: "INTERNET",
        1: "DMZ Web",
        2: "App Server",
        3: "DB Server",
        4: "DB Server [EXFIL]",
    }
    atk_pos = atk_positions.get(atk.stage, "Unknown")

    # Action display
    action_display = {0: "[dim]No Mutation[/dim]", 1: "[yellow]Moderate Shuffle[/yellow]", 2: "[red bold]AGGRESSIVE RANDOMIZE[/red bold]"}
    action_text = action_display.get(action, "")

    # Build topology ASCII art
    stage_color = STAGE_COLORS[min(atk.stage, 4)]
    atk_icon = f"[{stage_color}]◆ ATTACKER[/{stage_color}]"

    # Show attacker at their current position
    web_marker = f" {atk_icon}" if atk.stage == 1 else ""
    app_marker = f" {atk_icon}" if atk.stage == 2 else ""
    db_marker  = f" {atk_icon}" if atk.stage >= 3 else ""
    ext_marker = f" {atk_icon}" if atk.stage == 0 else ""

    topo = f"""
  ┌─────────────────────────────────────────────────────────────┐
  │                 MTD-Playground Topology                     │
  │                                                             │
  │   [cyan]Client[/cyan]              ┌──[bold]S0[/bold]──┐              [cyan]DMZ Web[/cyan]{web_marker}
  │   10.0.0.10 ─────────┤      ├───────── 10.0.0.30
  │                       └──┬───┘                              │
  │          Path A: {path_bar(0)}       │
  │                          │                                  │
  │                       ┌──┴───┐                              │
  │   [magenta]Attacker[/magenta]{ext_marker}  ┤  [bold]S1[/bold]  ├──── [cyan]Internal App[/cyan]{app_marker}
  │   10.0.0.20 ─────────┤      │     10.0.0.100
  │                       └──┬───┘                              │
  │          Path B: {path_bar(1)}       │
  │                          │                                  │
  │                       ┌──┴───┐                              │
  │                       │  [bold]S2[/bold]  ├──── [cyan]DB Server[/cyan]{db_marker}
  │                       │      │     10.0.0.40
  │                       └──────┘                              │
  │          Path C: {path_bar(2)}       │
  │                                                             │
  │   Step: {step:<4d}  Defender Action: {action_text:<40s}│
  └─────────────────────────────────────────────────────────────┘"""
    return topo


def build_metrics_panel(env, info, step, strategy_name):
    """Build live metrics display."""
    atk = env.attacker
    net = env.network

    # Attack stage with progress bar
    stage_idx = min(atk.stage, 4)
    stage_text = f"[{STAGE_COLORS[stage_idx]}]{STAGE_NAMES[stage_idx]}[/{STAGE_COLORS[stage_idx]}]"

    # Knowledge bar
    k = atk.knowledge
    k_filled = int(k * 30)
    k_bar = "█" * k_filled + "░" * (30 - k_filled)
    k_color = "red" if k > 0.6 else "yellow" if k > 0.3 else "green"

    # Entropy bar
    h = info.get("path_entropy", 0.5)
    h_filled = int(h * 30)
    h_bar = "█" * h_filled + "░" * (30 - h_filled)
    h_color = "green" if h > 0.7 else "yellow" if h > 0.4 else "red"

    metrics = f"""
  [bold underline]{strategy_name}[/bold underline]

  [bold]ATTACKER[/bold]
  Stage:     {stage_text}
  Knowledge: [{k_color}]{k_bar}[/{k_color}] {k:.0%}
  Recon Acc: {info.get('recon_accuracy', 0):.1%}

  [bold]DEFENDER[/bold]
  Path Entropy (H_P):     [{h_color}]{h_bar}[/{h_color}] {h:.3f}
  Attacker Entropy (H_A): {info.get('attacker_entropy', 0):.3f}
  Mutations Applied:      {info.get('mutation_count', 0)}

  [bold]NETWORK[/bold]
  Latency:      {info.get('latency_ms', 0):>7.1f} ms
  Availability: {info.get('service_availability', 1.0):>7.1%}

  [bold]REWARD[/bold]
  Step Reward:  {info.get('cumulative_reward', 0) - sum(env.episode_rewards[:-1]) if len(env.episode_rewards) > 1 else info.get('cumulative_reward', 0):>+7.3f}
  Cumulative:   {info.get('cumulative_reward', 0):>+7.2f}"""
    return metrics


def build_kill_chain_display(stage):
    """Visual kill chain progress."""
    stages = ["Recon", "Access", "Lateral", "PrivEsc", "Exfil"]
    parts = []
    for i, s in enumerate(stages):
        if i < stage:
            parts.append(f"[white on red] {s} [/white on red]")
        elif i == stage:
            parts.append(f"[black on yellow]>{s}<[/black on yellow]")
        else:
            parts.append(f"[dim] {s} [/dim]")
    return " → ".join(parts)


def get_strategy_fn(name, model_dir):
    """Load strategy function by name."""
    if name == "none":
        return lambda obs, step: 0, "No MTD (Baseline)"
    elif name == "random":
        return lambda obs, step: np.random.randint(0, 3), "Random MTD"
    elif name == "periodic":
        return lambda obs, step: 2 if step % 10 == 0 else 0, "Periodic MTD (every 10 steps)"
    elif name == "dqn":
        try:
            from stable_baselines3 import DQN
            model_path = os.path.join(model_dir, "best_model.zip")
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, "dqn_mtd_final.zip")
            if os.path.exists(model_path):
                env_tmp = MTDPlaygroundEnv(seed=0)
                model = DQN.load(model_path, env=env_tmp)
                def dqn_strategy(obs, step):
                    action, _ = model.predict(obs, deterministic=True)
                    return int(action)
                return dqn_strategy, "DQN Adaptive MTD (Eghtesad et al. 2020)"
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load DQN model: {e}[/yellow]")
        return lambda obs, step: 0, "DQN (FAILED TO LOAD -- falling back to No MTD)"
    else:
        return lambda obs, step: 0, f"Unknown strategy: {name}"


def run_simulation(strategy_name, strategy_fn, display_name, seed=42, speed=0.2, max_steps=200):
    """Run a single simulation with live display."""
    env = MTDPlaygroundEnv(seed=seed)
    obs, _ = env.reset()
    info = {
        "path_entropy": env.network.path_entropy(),
        "attacker_entropy": 1.0,
        "recon_accuracy": 0.0,
        "latency_ms": env.network.latency_ms,
        "service_availability": 1.0,
        "mutation_count": 0,
        "cumulative_reward": 0.0,
    }

    with Live(console=console, refresh_per_second=10, screen=True) as live:
        for step in range(max_steps):
            action = strategy_fn(obs, step)
            obs, reward, terminated, truncated, info = env.step(action)

            # Build display
            topo = build_topology_display(env, action, step)
            metrics = build_metrics_panel(env, info, step, display_name)
            kill_chain = build_kill_chain_display(env.attacker.stage)

            # Compose full display
            display = Text.from_markup(
                f"\n  Kill Chain: {kill_chain}\n"
                f"{topo}\n"
                f"{metrics}\n\n"
                f"  [dim]Press Ctrl+C to stop | Speed: {speed}s/step[/dim]"
            )

            live.update(Panel(display, title="[bold]MTD-Playground Live Simulation[/bold]",
                            border_style="blue", padding=(0, 1)))

            if terminated:
                # Flash red on exfiltration
                for _ in range(3):
                    alert = Text.from_markup(
                        f"\n  Kill Chain: {kill_chain}\n"
                        f"{topo}\n"
                        f"{metrics}\n\n"
                        f"  [bold white on red] !! ATTACK SUCCEEDED — DATA EXFILTRATED at step {step+1} !! [/bold white on red]"
                    )
                    live.update(Panel(alert, title="[bold red]BREACH DETECTED[/bold red]",
                                    border_style="red", padding=(0, 1)))
                    time.sleep(0.5)
                    live.update(Panel(display, title="[bold]MTD-Playground Live Simulation[/bold]",
                                    border_style="blue", padding=(0, 1)))
                    time.sleep(0.3)
                break

            if truncated:
                success_display = Text.from_markup(
                    f"\n  Kill Chain: {kill_chain}\n"
                    f"{topo}\n"
                    f"{metrics}\n\n"
                    f"  [bold white on green] DEFENSE HELD — Attacker failed to exfiltrate in {max_steps} steps [/bold white on green]"
                )
                live.update(Panel(success_display, title="[bold green]DEFENSE SUCCESSFUL[/bold green]",
                                border_style="green", padding=(0, 1)))
                time.sleep(3)
                break

            time.sleep(speed)

    # Print final summary
    console.print()
    result = "BREACHED" if env.attack_succeeded else "DEFENDED"
    color = "red" if env.attack_succeeded else "green"
    console.print(f"  [{color} bold]{result}[/{color} bold] | "
                  f"Steps: {step+1} | "
                  f"Mutations: {info.get('mutation_count', 0)} | "
                  f"Final Reward: {info.get('cumulative_reward', 0):.1f} | "
                  f"Latency: {info.get('latency_ms', 0):.1f}ms")
    return env.attack_succeeded, step + 1, info


def run_side_by_side(model_dir, seed=42):
    """Run all 4 strategies back to back for comparison."""
    strategies = [
        ("none", "No MTD"),
        ("periodic", "Periodic MTD"),
        ("dqn", "DQN Adaptive MTD"),
        ("random", "Random MTD"),
    ]

    results = []
    for name, label in strategies:
        fn, display = get_strategy_fn(name, model_dir)
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Running: {display}[/bold]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        time.sleep(1)
        breached, steps, info = run_simulation(name, fn, display, seed=seed, speed=0.15)
        results.append((display, breached, steps, info))
        time.sleep(2)

    # Final comparison table
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print("[bold]FINAL COMPARISON[/bold]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Strategy", style="cyan", width=30)
    table.add_column("Result", justify="center", width=12)
    table.add_column("Steps", justify="right", width=8)
    table.add_column("Mutations", justify="right", width=10)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Reward", justify="right", width=10)

    for display, breached, steps, info in results:
        result = "[red]BREACHED[/red]" if breached else "[green]DEFENDED[/green]"
        table.add_row(
            display,
            result,
            str(steps),
            str(info.get("mutation_count", 0)),
            f"{info.get('latency_ms', 0):.1f}ms",
            f"{info.get('cumulative_reward', 0):.1f}",
        )

    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Live MTD-Playground Simulation")
    parser.add_argument("--strategy", type=str, default="dqn",
                        choices=["none", "random", "periodic", "dqn"],
                        help="Defense strategy to simulate")
    parser.add_argument("--speed", type=float, default=0.2,
                        help="Seconds per simulation step (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="../models")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Run all 4 strategies back to back")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    console.print(Panel(
        "[bold]MTD-Playground: Live Network Defense Simulation[/bold]\n\n"
        "Simulates a multi-stage cyber attack against an enterprise SDN network.\n"
        "The defender uses Moving Target Defense (path randomization) to disrupt\n"
        "attacker reconnaissance and lateral movement.\n\n"
        "[dim]Paper: MTD-Playground (2026 CCS)\n"
        "RL Method: Eghtesad et al. (2020) GameSec — Adversarial Deep RL for MTD\n"
        "Topology: 5 hosts, 3 switches, 3 forwarding paths[/dim]",
        title="[bold blue]MTD-Playground[/bold blue]",
        border_style="blue",
    ))
    time.sleep(2)

    if args.side_by_side:
        run_side_by_side(args.model_dir, seed=args.seed)
    else:
        fn, display = get_strategy_fn(args.strategy, args.model_dir)
        run_simulation(args.strategy, fn, display,
                      seed=args.seed, speed=args.speed, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
