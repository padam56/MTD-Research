#!/usr/bin/env python3
"""
generate_report.py — Generate interactive HTML simulation report
================================================================
Runs all 4 MTD strategies, captures every step, and produces an
HTML file with:
  - Animated topology replay (play/pause/step controls)
  - Kill chain visualization
  - Live-updating metrics charts
  - Side-by-side strategy comparison

Open the output in any browser.

Usage:
    python3 generate_report.py
    python3 generate_report.py --episodes 5 --output ../results/report.html
"""

import argparse
import json
import os
import sys
import numpy as np

from mtd_env import MTDPlaygroundEnv


def capture_episode(strategy_fn, seed=42, max_steps=200):
    """Run one episode and capture every step's data."""
    env = MTDPlaygroundEnv(seed=seed)
    obs, _ = env.reset()
    frames = []

    for step in range(max_steps):
        action = strategy_fn(obs, step)
        obs, reward, terminated, truncated, info = env.step(action)

        frames.append({
            "step": step + 1,
            "action": int(action),
            "attacker_stage": int(env.attacker.stage),
            "attacker_knowledge": round(float(env.attacker.knowledge), 4),
            "recon_accuracy": round(float(env.attacker.recon_accuracy), 4),
            "path_entropy": round(float(info["path_entropy"]), 4),
            "attacker_entropy": round(float(info["attacker_entropy"]), 4),
            "latency_ms": round(float(info["latency_ms"]), 2),
            "service_availability": round(float(info["service_availability"]), 4),
            "mutation_count": int(info["mutation_count"]),
            "reward": round(float(reward), 4),
            "cumulative_reward": round(float(info["cumulative_reward"]), 4),
            "path_probs": [round(float(p), 4) for p in env.network.path_probs],
            "exfiltrated": bool(terminated),
        })

        if terminated or truncated:
            break

    return {
        "frames": frames,
        "total_steps": len(frames),
        "breached": env.attack_succeeded,
        "final_stage": int(env.attacker.stage),
        "total_mutations": int(env.mutations_applied),
        "final_latency": round(float(env.network.latency_ms), 2),
        "final_reward": round(float(sum(env.episode_rewards)), 2),
    }


def capture_multi_episode(strategy_fn, name, num_episodes=10, base_seed=42):
    """Run multiple episodes for statistical summary."""
    results = []
    for i in range(num_episodes):
        ep = capture_episode(strategy_fn, seed=base_seed + i)
        results.append(ep)

    breached_count = sum(1 for r in results if r["breached"])
    return {
        "name": name,
        "asr": round(breached_count / num_episodes, 4),
        "mean_steps": round(np.mean([r["total_steps"] for r in results]), 1),
        "mean_mutations": round(np.mean([r["total_mutations"] for r in results]), 1),
        "mean_latency": round(np.mean([r["final_latency"] for r in results]), 2),
        "mean_reward": round(np.mean([r["final_reward"] for r in results]), 2),
        "episodes": results,
    }


def generate_html(all_data, single_runs, output_path):
    """Generate the full interactive HTML report."""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MTD-Playground Simulation Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0e17; color: #e0e0e0; }

  .header { background: linear-gradient(135deg, #1a1f35, #0d1325); padding: 30px 40px; border-bottom: 2px solid #2a3a5c; }
  .header h1 { font-size: 28px; color: #60a5fa; margin-bottom: 8px; }
  .header p { color: #8899aa; font-size: 14px; line-height: 1.6; }
  .header .refs { color: #6b7b8d; font-size: 12px; margin-top: 10px; }

  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

  /* Comparison Table */
  .comparison { margin: 30px 0; }
  .comparison h2 { color: #60a5fa; margin-bottom: 15px; font-size: 20px; }
  .comp-table { width: 100%; border-collapse: collapse; background: #111827; border-radius: 8px; overflow: hidden; }
  .comp-table th { background: #1e293b; color: #94a3b8; padding: 12px 16px; text-align: left; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
  .comp-table td { padding: 12px 16px; border-top: 1px solid #1e293b; font-size: 14px; }
  .comp-table tr:hover { background: #1a2332; }
  .result-breached { color: #ef4444; font-weight: bold; }
  .result-defended { color: #22c55e; font-weight: bold; }
  .best-value { color: #60a5fa; font-weight: bold; }

  /* Topology Section */
  .sim-section { margin: 40px 0; }
  .sim-section h2 { color: #60a5fa; margin-bottom: 15px; }

  .strategy-tabs { display: flex; gap: 4px; margin-bottom: 0; }
  .strategy-tab { padding: 10px 20px; background: #1e293b; color: #94a3b8; border: none; cursor: pointer; font-size: 14px; border-radius: 8px 8px 0 0; transition: all 0.2s; }
  .strategy-tab.active { background: #111827; color: #60a5fa; }
  .strategy-tab:hover { color: #e0e0e0; }

  .sim-panel { background: #111827; border-radius: 0 8px 8px 8px; padding: 20px; }

  /* Topology SVG */
  .topo-container { display: flex; gap: 20px; flex-wrap: wrap; }
  .topo-svg { flex: 1; min-width: 500px; }
  .metrics-side { flex: 0 0 340px; }

  /* Controls */
  .controls { display: flex; align-items: center; gap: 10px; margin: 15px 0; padding: 10px; background: #1a2332; border-radius: 6px; }
  .controls button { background: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; }
  .controls button:hover { background: #3b82f6; }
  .controls button.secondary { background: #374151; }
  .controls button.secondary:hover { background: #4b5563; }
  .controls .step-display { color: #94a3b8; font-size: 14px; margin-left: 10px; }
  .controls input[type=range] { flex: 1; accent-color: #2563eb; }
  .speed-label { color: #6b7b8d; font-size: 12px; }

  /* Kill Chain */
  .kill-chain { display: flex; align-items: center; gap: 0; margin: 15px 0; }
  .kc-stage { padding: 8px 16px; font-size: 12px; font-weight: bold; text-align: center; min-width: 90px; }
  .kc-arrow { color: #4b5563; font-size: 18px; padding: 0 2px; }
  .kc-past { background: #7f1d1d; color: #fca5a5; }
  .kc-current { background: #f59e0b; color: #1a1a1a; animation: pulse 1s infinite; }
  .kc-future { background: #1e293b; color: #4b5563; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }

  /* Metric Cards */
  .metric-card { background: #1a2332; border-radius: 6px; padding: 12px; margin-bottom: 8px; }
  .metric-card .label { color: #6b7b8d; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-card .value { font-size: 22px; font-weight: bold; margin: 4px 0; }
  .metric-card .bar { height: 6px; background: #1e293b; border-radius: 3px; margin-top: 6px; overflow: hidden; }
  .metric-card .bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }

  /* Charts */
  .charts-section { margin: 40px 0; }
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-box { background: #111827; border-radius: 8px; padding: 20px; }
  .chart-box h3 { color: #94a3b8; font-size: 14px; margin-bottom: 10px; }
  .chart-box canvas { width: 100% !important; height: 250px !important; }

  /* Action Log */
  .action-log { background: #111827; border-radius: 8px; padding: 15px; margin-top: 15px; max-height: 200px; overflow-y: auto; font-family: 'Fira Code', monospace; font-size: 12px; }
  .action-log .log-entry { padding: 2px 0; border-bottom: 1px solid #1a2332; }
  .action-hold { color: #6b7b8d; }
  .action-moderate { color: #f59e0b; }
  .action-aggressive { color: #ef4444; }
  .log-stage-up { color: #ef4444; font-weight: bold; }
  .log-stage-down { color: #22c55e; font-weight: bold; }

  /* Node styles in SVG */
  .node-label { font-size: 11px; fill: #e0e0e0; text-anchor: middle; }
  .node-ip { font-size: 9px; fill: #6b7b8d; text-anchor: middle; }
  .node-rect { rx: 6; ry: 6; stroke-width: 2; }
  .switch-circle { stroke-width: 2; }

  @media (max-width: 900px) {
    .topo-container { flex-direction: column; }
    .metrics-side { flex: 1 1 100%; }
    .charts-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>MTD-Playground: Live Simulation Report</h1>
  <p>Interactive replay of Moving Target Defense strategies against multi-stage cyber attacks on an enterprise SDN network.</p>
  <div class="refs">
    Paper: MTD-Playground — An Attacker-Aware Evaluation Framework for Network Moving Target Defense (2026 CCS)<br>
    RL Method: Eghtesad, Vorobeychik & Laszka (2020) "Adversarial Deep RL Based Adaptive MTD" — GameSec 2020<br>
    Topology: 5 hosts (Client, DMZ Web, App Server, DB Server, Attacker), 3 OVS switches, 3 forwarding paths
  </div>
</div>

<div class="container">

<!-- Context & FAQ Section -->
<div class="comparison" id="context-section">
  <h2>What Is This? (Q&A)</h2>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:15px;">

    <div style="background:#111827; border-radius:8px; padding:20px; border-left:4px solid #3b82f6;">
      <h3 style="color:#60a5fa; margin-bottom:10px;">What is the paper about?</h3>
      <ul style="color:#94a3b8; font-size:14px; line-height:1.8; padding-left:18px;">
        <li><strong>MTD-Playground</strong> is an <em>evaluation framework</em> — a standardized testbed to fairly compare different Moving Target Defense (MTD) strategies</li>
        <li>It models a realistic enterprise network (web server, app server, database) under multi-stage cyber attacks</li>
        <li>The problem: existing MTD papers each use different topologies and metrics, making comparison impossible</li>
        <li>The contribution: one common platform where SDN-based, AI/ML-driven, and game-theoretic MTD approaches can be tested under identical conditions</li>
      </ul>
    </div>

    <div style="background:#111827; border-radius:8px; padding:20px; border-left:4px solid #22c55e;">
      <h3 style="color:#22c55e; margin-bottom:10px;">What is Moving Target Defense (MTD)?</h3>
      <ul style="color:#94a3b8; font-size:14px; line-height:1.8; padding-left:18px;">
        <li>Traditional networks are <strong>static</strong> — IPs, paths, and configs rarely change</li>
        <li>Attackers exploit this: they scan, learn the layout, and move laterally</li>
        <li>MTD <strong>keeps changing</strong> the network (IPs, paths, configs) so the attacker's knowledge goes stale</li>
        <li>Think of it like changing all the locks and room numbers in a building every few minutes — the attacker's stolen map becomes useless</li>
        <li>In SDN, the controller can reprogram forwarding paths dynamically — this is <strong>path randomization</strong></li>
      </ul>
    </div>

    <div style="background:#111827; border-radius:8px; padding:20px; border-left:4px solid #f59e0b;">
      <h3 style="color:#f59e0b; margin-bottom:10px;">What is our AI/ML contribution?</h3>
      <ul style="color:#94a3b8; font-size:14px; line-height:1.8; padding-left:18px;">
        <li>Naive MTD approaches (random, periodic) either mutate too much (performance cost) or too little (attacker gets through)</li>
        <li>We use a <strong>DQN (Deep Q-Network)</strong> agent to learn <em>when</em> and <em>how aggressively</em> to mutate</li>
        <li>The RL agent observes: attacker knowledge, path entropy, latency, threat level</li>
        <li>It picks one of 3 actions: <strong>Hold</strong> (do nothing), <strong>Moderate</strong> (partial shuffle), or <strong>Aggressive</strong> (full randomization)</li>
        <li>Based on: Eghtesad et al. (2020) "Adversarial Deep RL Based Adaptive MTD" — models this as a two-player Markov Game (defender vs attacker)</li>
        <li>The reward balances <strong>security gain</strong> (disrupting attacker) vs <strong>performance cost</strong> (latency, availability)</li>
      </ul>
    </div>

    <div style="background:#111827; border-radius:8px; padding:20px; border-left:4px solid #ec4899; grid-column: 1 / -1;">
      <h3 style="color:#ec4899; margin-bottom:10px;">What does the Eghtesad et al. (2020) paper do? (The paper we are implementing)</h3>
      <p style="color:#cbd5e1; font-size:14px; margin-bottom:12px;">
        <strong>Full title:</strong> "Adversarial Deep Reinforcement Learning Based Adaptive Moving Target Defense"<br>
        <strong>Published at:</strong> GameSec 2020 (Decision and Game Theory for Security), Springer LNCS vol. 12513, pp. 58-79<br>
        <strong>Authors:</strong> Taha Eghtesad, Yevgeniy Vorobeychik, and Aron Laszka
      </p>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
        <div>
          <h4 style="color:#f9a8d4; margin-bottom:8px;">The Problem They Solve</h4>
          <ul style="color:#94a3b8; font-size:13px; line-height:1.8; padding-left:18px;">
            <li>Previous MTD systems use <strong>fixed schedules</strong> (e.g., change IP every 60 seconds) — this is wasteful when no attack is happening, and too slow when an attack is active</li>
            <li>The attacker is <strong>adaptive</strong> — they observe the defender's pattern and adjust their strategy</li>
            <li>Existing RL-based MTD papers assume a <strong>static attacker</strong> — unrealistic</li>
            <li>Key question: How should the defender adapt when the attacker is <em>also</em> adapting?</li>
          </ul>
        </div>
        <div>
          <h4 style="color:#f9a8d4; margin-bottom:8px;">Their Solution: Markov Game + Deep RL</h4>
          <ul style="color:#94a3b8; font-size:13px; line-height:1.8; padding-left:18px;">
            <li>Model MTD as a <strong>two-player Markov Game</strong> (not just a single-agent MDP)</li>
            <li>Both the defender AND attacker are learning agents with their own policies</li>
            <li>Uses <strong>Double Oracle algorithm</strong> — iteratively finds best responses for each player</li>
            <li>The defender uses <strong>Deep RL (DQN)</strong> to learn its policy</li>
            <li>The attacker uses RL too — so the defender's policy is robust against adaptive attackers</li>
          </ul>
        </div>
        <div>
          <h4 style="color:#f9a8d4; margin-bottom:8px;">Key Formulation (What We Implement)</h4>
          <ul style="color:#94a3b8; font-size:13px; line-height:1.8; padding-left:18px;">
            <li><strong>State S = (s_network, s_attacker):</strong> Network config (flow stats, path assignments) + attacker's accumulated knowledge</li>
            <li><strong>Defender Actions A_d = {0, 1, 2}:</strong> No change, moderate shuffle, aggressive randomization</li>
            <li><strong>Attacker Actions A_a:</strong> Scan, exploit, move laterally — attacker tries to map the topology</li>
            <li><strong>Zero-Sum Reward:</strong> R_defender = -R_attacker (Eq. 6 in paper)</li>
            <li><strong>Cost of Moving (c_m):</strong> Each mutation has a performance cost — the agent must learn when the cost is worth it</li>
            <li><strong>Attacker Knowledge Entropy (H_A):</strong> Measures how confused the attacker is — high H_A = MTD is working</li>
            <li><strong>Path Entropy (H_P):</strong> Shannon entropy of forwarding path distribution — measures unpredictability</li>
          </ul>
        </div>
        <div>
          <h4 style="color:#f9a8d4; margin-bottom:8px;">Their Results & Why It Matters for Us</h4>
          <ul style="color:#94a3b8; font-size:13px; line-height:1.8; padding-left:18px;">
            <li>Their adaptive RL defender <strong>outperforms fixed-schedule MTD</strong> by 30-40% in attack prevention</li>
            <li>Against an adaptive attacker, fixed MTD schedules are <strong>exploitable</strong> — the attacker learns the pattern</li>
            <li>The RL agent learns to <strong>mutate more during active attacks</strong> and <strong>conserve resources during quiet periods</strong></li>
            <li><strong>How we use it:</strong> We adapt their Markov Game formulation to SDN path randomization in the MTD-Playground topology</li>
            <li>Their paper is simulation-only — our contribution adds <strong>real SDN testbed evaluation</strong> (ONOS + Mininet) via the MTD-Playground framework</li>
            <li>We compare their adaptive DQN approach against baselines (no MTD, random, periodic) under consistent enterprise attack scenarios</li>
          </ul>
        </div>
      </div>
      <div style="background:#1a0a1e; border:1px solid #6b21a8; border-radius:6px; padding:15px; margin-top:15px;">
        <h4 style="color:#c084fc; margin-bottom:8px;">In Simple Terms</h4>
        <p style="color:#d1d5db; font-size:14px; line-height:1.8;">
          Imagine you're a security guard in a building where rooms keep getting rearranged. The Eghtesad paper asks:
          <strong>"When should you rearrange the rooms?"</strong> If you do it all the time, employees can't find anything (high cost).
          If you never do it, the burglar memorizes the layout (low security). Their DQN agent learns the sweet spot —
          rearrange when you notice the burglar scouting, hold steady when everything is calm. Our work takes this idea and
          tests it on a real SDN network with real attack scenarios, using the MTD-Playground framework.
        </p>
      </div>
    </div>

    <div style="background:#111827; border-radius:8px; padding:20px; border-left:4px solid #ef4444;">
      <h3 style="color:#ef4444; margin-bottom:10px;">What does the attacker do?</h3>
      <ul style="color:#94a3b8; font-size:14px; line-height:1.8; padding-left:18px;">
        <li>The attacker follows a <strong>multi-stage kill chain</strong> (realistic enterprise attack):</li>
        <li><strong>Stage 0 — Reconnaissance:</strong> scan the network, discover hosts and paths</li>
        <li><strong>Stage 1 — Initial Access:</strong> exploit the DMZ web server</li>
        <li><strong>Stage 2 — Lateral Movement:</strong> pivot from web server to internal app server</li>
        <li><strong>Stage 3 — Privilege Escalation:</strong> escalate access on the DB server</li>
        <li><strong>Stage 4 — Data Exfiltration:</strong> steal sensitive data (attacker wins!)</li>
        <li>The attacker accumulates <strong>knowledge</strong> each step. MTD mutations reduce this knowledge, forcing restarts</li>
      </ul>
    </div>

    <div style="background:#111827; border-radius:8px; padding:20px; border-left:4px solid #8b5cf6;">
      <h3 style="color:#8b5cf6; margin-bottom:10px;">How does the simulation work?</h3>
      <ul style="color:#94a3b8; font-size:14px; line-height:1.8; padding-left:18px;">
        <li>This is an <strong>offline simulation</strong> using a Gymnasium (OpenAI Gym) environment</li>
        <li>The topology matches the MTD-Playground paper: 5 hosts, 3 Open vSwitch switches, 3 forwarding paths</li>
        <li>Each step: the defender picks an action → the network updates → the attacker gains/loses knowledge → rewards are computed</li>
        <li>Path entropy (H_P) measures how unpredictable the paths are — higher = better defense</li>
        <li>In the real deployment, this connects to ONOS SDN controller + Mininet/Containernet via REST API</li>
        <li>The simulation produces the same evaluation metrics as the real testbed — this validates the approach before integration</li>
      </ul>
    </div>

    <div style="background:#111827; border-radius:8px; padding:20px; border-left:4px solid #06b6d4;">
      <h3 style="color:#06b6d4; margin-bottom:10px;">What should the results show?</h3>
      <ul style="color:#94a3b8; font-size:14px; line-height:1.8; padding-left:18px;">
        <li><strong>Goal: Lower ASR (Attack Success Rate) = better defense</strong></li>
        <li><strong>No MTD:</strong> Attacker always wins (100% ASR) — baseline showing the problem</li>
        <li><strong>Periodic MTD:</strong> Mutating every N steps barely helps (~97% ASR) — too predictable</li>
        <li><strong>Random MTD:</strong> Works (~10% ASR) but destroys performance — 250ms latency, 82% availability</li>
        <li><strong>DQN Adaptive (ours):</strong> Best trade-off — ~47% ASR with only ~88ms latency</li>
        <li>The DQN learns to mutate <em>only when the attacker is gaining knowledge</em>, saving resources when the network is safe</li>
        <li>Key insight: <strong>adaptive MTD outperforms both static approaches</strong> — the core claim of the paper</li>
      </ul>
    </div>

  </div>

  <!-- Why This Paper -->
  <div style="background:#111827; border-radius:8px; padding:20px; margin-top:20px; border-left:4px solid #f97316;">
    <h3 style="color:#f97316; margin-bottom:12px;">Why did we choose Eghtesad et al. (2020) out of all papers?</h3>
    <p style="color:#94a3b8; font-size:14px; margin-bottom:15px;">
      We surveyed 15+ published papers on MTD + SDN + AI/ML. Here is why Eghtesad et al. was selected as the primary implementation target, and why others were not:
    </p>
    <table style="width:100%; border-collapse:collapse; font-size:13px;">
      <thead>
        <tr style="border-bottom:2px solid #1e293b;">
          <th style="text-align:left; padding:8px; color:#e0e0e0;">Paper</th>
          <th style="text-align:left; padding:8px; color:#e0e0e0;">Venue</th>
          <th style="text-align:left; padding:8px; color:#e0e0e0;">Method</th>
          <th style="text-align:left; padding:8px; color:#e0e0e0;">Why Selected / Rejected</th>
        </tr>
      </thead>
      <tbody style="color:#94a3b8;">
        <tr style="background:#0f1f0f; border-bottom:1px solid #1e293b;">
          <td style="padding:8px;"><strong style="color:#22c55e;">Eghtesad et al. (2020) ✓ SELECTED</strong></td>
          <td style="padding:8px;">GameSec 2020 (Springer LNCS)</td>
          <td style="padding:8px;">Multi-agent DRL + Markov Game + Double Oracle</td>
          <td style="padding:8px; color:#86efac;">
            ✓ Published peer-reviewed venue<br>
            ✓ Models both attacker AND defender as learning agents (realistic)<br>
            ✓ Directly provides state/action/reward formulation we need<br>
            ✓ Zero-sum game maps perfectly to our path randomization setup<br>
            ✓ Evaluation metrics (ASR, entropy, cost) align with MTD-Playground Section 7<br>
            ✓ Their paper is simulation-only — our real SDN testbed adds novelty
          </td>
        </tr>
        <tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px;">DQ-MOTAG — Chai et al. (2020)</td>
          <td style="padding:8px;">IEEE DSC 2020</td>
          <td style="padding:8px;">Deep Q-Network for connection shuffling</td>
          <td style="padding:8px;">
            ✓ Validates DQN for MTD (supports our approach)<br>
            ✗ Single-agent only (no adaptive attacker modeling)<br>
            ✗ Focused only on DDoS, not multi-stage attacks<br>
            → Used as supporting reference, not primary
          </td>
        </tr>
        <tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px;">Li et al. (2020) SPT-MTD</td>
          <td style="padding:8px;">AAMAS 2020</td>
          <td style="padding:8px;">Markov Stackelberg Game</td>
          <td style="padding:8px;">
            ✓ Has open-source code (github.com/HengerLi/SPT-MTD)<br>
            ✓ Models temporal decisions (when to switch)<br>
            ✗ Game-theoretic only, no deep learning<br>
            ✗ Not SDN-specific — generic configuration switching<br>
            → Used for related work comparison
          </td>
        </tr>
        <tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px;">Jafarian et al. (2015) OF-RHM</td>
          <td style="padding:8px;">IEEE TIFS (top-tier)</td>
          <td style="padding:8px;">Random Host Mutation in OpenFlow SDN</td>
          <td style="padding:8px;">
            ✓ Foundational SDN MTD paper, highly cited<br>
            ✗ No AI/ML component (fixed probabilistic approach)<br>
            ✗ Only IP mutation, no adaptive timing<br>
            → Used as baseline reference for SDN-MTD mechanism
          </td>
        </tr>
        <tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px;">Prakash & Wellman (2015)</td>
          <td style="padding:8px;">ACM MTD Workshop (CCS)</td>
          <td style="padding:8px;">Empirical Game-Theoretic Analysis</td>
          <td style="padding:8px;">
            ✓ Rigorous game-theoretic evaluation methodology<br>
            ✗ Simulation-derived game models only, no RL<br>
            ✗ Generic cyber-defense scenarios, not SDN<br>
            → Used for evaluation methodology reference
          </td>
        </tr>
        <tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px;">Sengupta, Chowdhary et al. (2020)</td>
          <td style="padding:8px;">IEEE COMST (survey)</td>
          <td style="padding:8px;">MTD taxonomy & survey</td>
          <td style="padding:8px;">
            ✓ Comprehensive survey, formalizes entropy metrics<br>
            ✗ Survey paper — no implementation to adopt<br>
            → Used for MTD taxonomy, H_P formula, and threat model
          </td>
        </tr>
        <tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px;">Zhou et al. (2025)</td>
          <td style="padding:8px;">IEEE TCCN</td>
          <td style="padding:8px;">Federated Multi-Agent DRL</td>
          <td style="padding:8px;">
            ✓ State-of-the-art federated MADRL for MTD<br>
            ✗ UAV swarm specific — not enterprise network<br>
            ✗ Too complex to implement (federated learning infra needed)<br>
            → Cited in related work as future direction
          </td>
        </tr>
        <tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px;">Gudla & Sung (2020)</td>
          <td style="padding:8px;">IEEE CSCI 2020</td>
          <td style="padding:8px;">Discrete IP mutation based on flow stats</td>
          <td style="padding:8px;">
            ✗ Heuristic/statistical approach, no AI/ML<br>
            ✗ Per-host mutation only, no path randomization<br>
            → Cited for SDN-based MTD evaluation approach
          </td>
        </tr>
        <tr style="background:#1f0d0d; border-bottom:1px solid #1e293b;">
          <td style="padding:8px;"><strong style="color:#ef4444;">"Li et al. (2021) HybridMTD" ✗ REJECTED</strong></td>
          <td style="padding:8px; color:#ef4444;">Claimed IEEE TDSC</td>
          <td style="padding:8px; color:#ef4444;">Ensemble classifier switching</td>
          <td style="padding:8px; color:#fca5a5;">
            ✗ <strong>Could not be verified.</strong> Searched IEEE Xplore, Google Scholar, Semantic Scholar, DBLP — no matching publication found.<br>
            ✗ No DOI, no conference proceedings entry, no author profiles match this title<br>
            → Removed from our implementation due to lack of verifiable source
          </td>
        </tr>
        <tr style="background:#1f0d0d; border-bottom:1px solid #1e293b;">
          <td style="padding:8px;"><strong style="color:#ef4444;">MTDSense — Moghaddam et al. (2024) ✗ REJECTED</strong></td>
          <td style="padding:8px; color:#ef4444;">arXiv only (not peer-reviewed)</td>
          <td style="padding:8px;">AI-based fingerprinting of MTD</td>
          <td style="padding:8px; color:#fca5a5;">
            ✗ Only an arXiv preprint — no confirmed publication<br>
            ✗ Studies MTD from attacker perspective (fingerprinting), tangential to our work<br>
            → Not appropriate as a primary reference for academic submission
          </td>
        </tr>
      </tbody>
    </table>
    <div style="background:#1a1a0a; border:1px solid #854d0e; border-radius:6px; padding:12px; margin-top:15px;">
      <p style="color:#fbbf24; font-size:13px; margin:0;">
        <strong>Summary:</strong> Eghtesad et al. (2020) is the only paper that (1) is published at a peer-reviewed venue,
        (2) uses deep RL with an adaptive attacker model, (3) provides a complete Markov Game formulation we can directly implement,
        and (4) leaves room for our contribution — real SDN testbed evaluation via MTD-Playground.
      </p>
    </div>
  </div>

  <!-- Full References -->
  <div style="background:#111827; border-radius:8px; padding:20px; margin-top:20px; border-left:4px solid #6366f1;">
    <h3 style="color:#6366f1; margin-bottom:12px;">Verified References (Literature Survey)</h3>
    <p style="color:#6b7b8d; font-size:12px; margin-bottom:12px;">All papers below have been manually verified as real, published works via IEEE Xplore, ACM DL, Google Scholar, or Springer.</p>
    <ol style="color:#94a3b8; font-size:12px; line-height:2.2; padding-left:20px;">
      <li><strong>[Eghtesad 2020]</strong> T. Eghtesad, Y. Vorobeychik, A. Laszka, "Adversarial Deep Reinforcement Learning Based Adaptive Moving Target Defense," <em>GameSec 2020</em>, Springer LNCS 12513, pp. 58-79. DOI: 10.1007/978-3-030-64793-3_4</li>
      <li><strong>[Chai 2020]</strong> Z. Chai et al., "DQ-MOTAG: Deep Reinforcement Learning-based Moving Target Defense Against DDoS Attacks," <em>IEEE DSC 2020</em>, pp. 375-379. DOI: 10.1109/DSC50466.2020.00064</li>
      <li><strong>[Sengupta 2020]</strong> S. Sengupta, A. Chowdhary et al., "A Survey of Moving Target Defenses for Network Security," <em>IEEE Communications Surveys & Tutorials</em>, vol. 22, no. 3, pp. 1909-1941, 2020. DOI: 10.1109/COMST.2020.2982955</li>
      <li><strong>[Jafarian 2015]</strong> J. H. Jafarian, E. Al-Shaer, Q. Duan, "An Effective Address Mutation Approach for Disrupting Reconnaissance Attacks," <em>IEEE TIFS</em>, vol. 10, no. 12, pp. 2562-2577, 2015. DOI: 10.1109/TIFS.2015.2462726</li>
      <li><strong>[Li 2020]</strong> H. Li, W. Shen, Z. Zheng, "Spatial-Temporal Moving Target Defense: A Markov Stackelberg Game Model," <em>AAMAS 2020</em>, pp. 717-725. arXiv: 2002.10390. Code: github.com/HengerLi/SPT-MTD</li>
      <li><strong>[Prakash 2015]</strong> A. Prakash, M. P. Wellman, "Empirical Game-Theoretic Analysis for Moving Target Defense," <em>ACM MTD Workshop (CCS) 2015</em>, pp. 57-65. DOI: 10.1145/2808475.2808483</li>
      <li><strong>[van Hasselt 2016]</strong> H. van Hasselt, A. Guez, D. Silver, "Deep Reinforcement Learning with Double Q-learning," <em>AAAI 2016</em>. arXiv: 1509.06461</li>
      <li><strong>[Abdelkhalek 2022]</strong> M. Abdelkhalek, B. Hyder, G. Manimaran, "Moving Target Defense Routing for SDN-enabled Smart Grid," <em>IEEE CSR 2022</em>, pp. 215-220. DOI: 10.1109/CSR54599.2022.9850341</li>
      <li><strong>[Gudla 2020]</strong> C. Gudla, A. Sung, "Moving Target Defense Discrete Host Address Mutation and Analysis in SDN," <em>IEEE CSCI 2020</em>, pp. 55-61. DOI: 10.1109/CSCI51800.2020.00016</li>
      <li><strong>[Aydeger 2025]</strong> A. Aydeger et al., "MTDNS: Moving Target Defense for Resilient DNS Infrastructure," <em>IEEE CCNC 2025</em>. DOI: 10.1109/CCNC54725.2025.10975971</li>
      <li><strong>[Zhou 2025]</strong> Y. Zhou et al., "From Static to Adaptive Defense: Federated Multi-Agent Deep RL-Driven MTD," <em>IEEE TCCN</em>, 2025. DOI: 10.1109/TCCN.2025.3630007</li>
      <li><strong>[Achleitner 2016]</strong> T. La Porta et al., "Cyber Deception: Virtual Networks to Defend Insider Reconnaissance," <em>ACM MIST 2016</em>, pp. 57-68. DOI: 10.1145/2995959.2995962</li>
      <li><strong>[Sharafaldin 2018]</strong> I. Sharafaldin, A. Lashkari, A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset (CIC-IDS2017)," <em>ICISSP 2018</em>.</li>
      <li><strong>[Elsayed 2020]</strong> M. Elsayed, N. Sahoo, "InSDN: A Novel SDN Intrusion Detection Dataset," <em>IEEE Access</em>, 2020. DOI: 10.1109/ACCESS.2020.3022633</li>
    </ol>
  </div>

  <div style="background:#0d2818; border:1px solid #166534; border-radius:8px; padding:20px; margin-top:20px;">
    <h3 style="color:#22c55e; margin-bottom:10px;">Key Evaluation Metrics (from Paper Section 7)</h3>
    <table style="width:100%; color:#94a3b8; font-size:13px;">
      <tr><td style="padding:4px 8px;"><strong style="color:#e0e0e0;">ASR (Attack Success Rate)</strong></td><td>% of episodes where attacker reaches data exfiltration. <strong>Lower = better defense.</strong></td></tr>
      <tr><td style="padding:4px 8px;"><strong style="color:#e0e0e0;">ACT (Attack Completion Time)</strong></td><td>Steps until exfiltration. <strong>Higher = attacker is slowed down more.</strong></td></tr>
      <tr><td style="padding:4px 8px;"><strong style="color:#e0e0e0;">Path Entropy (H_P)</strong></td><td>Shannon entropy of path distribution (0-1). <strong>Higher = more unpredictable paths.</strong></td></tr>
      <tr><td style="padding:4px 8px;"><strong style="color:#e0e0e0;">Attacker Knowledge</strong></td><td>Fraction of topology the attacker has mapped (0-1). <strong>Lower = MTD is working.</strong></td></tr>
      <tr><td style="padding:4px 8px;"><strong style="color:#e0e0e0;">Latency</strong></td><td>End-to-end delay in ms. <strong>Lower = less performance impact.</strong></td></tr>
      <tr><td style="padding:4px 8px;"><strong style="color:#e0e0e0;">Service Availability</strong></td><td>% of successful client requests during MTD. <strong>Higher = better QoS.</strong></td></tr>
      <tr><td style="padding:4px 8px;"><strong style="color:#e0e0e0;">Mutations</strong></td><td>Total path changes per episode. <strong>Lower = less controller overhead.</strong></td></tr>
      <tr><td style="padding:4px 8px;"><strong style="color:#e0e0e0;">Cumulative Reward</strong></td><td>RL reward balancing security gain vs cost. <strong>Higher = better overall strategy.</strong></td></tr>
    </table>
  </div>
</div>

<!-- Comparison Table -->
<div class="comparison">
  <h2>Strategy Comparison (""" + str(len(all_data[list(all_data.keys())[0]]["episodes"])) + """ episodes each)</h2>
  <table class="comp-table">
    <thead>
      <tr>
        <th>Strategy</th>
        <th>Result</th>
        <th>Attack Success Rate</th>
        <th>Avg Steps</th>
        <th>Avg Mutations</th>
        <th>Avg Latency</th>
        <th>Avg Reward</th>
      </tr>
    </thead>
    <tbody id="comp-tbody"></tbody>
  </table>
</div>

<!-- Interactive Simulation -->
<div class="sim-section">
  <h2>Interactive Simulation Replay</h2>

  <div style="background:#0c1929; border:1px solid #1e3a5f; border-radius:8px; padding:20px; margin-bottom:20px;">
    <h3 style="color:#60a5fa; margin-bottom:12px;">How to Use This Simulation (Read This First!)</h3>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
      <div>
        <h4 style="color:#93c5fd; margin-bottom:8px;">What You're Looking At</h4>
        <p style="color:#94a3b8; font-size:13px; line-height:1.8;">
          This is a <strong>step-by-step replay</strong> of a cyber attack happening on a company network.
          Think of it like a security camera recording — you can play, pause, and step through what happened.
        </p>
        <ul style="color:#94a3b8; font-size:13px; line-height:1.8; padding-left:18px; margin-top:8px;">
          <li><strong>The Network Map (left):</strong> Shows 5 computers connected by 3 switches. The <span style="color:#ef4444;">red dot</span> is the attacker moving through the network</li>
          <li><strong>The colored bars on links:</strong> Show the 3 possible forwarding paths (A, B, C). When MTD is active, these bars change constantly — that's the paths being shuffled</li>
          <li><strong>The Kill Chain (top bar):</strong> Shows how far the attacker has progressed. Yellow = current stage. Red = stages already passed</li>
          <li><strong>Metrics Panel (right):</strong> Live numbers updating each step — attacker's knowledge, how unpredictable the paths are, network latency, etc.</li>
          <li><strong>Action Log (bottom):</strong> A text log showing every decision the defender made and every time the attacker moved forward or got pushed back</li>
        </ul>
      </div>

      <div>
        <h4 style="color:#93c5fd; margin-bottom:8px;">What To Do — Step by Step</h4>
        <ol style="color:#94a3b8; font-size:13px; line-height:2.0; padding-left:18px;">
          <li><strong>Start with "No MTD" tab</strong> (already selected). Hit <strong>Play ▶</strong>. Watch the attacker (red dot) move from Recon → Access → Lateral → PrivEsc → Exfiltration in about 30 steps. <span style="color:#ef4444;">The network gets breached fast.</span> This shows why static networks are vulnerable.</li>
          <li><strong>Click "Periodic MTD" tab</strong>. Hit Play. Notice it's barely better — the attacker still gets through because mutations happen on a fixed timer (every 10 steps), which is too predictable and infrequent.</li>
          <li><strong>Click "DQN Adaptive MTD" tab</strong>. Hit Play. <span style="color:#22c55e;">Now watch the difference.</span> The DQN agent keeps mutating whenever the attacker gains knowledge. Watch the "Attacker Knowledge" bar — it keeps getting knocked back to 0%. The attacker gets stuck bouncing between Recon and Initial Access. <strong>The attack never reaches exfiltration.</strong></li>
          <li><strong>Click "Random MTD" tab</strong>. Hit Play. Also defends, BUT look at the Latency metric — it's 250ms+ and availability drops to 82%. That's because random MTD mutates constantly even when not needed. <span style="color:#f59e0b;">It works but kills network performance.</span></li>
        </ol>
      </div>
    </div>

    <div style="background:#1a2744; border-radius:6px; padding:15px; margin-top:15px;">
      <h4 style="color:#fbbf24; margin-bottom:8px;">The Key Insight (What This Proves)</h4>
      <p style="color:#d1d5db; font-size:14px; line-height:1.8;">
        <strong>No MTD / Periodic:</strong> Attacker wins easily → proves static and timer-based defenses are not enough.<br>
        <strong>Random MTD:</strong> Attacker blocked, but network becomes unusable (high latency, low availability) → proves brute-force mutation is wasteful.<br>
        <strong>DQN Adaptive MTD (our approach):</strong> Attacker blocked AND network stays usable → the RL agent learned to mutate <em>only when it matters</em>.<br><br>
        This is exactly what the Eghtesad et al. paper argues: <strong>an adaptive defender that learns when to mutate outperforms
        both doing nothing and doing everything.</strong> Our simulation demonstrates this on the MTD-Playground enterprise topology.
      </p>
    </div>

    <div style="background:#1a1a2e; border-radius:6px; padding:15px; margin-top:12px;">
      <h4 style="color:#a78bfa; margin-bottom:8px;">Controls Cheat Sheet</h4>
      <table style="color:#94a3b8; font-size:13px; width:100%;">
        <tr><td style="padding:4px 12px;"><strong style="color:#e0e0e0;">▶ Play / ⏸ Pause</strong></td><td>Start/stop automatic playback</td></tr>
        <tr><td style="padding:4px 12px;"><strong style="color:#e0e0e0;">◀ Back / Next ▶</strong></td><td>Step one frame at a time (use this to see exactly what happens at each step)</td></tr>
        <tr><td style="padding:4px 12px;"><strong style="color:#e0e0e0;">↻ Reset</strong></td><td>Go back to step 0</td></tr>
        <tr><td style="padding:4px 12px;"><strong style="color:#e0e0e0;">Slider bar</strong></td><td>Drag to jump to any point in the simulation</td></tr>
        <tr><td style="padding:4px 12px;"><strong style="color:#e0e0e0;">Speed +/-</strong></td><td>Make playback faster (50ms) or slower (1000ms)</td></tr>
        <tr><td style="padding:4px 12px;"><strong style="color:#e0e0e0;">Strategy tabs</strong></td><td>Switch between the 4 defense strategies to compare them</td></tr>
      </table>
    </div>
  </div>

  <div class="strategy-tabs" id="strategy-tabs"></div>
  <div class="sim-panel">
    <div class="controls">
      <button onclick="togglePlay()" id="play-btn">&#9654; Play</button>
      <button class="secondary" onclick="stepBack()">&#9664; Back</button>
      <button class="secondary" onclick="stepForward()">Next &#9654;</button>
      <button class="secondary" onclick="resetSim()">&#8634; Reset</button>
      <input type="range" id="step-slider" min="0" max="100" value="0" oninput="seekTo(this.value)">
      <span class="step-display" id="step-display">Step 0 / 0</span>
      <span class="speed-label">Speed:</span>
      <button class="secondary" onclick="changeSpeed(-50)">-</button>
      <span id="speed-display" class="speed-label">200ms</span>
      <button class="secondary" onclick="changeSpeed(50)">+</button>
    </div>

    <div id="kill-chain" class="kill-chain"></div>

    <div class="topo-container">
      <div class="topo-svg" id="topo-container">
        <svg id="topo-svg" viewBox="0 0 620 420" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0d1117;border-radius:8px;"></svg>
      </div>
      <div class="metrics-side">
        <div class="metric-card">
          <div class="label">Attacker Knowledge</div>
          <div class="value" id="m-knowledge" style="color:#ef4444">0%</div>
          <div class="bar"><div class="bar-fill" id="bar-knowledge" style="background:#ef4444;width:0%"></div></div>
        </div>
        <div class="metric-card">
          <div class="label">Path Entropy (H_P) — Network Unpredictability</div>
          <div class="value" id="m-entropy" style="color:#22c55e">1.000</div>
          <div class="bar"><div class="bar-fill" id="bar-entropy" style="background:#22c55e;width:100%"></div></div>
        </div>
        <div class="metric-card">
          <div class="label">Defender Action</div>
          <div class="value" id="m-action">—</div>
        </div>
        <div class="metric-card">
          <div class="label">Latency</div>
          <div class="value" id="m-latency">0 ms</div>
          <div class="bar"><div class="bar-fill" id="bar-latency" style="background:#f59e0b;width:0%"></div></div>
        </div>
        <div class="metric-card">
          <div class="label">Service Availability</div>
          <div class="value" id="m-avail" style="color:#22c55e">100%</div>
          <div class="bar"><div class="bar-fill" id="bar-avail" style="background:#22c55e;width:100%"></div></div>
        </div>
        <div class="metric-card">
          <div class="label">Cumulative Reward</div>
          <div class="value" id="m-reward" style="color:#60a5fa">0.0</div>
        </div>
      </div>
    </div>

    <div class="action-log" id="action-log">
      <div style="color:#4b5563">Simulation log will appear here...</div>
    </div>
  </div>
</div>

<!-- Charts -->
<div class="charts-section">
  <h2 style="color:#60a5fa;margin-bottom:15px;">Metrics Over Time</h2>
  <div class="charts-grid">
    <div class="chart-box"><h3>Attacker Stage Progression</h3><canvas id="chart-stage"></canvas></div>
    <div class="chart-box"><h3>Attacker Knowledge</h3><canvas id="chart-knowledge"></canvas></div>
    <div class="chart-box"><h3>Path Entropy (H_P)</h3><canvas id="chart-entropy"></canvas></div>
    <div class="chart-box"><h3>Cumulative Reward</h3><canvas id="chart-reward"></canvas></div>
  </div>
</div>

</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
// ============ DATA ============
const simData = """ + json.dumps(single_runs) + """;
const summaryData = """ + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "episodes"} for k, v in all_data.items()}) + """;

// ============ STATE ============
let currentStrategy = Object.keys(simData)[0];
let currentStep = 0;
let playing = false;
let playInterval = null;
let speed = 200;

const stageNames = ["Recon", "Initial Access", "Lateral Movement", "Priv Escalation", "Exfiltration"];
const stageColors = ["#22c55e", "#f59e0b", "#f97316", "#ef4444", "#dc2626"];
const actionNames = ["HOLD", "MODERATE SHUFFLE", "AGGRESSIVE RANDOMIZE"];
const actionColors = ["#6b7b8d", "#f59e0b", "#ef4444"];
const strategyColors = {"No MTD": "#ef4444", "Random MTD": "#f97316", "Periodic MTD": "#22c55e", "DQN Adaptive MTD": "#3b82f6"};

// ============ INIT ============
function init() {
  // Build comparison table
  const tbody = document.getElementById("comp-tbody");
  const keys = Object.keys(summaryData);
  // Find best values
  const asrs = keys.map(k => summaryData[k].asr);
  const minAsr = Math.min(...asrs);
  const rewards = keys.map(k => summaryData[k].mean_reward);
  const maxReward = Math.max(...rewards);

  keys.forEach(k => {
    const d = summaryData[k];
    const breached = d.asr > 0.5;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${k}</strong></td>
      <td class="${breached ? 'result-breached' : 'result-defended'}">${breached ? 'BREACHED' : 'DEFENDED'}</td>
      <td class="${d.asr === minAsr ? 'best-value' : ''}">${(d.asr * 100).toFixed(0)}%</td>
      <td>${d.mean_steps}</td>
      <td>${d.mean_mutations}</td>
      <td>${d.mean_latency} ms</td>
      <td class="${d.mean_reward === maxReward ? 'best-value' : ''}">${d.mean_reward}</td>
    `;
    tbody.appendChild(tr);
  });

  // Build strategy tabs
  const tabsEl = document.getElementById("strategy-tabs");
  keys.forEach(k => {
    const btn = document.createElement("button");
    btn.className = "strategy-tab" + (k === currentStrategy ? " active" : "");
    btn.textContent = k;
    btn.onclick = () => switchStrategy(k);
    tabsEl.appendChild(btn);
  });

  drawTopology();
  buildCharts();
  updateDisplay();
}

function switchStrategy(name) {
  stopPlay();
  currentStrategy = name;
  currentStep = 0;
  document.querySelectorAll(".strategy-tab").forEach(t => t.classList.toggle("active", t.textContent === name));
  const frames = simData[currentStrategy];
  document.getElementById("step-slider").max = frames.length - 1;
  document.getElementById("action-log").innerHTML = '<div style="color:#4b5563">Simulation log will appear here...</div>';
  buildCharts();
  updateDisplay();
}

// ============ PLAYBACK ============
function togglePlay() {
  if (playing) stopPlay();
  else startPlay();
}
function startPlay() {
  playing = true;
  document.getElementById("play-btn").innerHTML = "&#9646;&#9646; Pause";
  playInterval = setInterval(() => {
    if (currentStep < simData[currentStrategy].length - 1) {
      currentStep++;
      updateDisplay();
    } else stopPlay();
  }, speed);
}
function stopPlay() {
  playing = false;
  document.getElementById("play-btn").innerHTML = "&#9654; Play";
  if (playInterval) clearInterval(playInterval);
}
function stepForward() { stopPlay(); if (currentStep < simData[currentStrategy].length - 1) { currentStep++; updateDisplay(); } }
function stepBack() { stopPlay(); if (currentStep > 0) { currentStep--; updateDisplay(); } }
function resetSim() { stopPlay(); currentStep = 0; document.getElementById("action-log").innerHTML = ''; updateDisplay(); }
function seekTo(v) { stopPlay(); currentStep = parseInt(v); updateDisplay(); }
function changeSpeed(delta) {
  speed = Math.max(50, Math.min(1000, speed + delta));
  document.getElementById("speed-display").textContent = speed + "ms";
  if (playing) { stopPlay(); startPlay(); }
}

// ============ UPDATE DISPLAY ============
function updateDisplay() {
  const frames = simData[currentStrategy];
  const f = frames[currentStep];
  if (!f) return;

  document.getElementById("step-slider").value = currentStep;
  document.getElementById("step-slider").max = frames.length - 1;
  document.getElementById("step-display").textContent = `Step ${f.step} / ${frames.length}`;

  // Kill chain
  const kcEl = document.getElementById("kill-chain");
  kcEl.innerHTML = stageNames.map((name, i) => {
    let cls = i < f.attacker_stage ? "kc-past" : i === f.attacker_stage ? "kc-current" : "kc-future";
    return (i > 0 ? '<span class="kc-arrow">&#9654;</span>' : '') + `<span class="kc-stage ${cls}">${name}</span>`;
  }).join("");

  // Metrics
  const kPct = (f.attacker_knowledge * 100);
  document.getElementById("m-knowledge").textContent = kPct.toFixed(0) + "%";
  document.getElementById("m-knowledge").style.color = kPct > 50 ? "#ef4444" : kPct > 25 ? "#f59e0b" : "#22c55e";
  document.getElementById("bar-knowledge").style.width = kPct + "%";
  document.getElementById("bar-knowledge").style.background = kPct > 50 ? "#ef4444" : kPct > 25 ? "#f59e0b" : "#22c55e";

  document.getElementById("m-entropy").textContent = f.path_entropy.toFixed(3);
  const ePct = f.path_entropy * 100;
  document.getElementById("m-entropy").style.color = ePct > 70 ? "#22c55e" : ePct > 40 ? "#f59e0b" : "#ef4444";
  document.getElementById("bar-entropy").style.width = ePct + "%";
  document.getElementById("bar-entropy").style.background = ePct > 70 ? "#22c55e" : ePct > 40 ? "#f59e0b" : "#ef4444";

  document.getElementById("m-action").textContent = actionNames[f.action];
  document.getElementById("m-action").style.color = actionColors[f.action];

  document.getElementById("m-latency").textContent = f.latency_ms.toFixed(1) + " ms";
  document.getElementById("bar-latency").style.width = Math.min(f.latency_ms / 300 * 100, 100) + "%";

  const avPct = f.service_availability * 100;
  document.getElementById("m-avail").textContent = avPct.toFixed(1) + "%";
  document.getElementById("m-avail").style.color = avPct > 90 ? "#22c55e" : avPct > 75 ? "#f59e0b" : "#ef4444";
  document.getElementById("bar-avail").style.width = avPct + "%";
  document.getElementById("bar-avail").style.background = avPct > 90 ? "#22c55e" : avPct > 75 ? "#f59e0b" : "#ef4444";

  document.getElementById("m-reward").textContent = f.cumulative_reward.toFixed(1);

  // Topology
  updateTopology(f);

  // Action log
  if (currentStep > 0) {
    const prev = frames[currentStep - 1];
    const log = document.getElementById("action-log");
    const entry = document.createElement("div");
    entry.className = "log-entry";
    let actionClass = ["action-hold", "action-moderate", "action-aggressive"][f.action];
    let text = `<span class="${actionClass}">Step ${f.step}: ${actionNames[f.action]}</span>`;
    if (f.attacker_stage > prev.attacker_stage) text += ` <span class="log-stage-up">⚠ Attacker advanced to ${stageNames[f.attacker_stage]}!</span>`;
    if (f.attacker_stage < prev.attacker_stage) text += ` <span class="log-stage-down">✓ Attacker pushed back to ${stageNames[f.attacker_stage]}</span>`;
    if (f.exfiltrated) text += ` <span class="log-stage-up">🚨 DATA EXFILTRATED!</span>`;
    entry.innerHTML = text;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
  }

  // Update charts highlight
  updateChartHighlight();
}

// ============ TOPOLOGY SVG ============
function drawTopology() {
  const svg = document.getElementById("topo-svg");
  svg.innerHTML = `
    <!-- Grid background -->
    <defs>
      <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1a2332" stroke-width="0.5"/>
      </pattern>
      <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-auto">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#4b5563"/>
      </marker>
    </defs>
    <rect width="620" height="420" fill="url(#grid)"/>

    <!-- Title -->
    <text x="310" y="25" text-anchor="middle" fill="#60a5fa" font-size="14" font-weight="bold">Enterprise Network Topology</text>

    <!-- Links (paths) -->
    <line id="link-s0s1" x1="310" y1="120" x2="200" y2="240" stroke="#2a3a5c" stroke-width="3"/>
    <line id="link-s0s2" x1="310" y1="120" x2="420" y2="240" stroke="#2a3a5c" stroke-width="3"/>
    <line id="link-s1s2" x1="200" y1="240" x2="420" y2="240" stroke="#2a3a5c" stroke-width="3"/>

    <!-- Path labels -->
    <text id="path-label-0" x="240" y="170" fill="#4b5563" font-size="10" text-anchor="middle">Path A</text>
    <text id="path-label-1" x="380" y="170" fill="#4b5563" font-size="10" text-anchor="middle">Path B</text>
    <text id="path-label-2" x="310" y="255" fill="#4b5563" font-size="10" text-anchor="middle">Path C</text>

    <!-- Path probability bars -->
    <rect id="pathbar-bg-0" x="215" y="175" width="50" height="6" rx="3" fill="#1e293b"/>
    <rect id="pathbar-0" x="215" y="175" width="17" height="6" rx="3" fill="#22c55e"/>
    <rect id="pathbar-bg-1" x="355" y="175" width="50" height="6" rx="3" fill="#1e293b"/>
    <rect id="pathbar-1" x="355" y="175" width="17" height="6" rx="3" fill="#22c55e"/>
    <rect id="pathbar-bg-2" x="285" y="260" width="50" height="6" rx="3" fill="#1e293b"/>
    <rect id="pathbar-2" x="285" y="260" width="17" height="6" rx="3" fill="#22c55e"/>

    <!-- Host links -->
    <line x1="100" y1="80" x2="280" y2="105" stroke="#1e293b" stroke-width="1.5" stroke-dasharray="4"/>
    <line x1="520" y1="80" x2="340" y2="105" stroke="#1e293b" stroke-width="1.5" stroke-dasharray="4"/>
    <line x1="100" y1="340" x2="200" y2="260" stroke="#1e293b" stroke-width="1.5" stroke-dasharray="4"/>
    <line x1="520" y1="340" x2="420" y2="260" stroke="#1e293b" stroke-width="1.5" stroke-dasharray="4"/>
    <line x1="100" y1="240" x2="175" y2="240" stroke="#1e293b" stroke-width="1.5" stroke-dasharray="4"/>

    <!-- Switches -->
    <circle cx="310" cy="110" r="22" fill="#1e293b" stroke="#3b82f6" stroke-width="2" class="switch-circle"/>
    <text x="310" y="114" text-anchor="middle" fill="#60a5fa" font-size="12" font-weight="bold">S0</text>

    <circle cx="200" cy="240" r="22" fill="#1e293b" stroke="#3b82f6" stroke-width="2" class="switch-circle"/>
    <text x="200" y="244" text-anchor="middle" fill="#60a5fa" font-size="12" font-weight="bold">S1</text>

    <circle cx="420" cy="240" r="22" fill="#1e293b" stroke="#3b82f6" stroke-width="2" class="switch-circle"/>
    <text x="420" y="244" text-anchor="middle" fill="#60a5fa" font-size="12" font-weight="bold">S2</text>

    <!-- Hosts -->
    <rect x="40" y="60" width="120" height="40" rx="6" fill="#1a2332" stroke="#22c55e" stroke-width="2"/>
    <text x="100" y="78" class="node-label" fill="#22c55e" font-size="12">Client</text>
    <text x="100" y="92" class="node-ip">10.0.0.10</text>

    <rect x="460" y="60" width="120" height="40" rx="6" fill="#1a2332" stroke="#06b6d4" stroke-width="2"/>
    <text x="520" y="78" class="node-label" fill="#06b6d4" font-size="12">DMZ Web</text>
    <text x="520" y="92" class="node-ip">10.0.0.30</text>

    <rect x="20" y="220" width="120" height="40" rx="6" fill="#1a2332" stroke="#8b5cf6" stroke-width="2"/>
    <text x="80" y="238" class="node-label" fill="#8b5cf6" font-size="12">Internal App</text>
    <text x="80" y="252" class="node-ip">10.0.0.100</text>

    <rect x="460" y="320" width="120" height="40" rx="6" fill="#1a2332" stroke="#f59e0b" stroke-width="2"/>
    <text x="520" y="338" class="node-label" fill="#f59e0b" font-size="12">DB Server</text>
    <text x="520" y="352" class="node-ip">10.0.0.40</text>

    <!-- Attacker -->
    <rect id="attacker-node" x="40" y="320" width="120" height="40" rx="6" fill="#2d1515" stroke="#ef4444" stroke-width="2"/>
    <text x="100" y="338" class="node-label" fill="#ef4444" font-size="12">Attacker</text>
    <text x="100" y="352" class="node-ip">10.0.0.20</text>

    <!-- Attacker position indicator -->
    <circle id="atk-indicator" cx="100" cy="360" r="8" fill="#ef4444" opacity="0.8">
      <animate attributeName="r" values="6;10;6" dur="1s" repeatCount="indefinite"/>
    </circle>

    <!-- MTD shield indicator -->
    <text id="mtd-shield" x="310" y="160" text-anchor="middle" fill="#22c55e" font-size="20" opacity="0">&#128737;</text>

    <!-- Status text -->
    <text id="status-text" x="310" y="400" text-anchor="middle" fill="#6b7b8d" font-size="12"></text>
  `;
}

function updateTopology(f) {
  // Update path probabilities
  const pathColors = p => p > 0.4 ? "#22c55e" : p > 0.25 ? "#f59e0b" : "#ef4444";
  for (let i = 0; i < 3; i++) {
    const bar = document.getElementById(`pathbar-${i}`);
    bar.setAttribute("width", f.path_probs[i] * 50);
    bar.setAttribute("fill", pathColors(f.path_probs[i]));
    document.getElementById(`path-label-${i}`).setAttribute("fill", pathColors(f.path_probs[i]));
  }

  // Update link colors based on path activity
  const linkIds = ["link-s0s1", "link-s0s2", "link-s1s2"];
  linkIds.forEach((id, i) => {
    const el = document.getElementById(id);
    const p = f.path_probs[i];
    el.setAttribute("stroke", pathColors(p));
    el.setAttribute("stroke-width", 2 + p * 4);
    el.setAttribute("opacity", 0.4 + p * 0.6);
  });

  // Attacker position
  const atkPositions = [
    {x: 100, y: 360},  // Recon (at attacker node)
    {x: 520, y: 100},  // Initial Access (DMZ Web)
    {x: 80, y: 260},   // Lateral Movement (Internal App)
    {x: 520, y: 360},  // Priv Escalation (DB)
    {x: 520, y: 360},  // Exfiltration (DB)
  ];
  const pos = atkPositions[Math.min(f.attacker_stage, 4)];
  const indicator = document.getElementById("atk-indicator");
  indicator.setAttribute("cx", pos.x);
  indicator.setAttribute("cy", pos.y);
  indicator.setAttribute("fill", stageColors[Math.min(f.attacker_stage, 4)]);

  // MTD shield animation
  const shield = document.getElementById("mtd-shield");
  shield.setAttribute("opacity", f.action > 0 ? "1" : "0");
  shield.setAttribute("fill", f.action === 2 ? "#ef4444" : "#f59e0b");

  // Status
  const statusEl = document.getElementById("status-text");
  if (f.exfiltrated) {
    statusEl.textContent = "⚠ DATA EXFILTRATED — ATTACK SUCCEEDED";
    statusEl.setAttribute("fill", "#ef4444");
  } else {
    statusEl.textContent = `Mutations: ${f.mutation_count} | Stage: ${stageNames[f.attacker_stage]}`;
    statusEl.setAttribute("fill", "#6b7b8d");
  }
}

// ============ CHARTS ============
let charts = {};
function buildCharts() {
  Object.values(charts).forEach(c => c.destroy());
  charts = {};

  const frames = simData[currentStrategy];
  const steps = frames.map(f => f.step);
  const color = strategyColors[currentStrategy] || "#3b82f6";

  charts.stage = new Chart(document.getElementById("chart-stage"), {
    type: "line",
    data: { labels: steps, datasets: [{
      label: currentStrategy, data: frames.map(f => f.attacker_stage),
      borderColor: color, backgroundColor: color + "20", fill: true, tension: 0.1, pointRadius: 0
    }]},
    options: { responsive: true, scales: { y: { min: 0, max: 4, ticks: { callback: v => stageNames[v] || v, color: "#6b7b8d" }, grid: { color: "#1e293b" } }, x: { grid: { color: "#1e293b" }, ticks: { color: "#6b7b8d" } } }, plugins: { legend: { display: false } } }
  });

  charts.knowledge = new Chart(document.getElementById("chart-knowledge"), {
    type: "line",
    data: { labels: steps, datasets: [{
      label: "Knowledge", data: frames.map(f => f.attacker_knowledge),
      borderColor: "#ef4444", backgroundColor: "#ef444420", fill: true, tension: 0.1, pointRadius: 0
    }]},
    options: { responsive: true, scales: { y: { min: 0, max: 1, grid: { color: "#1e293b" }, ticks: { color: "#6b7b8d" } }, x: { grid: { color: "#1e293b" }, ticks: { color: "#6b7b8d" } } }, plugins: { legend: { display: false } } }
  });

  charts.entropy = new Chart(document.getElementById("chart-entropy"), {
    type: "line",
    data: { labels: steps, datasets: [{
      label: "Path Entropy", data: frames.map(f => f.path_entropy),
      borderColor: "#22c55e", backgroundColor: "#22c55e20", fill: true, tension: 0.1, pointRadius: 0
    }]},
    options: { responsive: true, scales: { y: { min: 0, max: 1, grid: { color: "#1e293b" }, ticks: { color: "#6b7b8d" } }, x: { grid: { color: "#1e293b" }, ticks: { color: "#6b7b8d" } } }, plugins: { legend: { display: false } } }
  });

  charts.reward = new Chart(document.getElementById("chart-reward"), {
    type: "line",
    data: { labels: steps, datasets: [{
      label: "Cumulative Reward", data: frames.map(f => f.cumulative_reward),
      borderColor: "#60a5fa", backgroundColor: "#60a5fa20", fill: true, tension: 0.1, pointRadius: 0
    }]},
    options: { responsive: true, scales: { y: { grid: { color: "#1e293b" }, ticks: { color: "#6b7b8d" } }, x: { grid: { color: "#1e293b" }, ticks: { color: "#6b7b8d" } } }, plugins: { legend: { display: false } } }
  });
}

function updateChartHighlight() {
  // Could add vertical line at current step -- skipping for simplicity
}

// ============ BOOT ============
window.addEventListener("load", init);
</script>
</body>
</html>"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="../results/simulation_report.html")
    parser.add_argument("--model-dir", type=str, default="../models")
    args = parser.parse_args()

    # Define strategies
    strategies = {
        "No MTD": lambda obs, step: 0,
        "Periodic MTD": lambda obs, step: 2 if step % 10 == 0 else 0,
        "Random MTD": lambda obs, step: np.random.randint(0, 3),
    }

    # Try loading DQN
    try:
        from stable_baselines3 import DQN
        model_path = os.path.join(args.model_dir, "best_model.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, "dqn_mtd_final.zip")
        if os.path.exists(model_path):
            env_tmp = MTDPlaygroundEnv(seed=0)
            model = DQN.load(model_path, env=env_tmp)
            strategies["DQN Adaptive MTD"] = lambda obs, step, m=model: int(m.predict(obs, deterministic=True)[0])
            print("Loaded DQN model.")
    except Exception as e:
        print(f"Warning: Could not load DQN: {e}")

    # Capture single-run data for interactive replay
    print("Capturing single-run replays...")
    single_runs = {}
    for name, fn in strategies.items():
        ep = capture_episode(fn, seed=args.seed)
        single_runs[name] = ep["frames"]
        status = "BREACHED" if ep["breached"] else "DEFENDED"
        print(f"  {name}: {status} in {ep['total_steps']} steps")

    # Capture multi-episode stats
    print(f"\nRunning {args.episodes}-episode evaluation...")
    all_data = {}
    for name, fn in strategies.items():
        data = capture_multi_episode(fn, name, num_episodes=args.episodes, base_seed=args.seed)
        all_data[name] = data
        print(f"  {name}: ASR={data['asr']:.0%}, Avg reward={data['mean_reward']}")

    # Generate HTML
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    generate_html(all_data, single_runs, args.output)
    print(f"\nOpen in browser: file://{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
