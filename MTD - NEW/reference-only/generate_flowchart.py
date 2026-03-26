#!/usr/bin/env python3
"""
generate_flowchart.py — Clean, readable project flowchart
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(28, 40))
ax.set_xlim(0, 28)
ax.set_ylim(0, 40)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Colors
BLUE = "#2563eb"
GREEN = "#16a34a"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"
RED = "#dc2626"
GRAY = "#64748b"
LIGHT_BLUE = "#eff6ff"
LIGHT_GREEN = "#f0fdf4"
LIGHT_PURPLE = "#f5f3ff"
LIGHT_ORANGE = "#fff7ed"
LIGHT_RED = "#fef2f2"
LIGHT_GRAY = "#f8fafc"
DARK = "#1e293b"


def box(x, y, w, h, title, lines, color, bg, badge=None):
    """Draw a box with title and bullet points, text fully inside."""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                           facecolor=bg, edgecolor=color, linewidth=2.5)
    ax.add_patch(rect)

    # Title bar
    title_rect = FancyBboxPatch((x, y + h - 0.8), w, 0.8,
                                 boxstyle="round,pad=0.2",
                                 facecolor=color, edgecolor=color, linewidth=0)
    ax.add_patch(title_rect)
    ax.text(x + w/2, y + h - 0.4, title, fontsize=10, color="white",
            fontweight="bold", ha="center", va="center")

    # Badge
    if badge:
        bc = GREEN if badge == "DONE" else BLUE
        ax.text(x + w - 0.3, y + h - 0.4, badge, fontsize=7, color="white",
                fontweight="bold", ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=bc, edgecolor="none"))

    # Content lines
    for i, line in enumerate(lines):
        ax.text(x + 0.4, y + h - 1.2 - i * 0.4, line, fontsize=8.5,
                color=DARK, ha="left", va="top")


def arrow(x1, y1, x2, y2, label=""):
    """Draw arrow between boxes."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2,
                                connectionstyle="arc3,rad=0"))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.15, label, fontsize=7.5, color=GRAY,
                ha="center", va="bottom", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none"))


def arrow_down(x, y1, y2, label=""):
    arrow(x, y1, x, y2, label)


def section(y, text, color):
    """Section divider."""
    ax.plot([0.5, 27.5], [y, y], color=color, linewidth=2, alpha=0.4)
    ax.text(14, y + 0.2, text, fontsize=14, color=color, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, linewidth=1.5))


# ============ TITLE ============
ax.text(14, 39.3, "MTD-Playground: Complete Project Flowchart",
        fontsize=20, color=DARK, fontweight="bold", ha="center")
ax.text(14, 38.8, "Everything We Did — From Literature Survey to Live ONOS Demo",
        fontsize=11, color=GRAY, ha="center")

# ============ PHASE 1 ============
section(38, "PHASE 1 — Literature Survey & Paper Selection", BLUE)

box(1, 35, 7.5, 2.7, "Step 1: Survey 15+ Papers", [
    "• Searched IEEE Xplore, ACM DL, Google Scholar",
    "• Checked each paper actually exists (DOI, venue)",
    "• Found 2 bogus papers from old codebase",
    "• Verified 14 real, published papers",
], BLUE, LIGHT_BLUE, "DONE")

box(10, 35, 7.5, 2.7, "Step 2: Selected Eghtesad et al. (2020)", [
    "• Published at GameSec 2020 (Springer LNCS)",
    "• Markov Game + Deep RL formulation",
    "• Models both attacker AND defender learning",
    "• Maps directly to SDN path randomization",
], BLUE, LIGHT_BLUE, "DONE")

box(19.5, 35, 7.5, 2.7, "Output: Literature Survey", [
    "• docs/LITERATURE_SURVEY.md",
    "• 14 verified references with DOIs",
    "• Why we chose this paper (rationale table)",
    "• Citation strategy for paper sections",
], ORANGE, LIGHT_ORANGE, "DONE")

arrow(8.5, 36.3, 10, 36.3, "selected")
arrow(17.5, 36.3, 19.5, 36.3, "produced")

# ============ PHASE 2 ============
section(34, "PHASE 2 — Offline Simulation (reference-only/)", GREEN)

box(1, 30.5, 8.5, 3.2, "Step 3: Built Gymnasium Environment", [
    "• mtd_env.py — the core simulation",
    "• Simulates: 5 hosts, 3 switches, 3 paths",
    "• AttackerSim: 5-stage kill chain",
    "• NetworkSim: path probabilities + entropy",
    "• 14-dim observation vector → Discrete(3) actions",
    "• Zero-sum reward: R_defender = -R_attacker",
], GREEN, LIGHT_GREEN, "DONE")

box(10.5, 30.5, 8.5, 3.2, "Step 4: Trained DQN Agent", [
    "• train.py — Double DQN (van Hasselt 2016)",
    "• Network: MLP [256, 256] layers",
    "• 200,000 timesteps of RL training",
    "• NOT trained on any dataset!",
    "• Learns by trial-and-error in environment",
    "• Output: models/best_model.zip",
], GREEN, LIGHT_GREEN, "DONE")

box(20, 30.5, 7, 3.2, "Step 5: Evaluated Strategies", [
    "• evaluate.py — 50 episodes each",
    "• No MTD → 100% ASR (always fails)",
    "• Periodic → 97% ASR (barely helps)",
    "• Random → 10% ASR, 221ms latency",
    "• DQN → 47% ASR, 74ms latency",
    "• Output: CSVs + 6 plots (fig1-6.png)",
], GREEN, LIGHT_GREEN, "DONE")

arrow(9.5, 32.1, 10.5, 32.1, "train on")
arrow(19, 32.1, 20, 32.1, "evaluate")

# Visualization row
box(1, 27.5, 8.5, 2.5, "Step 6: Terminal + HTML Visualization", [
    "• simulate.py — live terminal animation (Rich)",
    "• generate_report.py → simulation_report.html",
    "• Interactive replay: play/pause/step controls",
    "• SVG topology + Chart.js graphs + Q&A section",
], GREEN, LIGHT_GREEN, "DONE")

box(10.5, 27.5, 8.5, 2.5, "Output: Interactive HTML Report", [
    "• results/simulation_report.html",
    "• Open in any browser — self-contained",
    "• Full paper context + literature survey",
    "• Explains what/why/how for each metric",
], ORANGE, LIGHT_ORANGE, "DONE")

arrow(9.5, 28.75, 10.5, 28.75, "generates")
arrow_down(5.25, 30.5, 30, "")

# ============ PHASE 3 ============
section(26.5, "PHASE 3 — Real SDN Infrastructure (on your machine!)", PURPLE)

box(1, 23.5, 6, 2.7, "Step 7: Install SDN Stack", [
    "• apt install mininet (2.3.0)",
    "• apt install openvswitch-switch (3.3.4)",
    "• docker pull onosproject/onos:2.7.0",
    "• All running on your laptop",
], PURPLE, LIGHT_PURPLE, "DONE")

box(8, 23.5, 6, 2.7, "Step 8: Start ONOS Controller", [
    "• Docker container on port 8181",
    "• GUI: http://localhost:8181/onos/ui",
    "• Login: onos / rocks",
    "• Activated: OpenFlow, fwd, host apps",
], PURPLE, LIGHT_PURPLE, "DONE")

box(15, 23.5, 6, 2.7, "Step 9: Create Topology", [
    "• topology.py — Mininet script",
    "• 5 hosts, 3 OVS switches",
    "• Matches paper Figure 1 exactly",
    "• Connected via OpenFlow 1.3",
], PURPLE, LIGHT_PURPLE, "DONE")

box(22, 23.5, 5.5, 2.7, "ONOS GUI Shows", [
    "• 3 switches visible",
    "• 6 links (bidirectional)",
    "• 5 hosts with IPs",
    "• Flow rules on each switch",
], ORANGE, LIGHT_ORANGE, "DONE")

arrow(7, 24.85, 8, 24.85)
arrow(14, 24.85, 15, 24.85)
arrow(21, 24.85, 22, 24.85)

# ============ PHASE 4 ============
section(22.5, "PHASE 4 — Live ONOS + DQN Integration", ORANGE)

box(1, 19.2, 8.5, 3, "Step 10: ONOS REST API Client", [
    "• onos_client.py — talks to real ONOS",
    "• GET /flows → read actual flow rules",
    "• GET /statistics/ports → real packet counts",
    "• POST /flows/{id} → install new rules",
    "• DELETE → remove rules (path mutation!)",
    "• Randomizes output ports on switches",
], ORANGE, LIGHT_ORANGE, "DONE")

box(10.5, 19.2, 8.5, 3, "Step 11: Live Gym Environment", [
    "• mtd_env_live.py — same 14-dim obs space",
    "• Reads REAL flows from ONOS (not simulated)",
    "• Computes H_P from actual flow rule ports",
    "• Estimates knowledge from flow stability",
    "• Same action space: {0=hold, 1=mod, 2=aggr}",
    "• Trained model works WITHOUT retraining!",
], ORANGE, LIGHT_ORANGE, "DONE")

box(20, 19.2, 7.5, 3, "Step 12: Run DQN on Real Network", [
    "• run_live.py — loads best_model.zip",
    "• Every 2 seconds: read ONOS → pick action",
    "• DQN applied MODERATE mutations mostly",
    "• Real flow rules changed on OVS switches",
    "• Measured: 3-12ms latency, H_P=0.87-0.95",
    "• Output: results/live_results.csv",
], ORANGE, LIGHT_ORANGE, "DONE")

arrow(9.5, 20.7, 10.5, 20.7, "reads from")
arrow(19, 20.7, 20, 20.7, "uses")

# ============ PHASE 5 ============
section(18.2, "PHASE 5 — Full Demo (Attacker + Defender Simultaneously)", RED)

box(1, 15, 8.5, 2.9, "Step 13: Attack Scripts", [
    "• recon.sh — real ping sweep from attacker host",
    "• ddos.sh — SYN flood (hping3/ping -f)",
    "• lateral.sh — pivot through web→app→db",
    "• All run inside Mininet (real packets!)",
    "• Packets go through actual OVS switches",
], RED, LIGHT_RED, "DONE")

box(10.5, 15, 8.5, 2.9, "Step 14: Full Demo Script", [
    "• run_demo.py — does EVERYTHING in one script",
    "• Creates Mininet topology automatically",
    "• Attacker thread: runs kill chain stages",
    "• Defender thread: DQN reads ONOS + mutates",
    "• Both run concurrently on real network",
], RED, LIGHT_RED, "DONE")

box(20, 15, 7.5, 2.9, "Demo Results", [
    "• Attacker ran real pings through SDN",
    "• DQN defended with real flow rule changes",
    "• ONOS GUI showed flows updating live",
    "• Avg latency: 6.5ms, H_P: 0.921",
    "• Output: live_demo_results.csv",
], ORANGE, LIGHT_ORANGE, "DONE")

arrow(9.5, 16.45, 10.5, 16.45, "while")
arrow(19, 16.45, 20, 16.45, "produced")

# ============ KEY INSIGHT ============
section(14, "The Key Insight — Why This All Matters", DARK)

box(1, 10.8, 12.5, 2.9, "What the Results Prove", [
    "• No MTD:     Attacker ALWAYS wins (100% success rate) — static networks are vulnerable",
    "• Periodic:    Timer-based mutation barely helps (97% success) — too predictable",
    "• Random:     Works (10% success) but KILLS performance (221ms latency, 82% availability)",
    "• DQN:          Best trade-off: 47% success, only 74ms latency, 89% availability",
    "",
    "CONCLUSION: Adaptive RL-based MTD outperforms both static and brute-force approaches.",
    "            The DQN learns to mutate ONLY when the attacker is gaining knowledge.",
], DARK, LIGHT_GRAY)

box(14.5, 10.8, 13, 2.9, "Model Input → Output", [
    "INPUT (14 features):  flow count, latency, path entropy, attacker knowledge,",
    "                      service availability, mutation count, recon accuracy, ...",
    "",
    "DQN AGENT:   MLP [256,256] → picks action with highest Q-value",
    "",
    "OUTPUT (1 of 3):  0=HOLD (do nothing)  |  1=MODERATE (shuffle 1 path)",
    "                  2=AGGRESSIVE (randomize all paths)",
], DARK, LIGHT_GRAY)

# ============ FILE MAP ============
section(9.8, "Complete File Map", GRAY)

files = [
    ("MTD - NEW/", ""),
    ("├── setup.sh", "Install ONOS + Mininet + OVS"),
    ("├── requirements.txt", "Python dependencies"),
    ("│", ""),
    ("├── reference-only/", "OFFLINE SIMULATION"),
    ("│   ├── mtd_env.py", "Gymnasium environment (THE CORE)"),
    ("│   ├── train.py", "Train DQN (200k steps)"),
    ("│   ├── evaluate.py", "Compare strategies → CSV + plots"),
    ("│   ├── simulate.py", "Terminal live animation"),
    ("│   └── generate_report.py", "Interactive HTML report"),
    ("│", ""),
    ("├── mininet-ready/", "REAL SDN INTEGRATION"),
    ("│   ├── topology.py", "Mininet topology (5h, 3sw)"),
    ("│   ├── onos_client.py", "ONOS REST API wrapper"),
    ("│   ├── mtd_env_live.py", "Live Gym env (reads ONOS)"),
    ("│   ├── run_live.py", "Run DQN on real network"),
    ("│   ├── run_demo.py", "Full demo: attacker + defender"),
    ("│   └── attack_scripts/", "recon.sh, ddos.sh, lateral.sh"),
    ("│", ""),
    ("├── models/", "best_model.zip, dqn_mtd_final.zip"),
    ("├── results/", "CSVs, plots, HTML report, flowchart"),
    ("└── docs/", "LITERATURE_SURVEY.md, ROADMAP.md"),
]

for i, (path, desc) in enumerate(files):
    y_pos = 9.2 - i * 0.38
    ax.text(2, y_pos, path, fontsize=9, color=DARK, fontfamily="monospace",
            fontweight="bold" if path.endswith("/") else "normal")
    if desc:
        ax.text(14, y_pos, f"← {desc}", fontsize=8.5, color=GRAY)

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
plt.savefig("../results/project_flowchart.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: ../results/project_flowchart.png")
