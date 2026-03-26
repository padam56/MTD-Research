# MTD-Playground: AI/ML Decision Engine

Fresh implementation for the MTD-Playground evaluation framework.
Second-author component: RL-based adaptive mutation scheduler + threat detection.

## Primary Paper Basis
- **Eghtesad et al. (2020)** — Adversarial Deep RL for Adaptive MTD (GameSec 2020)
- Markov Game formulation: Defender vs Attacker with DQN policy learning

## Structure
```
reference-only/    # Offline simulation (no ONOS needed)
mininet-ready/     # ONOS/Mininet integration code
docs/              # Literature survey, notes
results/           # Evaluation outputs (CSV, plots)
models/            # Trained model checkpoints
scripts/           # Utility scripts
```

## Milestones
1. Offline run completes → CSV + plots (reference-only/)
2. ONOS/Mininet integration executes MTD actions (mininet-ready/)

## Verified References
See [docs/LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md) for the full verified literature survey.
