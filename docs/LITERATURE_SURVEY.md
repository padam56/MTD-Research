# Literature Survey: MTD + SDN + AI/ML
## For MTD-Playground Paper (2026 CCS Submission)

> Last updated: 2026-03-26
> All papers below have been manually verified as real, published works.

---

## 1. PRIMARY PAPER (Implementation Target)

### Eghtesad, Vorobeychik & Laszka (2020) — Adversarial Deep RL for Adaptive MTD
- **Venue:** GameSec 2020 (Springer LNCS vol. 12513, pp. 58-79)
- **DOI:** 10.1007/978-3-030-64793-3_4
- **arXiv:** 1911.11972
- **Status:** Published, peer-reviewed conference paper

**What it does:**
- Models MTD as a two-player general-sum game under partial observability
- Defender chooses system configurations; attacker performs reconnaissance + exploitation
- Uses multi-agent Deep RL with Double Oracle algorithm
- Formulates as PO-MDP (Partially Observable Markov Decision Process)

**Why it fits our work:**
- Directly provides the Markov Game formulation for attacker-defender interaction
- Maps to our path randomization: defender selects network paths, attacker tries to discover/exploit
- Evaluation metrics (attack success rate, defender cost) align with MTD-Playground Section 7
- Zero-sum payoff structure: R_d = -R_a (Eq. 6 in their paper)

**Key concepts we use:**
| Concept | Our Implementation |
|---|---|
| State S = (s_net, s_attacker) | Network flow stats + attacker knowledge estimate |
| Action A = {no_move, moderate, aggressive} | Discrete(3) mutation levels |
| Attacker Knowledge Entropy H_A | Measures how much attacker knows about topology |
| Path Entropy H_P | Measures unpredictability of network paths |
| Cost-of-Moving c_m | Performance penalty for reconfiguration |

**Code availability:** None (we implement from the paper's formulation)

---

## 2. SUPPORTING PAPERS (Validated & Used)

### 2a. Sengupta, Chowdhary et al. (2020) — MTD Survey
- **Title:** "A Survey of Moving Target Defenses for Network Security"
- **Venue:** IEEE Communications Surveys & Tutorials, vol. 22, no. 3, pp. 1909-1941
- **DOI:** 10.1109/COMST.2020.2982955
- **Status:** Published journal (top-tier survey venue)
- **Use:** MTD taxonomy, entropy metric formalization, evaluation framework concepts
- **Key contribution to our work:** Formalizes the "what/how/when" MTD design dimensions we reference

### 2b. Jafarian, Al-Shaer & Duan (2015) — Random Host Mutation
- **Title:** "An Effective Address Mutation Approach for Disrupting Reconnaissance Attacks"
- **Venue:** IEEE Trans. Information Forensics & Security (TIFS), vol. 10, no. 12, pp. 2562-2577
- **DOI:** 10.1109/TIFS.2015.2462726
- **Status:** Published journal (top-tier security venue)
- **Use:** Foundational IP mutation technique for SDN using OpenFlow
- **Key contribution to our work:** Baseline IP randomization mechanism, probabilistic analysis of reconnaissance disruption
- **Note:** No AI/ML (analytical/probabilistic approach) — used as baseline comparison

### 2c. DQ-MOTAG — Chai et al. (2020)
- **Title:** "DQ-MOTAG: Deep Reinforcement Learning-based Moving Target Defense Against DDoS Attacks"
- **Venue:** IEEE Fifth International Conference on Data Science in Cyberspace (DSC), pp. 375-379
- **DOI:** 10.1109/DSC50466.2020.00064
- **Status:** Published conference paper
- **Use:** Validates DQN as a viable approach for MTD action selection
- **Key contribution to our work:** Deep Q-learning for balancing security gain vs reconfiguration cost — directly comparable to our approach

### 2d. Prakash & Wellman (2015) — Empirical Game-Theoretic Analysis for MTD
- **Title:** "Empirical Game-Theoretic Analysis for Moving Target Defense"
- **Venue:** ACM MTD Workshop (co-located with CCS 2015), pp. 57-65
- **DOI:** 10.1145/2808475.2808483
- **Status:** Published workshop paper
- **Use:** Game-theoretic evaluation methodology
- **Key contribution to our work:** Framework for analyzing attacker-defender strategy interactions via simulation-derived game models

### 2e. Henger Li, Shen & Zheng (2020) — Spatial-Temporal MTD
- **Title:** "Spatial-Temporal Moving Target Defense: A Markov Stackelberg Game Model"
- **Venue:** AAMAS 2020, pp. 717-725
- **arXiv:** 2002.10390
- **Status:** Published conference paper (top-tier multi-agent systems venue)
- **Code:** Available at github.com/HengerLi/SPT-MTD
- **Use:** Stackelberg game formulation for optimal timing of MTD reconfigurations
- **Key contribution to our work:** Models WHEN to switch (temporal dimension) + switching costs

---

## 3. INFRASTRUCTURE / DATASET PAPERS

### 3a. Sharafaldin, Lashkari & Ghorbani (2018) — CIC-IDS2017
- **Title:** "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"
- **Venue:** ICISSP 2018
- **Status:** Published, widely cited (4000+ citations)
- **Use:** Training data for threat detector (PortScan, DDoS, Web Attack classes)

### 3b. Elsayed & Sahoo (2020) — InSDN Dataset
- **Title:** "InSDN: A Novel SDN Intrusion Detection Dataset"
- **Venue:** IEEE Access
- **DOI:** 10.1109/ACCESS.2020.3022633
- **Use:** SDN-specific attack classes for broader coverage

---

## 4. ADDITIONAL REAL PAPERS IN THE SPACE (for related work section)

### 4a. Abdelkhalek, Hyder & Manimaran (2022)
- **Title:** "Moving Target Defense Routing for SDN-enabled Smart Grid"
- **Venue:** IEEE CSR 2022, pp. 215-220
- **DOI:** 10.1109/CSR54599.2022.9850341
- **Relevance:** Domain-specific (smart grid) SDN route randomization

### 4b. Gudla & Sung (2020)
- **Title:** "Moving Target Defense Discrete Host Address Mutation and Analysis in SDN"
- **Venue:** IEEE CSCI 2020, pp. 55-61
- **DOI:** 10.1109/CSCI51800.2020.00016
- **Relevance:** Per-host individualized mutation intervals based on flow statistics

### 4c. Aydeger et al. (2025) — MTDNS
- **Title:** "MTDNS: Moving Target Defense for Resilient DNS Infrastructure"
- **Venue:** IEEE CCNC 2025
- **arXiv:** 2410.02254
- **Relevance:** DNS server shuffling using SDN+NFV

### 4d. Zhou et al. (2025) — Federated MADRL for MTD
- **Title:** "From Static to Adaptive Defense: Federated Multi-Agent Deep RL-Driven MTD"
- **Venue:** IEEE Trans. Cognitive Communications and Networking (TCCN), 2025
- **Relevance:** Federated multi-agent DRL for MTD in distributed environments (UAV swarms)

### 4e. Achleitner et al. (2016) — Cyber Deception
- **Title:** "Cyber Deception: Virtual Networks to Defend Insider Reconnaissance"
- **Venue:** ACM MIST '16, pp. 57-68
- **DOI:** 10.1145/2995959.2995962
- **Relevance:** Deception-based MTD using virtual network views

---

## 5. PAPERS REMOVED (Bogus / Not Appropriate)

| Paper | Problem |
|---|---|
| "Li et al. (2021) HybridMTD, IEEE TDSC" | **DOES NOT EXIST.** No such publication found on IEEE Xplore, Google Scholar, Semantic Scholar, or DBLP. Fabricated by ChatGPT. |
| MTDSense (Moghaddam et al. 2024) | arXiv preprint only (not peer-reviewed). Studies MTD fingerprinting from attacker perspective — tangential to our evaluation framework. |

---

## 6. RECOMMENDED CITATION STRATEGY FOR THE PAPER

**For Introduction (why MTD matters):**
- Sengupta et al. 2020 survey [general MTD]
- Jafarian et al. 2015 [foundational IP mutation]

**For Related Work (SDN-based MTD):**
- Jafarian et al. 2015, Abdelkhalek et al. 2022, Gudla & Sung 2020, Aydeger et al. 2025

**For Related Work (AI/ML-driven MTD):**
- Eghtesad et al. 2020, DQ-MOTAG 2020, Zhou et al. 2025

**For Related Work (Game-theoretic MTD):**
- Prakash & Wellman 2015, Li et al. 2020 (SPT-MTD)

**For Our Method (what we implement):**
- Eghtesad et al. 2020 [primary — Markov game + DRL formulation]
- DQ-MOTAG 2020 [validates DQN approach]
- Sengupta et al. 2020 [entropy metric]

**For Evaluation:**
- CIC-IDS2017, InSDN datasets
