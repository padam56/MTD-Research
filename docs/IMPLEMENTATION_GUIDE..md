# MTD-Brain: Intelligent Decision Engine for SDN-MTD
## Project Playbook for Second Author (AI/ML & RL Logic)

### 1. Project README (For Documentation & Team Sync)
(This section is for your GitHub Repo and Paper Appendix)

**Overview:** This repository contains the AI/ML Decision Layer for an SDN-based Moving Target Defense system. It acts as the "Brain," interfacing with an ONOS Controller to execute adaptive IP and Path randomization.

**Key Components:**
- **Phase 1 (RL):** Adaptive Mutation Scheduler (DQN/PPO).
- **Phase 2 (ML):** Anomaly-based Threat Detector (Random Forest).
- **Integration:** Northbound REST API connection to ONOS.

---

### 2. GitHub Copilot Master Prompts

#### Prompt 1: The Messenger (onos_client.py)
"Create a Python class named 'ONOSClient' to act as a modular wrapper for the ONOS Northbound REST API. It must use the 'requests' library with Basic Auth. Implement 'get_network_state()' to fetch flows and port stats, and 'execute_mutation()' to POST new Intents for IP/Path randomization."

#### Prompt 2: The RL Environment (mtd_env.py)
"Using the 'ONOSClient' class, create a Gymnasium (OpenAI Gym) environment called 'SDN_MTD_Env'. The State should be flow statistics from ONOS. Actions should be Discrete(3): [No Move, Moderate, Aggressive]. The reward must balance Security Gain against Controller Latency."

#### Prompt 3: The ML Detector (threat_detector.py)
"Create a script using scikit-learn to detect 'PortScan' and 'DDoS' patterns from CIC-IDS2017. If a threat is detected via the ONOSClient flow data with >90% probability, it must trigger an immediate high-priority MTD mutation."

---

### 3. Evaluation Metrics (Paper Deliverables)
- **MTD Efficiency Score (MES):** Ratio of Security Gain to Performance Cost.
- **Attack Success Rate (ASR):** % of probes reaching the target.
- **Path Entropy:** Statistical measure of network unpredictability.