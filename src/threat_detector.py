"""
threat_detector.py
------------------
ML-based anomaly trigger for the MTD Decision Engine.

Primary Technique: Ensemble Switching (HybridMTD)
    Ref: Li et al. (2021) — "HybridMTD: Ensemble-based Moving Target Defense
    with Adaptive Classifier Selection", IEEE TDSC 2021.
    The detector dynamically selects from a pool of classifiers based on the
    network's current traffic regime (low / high / anomaly state), rather than
    relying on a single model for all conditions.

Datasets: CIC-IDS2017 + InSDN
    Ref: Elsayed & Sahoo (2020) — InSDN: A Novel SDN Intrusion Detection Dataset.
    IEEE Access 2020.  Provides SDN-specific attack flows (DoS, probe, R2L).
    Combined with CIC-IDS2017 for broader attack coverage.

Ensemble Components:
    - Regime A (Low Traffic):    Random Forest  —  high interpretability, robust recall
    - Regime B (High Traffic):   XGBoost        —  speed + accuracy under load
    - Regime C (Anomaly State):  Isolation Forest — unsupervised, catches zero-days

Function:
    When suspicious flow patterns are detected with >90% confidence (or anomaly
    score threshold), calls ONOSClient.trigger_high_alert_mutation() for immediate MTD.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import os
import time
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("shap not installed. explainability disabled. Run: pip install shap")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Regime B will fall back to Random Forest.")

from .onos_client import ONOSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ThreatDetector] %(message)s")

# Labels from CIC-IDS2017 that represent active reconnaissance / attack
THREAT_LABELS = {"PortScan", "DDoS", "Web Attack – Sql Injection",
                 "Web Attack – Brute Force", "Web Attack – XSS", "Infiltration"}

# Confidence threshold above which an alert is raised
ALERT_THRESHOLD = 0.90

# Isolation Forest anomaly score threshold (contamination fraction proxy)
ANOMALY_THRESHOLD = -0.05  # scores below this are declared anomalous

# Traffic regime thresholds (flows per second)
# Ref: HybridMTD — regime boundaries tuned on CIC-IDS2017 baseline distribution
REGIME_LOW_HIGH_BOUNDARY  = 500   # < 500 flows → Regime A (RF)
REGIME_HIGH_ANML_BOUNDARY = 2000  # 500–2000 flows → Regime B (XGB); >2000 → Regime C

MODEL_PATH       = "models/threat_detector_rf.pkl"
XGB_MODEL_PATH   = "models/threat_detector_xgb.pkl"
ISOF_MODEL_PATH  = "models/threat_detector_isof.pkl"
SCALER_PATH      = "models/scaler.pkl"

# CIC-IDS2017 feature columns used for training
FEATURE_COLS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Total", "Bwd IAT Total", "Fwd PSH Flags", "Bwd PSH Flags",
    "SYN Flag Count", "RST Flag Count", "ACK Flag Count",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
]


class EnsembleSwitchingDetector:
    """
    Implements the Ensemble Switching (HybridMTD) anomaly detection strategy.

    Instead of committing to one model, the detector evaluates the current
    network traffic regime and delegates to the most appropriate classifier.
    This directly mirrors the architecture in:
        Li et al. (2021) — HybridMTD, IEEE TDSC, Section IV-B.

    Regimes and assigned classifiers:
        A — Low traffic  (≤ REGIME_LOW_HIGH_BOUNDARY flows):
              RandomForestClassifier — high recall, interpretable feature importance.
        B — High traffic  (REGIME_LOW_HIGH_BOUNDARY < flows ≤ REGIME_HIGH_ANML_BOUNDARY):
              XGBClassifier — optimised for speed under high-throughput conditions.
        C — Anomaly state (> REGIME_HIGH_ANML_BOUNDARY flows OR sudden spike):
              IsolationForest — unsupervised; detects novel zero-day patterns that
              supervised models miss due to concept drift.
    """

    def __init__(self):
        self.rf_model:   RandomForestClassifier | None = None
        self.xgb_model:  object | None                 = None   # XGBClassifier or RF fallback
        self.isof_model: IsolationForest | None        = None
        self.scaler:     StandardScaler | None         = None
        self._prev_flow_count: int = 0
        self._load_all()

    def _load_all(self):
        """Loads all three classifier models and the shared scaler from disk."""
        for path, attr, label in [
            (MODEL_PATH,      "rf_model",   "RF"),
            (XGB_MODEL_PATH,  "xgb_model",  "XGB"),
            (ISOF_MODEL_PATH, "isof_model", "IsolationForest"),
            (SCALER_PATH,     "scaler",     "Scaler"),
        ]:
            if os.path.exists(path):
                setattr(self, attr, joblib.load(path))
                logging.info(f"{label} loaded from {path}")

    def save_all(self):
        os.makedirs("models", exist_ok=True)
        if self.rf_model:   joblib.dump(self.rf_model,   MODEL_PATH)
        if self.xgb_model:  joblib.dump(self.xgb_model,  XGB_MODEL_PATH)
        if self.isof_model: joblib.dump(self.isof_model, ISOF_MODEL_PATH)
        if self.scaler:     joblib.dump(self.scaler,     SCALER_PATH)
        logging.info("All ensemble models saved.")

    def select_regime(self, flow_count: int) -> str:
        """
        Determines the current traffic regime based on flow volume.
        Also detects sudden spikes (>3x previous count) and escalates to Regime C.

        Ref: HybridMTD — Regime Selection Algorithm, Table II.
        """
        spike = flow_count > self._prev_flow_count * 3 and self._prev_flow_count > 0
        self._prev_flow_count = flow_count

        if spike or flow_count > REGIME_HIGH_ANML_BOUNDARY:
            return "C"  # Anomaly / spike state → IsolationForest
        elif flow_count > REGIME_LOW_HIGH_BOUNDARY:
            return "B"  # High traffic → XGBoost
        else:
            return "A"  # Low traffic  → Random Forest

    def predict(self, features_scaled: np.ndarray, flow_count: int) -> tuple[int, float]:
        """
        Routes the prediction request to the classifier for the current regime.

        Returns:
            (prediction, confidence): prediction is 0 (benign) or 1 (threat).
        """
        regime = self.select_regime(flow_count)
        logging.debug(f"Traffic regime: {regime} ({flow_count} flows)")

        if regime == "C" and self.isof_model is not None:
            # IsolationForest returns -1 (anomaly) or +1 (normal)
            # Ref: HybridMTD — unsupervised path for zero-day detection
            score = self.isof_model.score_samples(features_scaled)[0]
            is_anomaly = score < ANOMALY_THRESHOLD
            prediction = 1 if is_anomaly else 0
            # Convert score to a 0–1 confidence proxy
            confidence = float(np.clip(1.0 - (score + 0.5), 0.0, 1.0))
            logging.info(f"Regime C (IsolationForest): anomaly={is_anomaly}, score={score:.3f}")
            return prediction, confidence

        elif regime == "B" and self.xgb_model is not None:
            prediction = int(self.xgb_model.predict(features_scaled)[0])
            confidence = float(self.xgb_model.predict_proba(features_scaled)[0][prediction])
            logging.info(f"Regime B (XGBoost): pred={prediction}, conf={confidence:.2%}")
            return prediction, confidence

        else:
            # Regime A (or fallback when other models unavailable)
            if self.rf_model is None:
                raise RuntimeError("No classifiers available. Call train() first.")
            prediction = int(self.rf_model.predict(features_scaled)[0])
            confidence = float(self.rf_model.predict_proba(features_scaled)[0][prediction])
            logging.info(f"Regime A (RandomForest): pred={prediction}, conf={confidence:.2%}")
            return prediction, confidence

    def train_all(self, X_train, X_test, y_train, y_test):
        """
        Trains all three classifiers on the same dataset split.
        Ref: HybridMTD — recommends training all ensemble members on the
             full labelled dataset; regime selection is done at inference only.
        """
        # --- Regime A: Random Forest ---
        logging.info("Training Regime A: Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1,
            class_weight="balanced", random_state=42,
        )
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        logging.info("RF Results:\n" + classification_report(
            y_test, rf_pred, target_names=["Benign", "Threat"]))

        # --- Regime B: XGBoost ---
        if XGBOOST_AVAILABLE:
            logging.info("Training Regime B: XGBoost...")
            self.xgb_model = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                use_label_encoder=False, eval_metric="logloss",
                n_jobs=-1, random_state=42,
            )
            self.xgb_model.fit(X_train, y_train)
            xgb_pred = self.xgb_model.predict(X_test)
            logging.info("XGB Results:\n" + classification_report(
                y_test, xgb_pred, target_names=["Benign", "Threat"]))
        else:
            logging.warning("XGBoost unavailable — Regime B falls back to RF.")
            self.xgb_model = self.rf_model

        # --- Regime C: Isolation Forest (unsupervised) ---
        logging.info("Training Regime C: Isolation Forest (unsupervised)...")
        # Train on benign samples only if labels available, else full set
        benign_mask = y_train == 0
        train_data = X_train[benign_mask] if benign_mask.sum() > 100 else X_train
        self.isof_model = IsolationForest(
            n_estimators=200, contamination=0.05,
            n_jobs=-1, random_state=42,
        )
        self.isof_model.fit(train_data)
        isof_scores = self.isof_model.score_samples(X_test)
        isof_pred = (isof_scores < ANOMALY_THRESHOLD).astype(int)
        logging.info("IsolationForest Results:\n" + classification_report(
            y_test, isof_pred, target_names=["Benign", "Threat"]))


class ThreatDetector:
    """
    Wraps the EnsembleSwitchingDetector to classify live ONOS flow data.
    Implements the HybridMTD strategy: regime-based classifier selection.
    Triggers immediate MTD via ONOSClient if a threat is detected with
    probability > ALERT_THRESHOLD (or IsolationForest anomaly threshold).

    Ref: Li et al. (2021) — HybridMTD, IEEE TDSC.
         Elsayed & Sahoo (2020) — InSDN dataset for SDN-specific attack classes.
    """

    def __init__(self, config_path: str = "onos_config.json", client=None):
        """
        Args:
            config_path: Path to onos_config.json (used when client is None).
            client: Pre-built ONOSClient or MockONOSClient instance.
                    Passing a client directly enables offline/mock usage
                    without requiring a live controller or config file.
        """
        self.client  = client if client is not None else ONOSClient(config_path)
        self.ensemble = EnsembleSwitchingDetector()
        # Keep direct references for backward compatibility
        self.model  = self.ensemble.rf_model
        self.scaler = self.ensemble.scaler

    # ------------------------------------------------------------------
    # Model Persistence
    # ------------------------------------------------------------------

    def _load_model(self):
        """Loads ensemble models from disk. Delegated to EnsembleSwitchingDetector."""
        self.ensemble._load_all()
        self.model  = self.ensemble.rf_model
        self.scaler = self.ensemble.scaler

    def save_model(self):
        """Saves all ensemble models to disk."""
        self.ensemble.save_all()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, dataset_path: str, sample_size: int = 100_000):
        """
        Trains all three ensemble classifiers (RF, XGBoost, IsolationForest)
        on a CIC-IDS2017 or InSDN CSV file.

        Ref: HybridMTD \u2014 Li et al. (2021) recommends training all ensemble
             members on the same labelled dataset. Regime selection is deferred
             to inference time based on live traffic volume.

        Args:
            dataset_path: Path to a CIC-IDS2017 / InSDN .csv file.
            sample_size:  Rows to sample (for speed). Use None for full dataset.
        """
        logging.info(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path, low_memory=False)

        # Strip whitespace from column names (common CIC-IDS2017 formatting issue)
        df.columns = df.columns.str.strip()

        # Binary label: 1 = threat, 0 = benign
        df["label"] = df[" Label"].apply(
            lambda x: 1 if x.strip() in THREAT_LABELS else 0
        )

        # Drop rows with NaN/inf
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=FEATURE_COLS, inplace=True)

        available_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available_cols].values
        y = df["label"].values

        if sample_size and len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X, y = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit shared scaler on training data
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        # Pass scaler to ensemble before training so save_all() includes it
        self.ensemble.scaler = self.scaler
        self.ensemble.train_all(X_train, X_test, y_train, y_test)
        self.ensemble.save_all()

        # Sync backward-compat references
        self.model  = self.ensemble.rf_model
        self.scaler = self.ensemble.scaler
        return self.ensemble.rf_model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_from_flows(self, flows: list) -> tuple[int, float]:
        """
        Converts raw ONOS flow data into a feature vector and runs
        ensemble-switched inference.

        Routes to the appropriate classifier based on current traffic regime.
        Ref: HybridMTD — Algorithm 1 (Ensemble Switching).

        Returns:
            (prediction, confidence): prediction is 0 (benign) or 1 (threat).
        """
        if self.ensemble.rf_model is None and self.ensemble.xgb_model is None:
            raise RuntimeError("No models loaded. Call train() first.")
        if self.scaler is None:
            raise RuntimeError("Scaler not loaded. Call train() first.")

        features = self._extract_features_from_flows(flows)
        features_scaled = self.scaler.transform([features])
        prediction, confidence = self.ensemble.predict(features_scaled, flow_count=len(flows))
        return prediction, confidence

    def run_detection_loop(self, poll_interval: float = 5.0):
        """
        Continuously polls ONOS flow data and triggers MTD on high-confidence threats.

        Args:
            poll_interval: Seconds between each poll cycle.
        """
        logging.info(f"Detection loop started. Polling every {poll_interval}s.")
        while True:
            try:
                state = self.client.get_network_state()
                flows = state.get("flows", [])

                if not flows:
                    logging.debug("No flows returned from ONOS.")
                    time.sleep(poll_interval)
                    continue

                prediction, confidence = self.predict_from_flows(flows)

                if prediction == 1 and confidence >= ALERT_THRESHOLD:
                    logging.warning(
                        f"THREAT DETECTED! Confidence: {confidence:.2%}. Triggering MTD."
                    )
                    self.client.trigger_high_alert_mutation()
                else:
                    logging.info(
                        f"Network nominal. Threat probability: {confidence if prediction == 1 else 1 - confidence:.2%}"
                    )

            except Exception as e:
                logging.error(f"Detection loop error: {e}")

            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    def _extract_features_from_flows(self, flows: list) -> list:
        """
        Maps ONOS flow objects to the CIC-IDS2017-aligned feature vector.

        Supports two flow formats:
          - MockONOSClient flows: carry a 'features' sub-dict with CIC-IDS2017-style
            keys (flow_bytes_per_sec, syn_flag_count, etc.). Used for offline demos.
          - Live ONOS flows: carry top-level 'packets' and 'bytes' counters.
        """
        if not flows:
            return [0.0] * len(FEATURE_COLS)

        n = max(len(flows), 1)

        # ── Aggregate mock feature dicts (offline / MockONOSClient) ───────
        mock_feats = [f.get("features", {}) for f in flows if f.get("features")]
        if mock_feats:
            # Use _mean so per-flow values match the per-row training scale.
            # Using _sum would inflate flag counts by n_flows (~1000x mismatch).
            def _mean(key): return sum(d.get(key, 0.0) for d in mock_feats) / len(mock_feats)

            dur  = _mean("flow_duration")      # 0.001-1s attack; 5-60s normal
            bps  = _mean("flow_bytes_per_sec") # 50k-500k attack; 100-5k normal
            pps  = _mean("flow_pkts_per_sec")  # 500-10k attack; 0.5-50 normal
            fpl  = _mean("fwd_pkt_len_mean")   # 40-80 attack; 200-1200 normal
            syn  = _mean("syn_flag_count")     # 400-1000 attack; 0-3 normal (key feature)
            rst  = _mean("rst_flag_count")     # 50-300 attack; 0-3 normal
            psh  = _mean("psh_flag_count")
            ack  = _mean("ack_flag_count")
            idle = _mean("idle_mean")          # ~0.001 attack; 0.01-2 normal
            act  = _mean("active_mean")        # Flow IAT Std proxy
            sfb  = _mean("subflow_fwd_bytes")  # subflow bytes
            avg_pkt_sz = bps / max(pps, 1.0)

            features = [
                dur,         # Flow Duration          -- 0.001-1s DDoS vs 5-60s benign
                pps * dur,   # Total Fwd Packets      -- pps x duration proxy
                0.0,         # Total Backward Packets
                sfb,         # Total Length Fwd Pkts  -- subflow_fwd_bytes
                0.0,         # Total Length Bwd Pkts
                fpl * 1.5,   # Fwd Packet Length Max
                fpl * 0.5,   # Fwd Packet Length Min
                fpl,         # Fwd Packet Length Mean -- 40-80 DDoS vs 200-1200 benign
                0.0, 0.0, 0.0,  # Bwd Packet Length stats
                bps,         # Flow Bytes/s            -- 50k-500k DDoS vs 100-5k benign
                pps,         # Flow Packets/s          -- 500-10k DDoS vs 0.5-50 benign
                0.0,         # Flow IAT Mean
                act,         # Flow IAT Std            -- active_mean proxy
                0.0,         # Fwd IAT Total
                idle,        # Bwd IAT Total           -- idle_mean
                psh,         # Fwd PSH Flags
                0.0,         # Bwd PSH Flags
                syn,         # SYN Flag Count          -- 400-1000 DDoS vs 0-3 benign (KEY)
                rst,         # RST Flag Count
                ack,         # ACK Flag Count
                avg_pkt_sz,  # Average Packet Size
                0.0, 0.0,    # Avg Fwd / Bwd Segment Size
            ]
            return features[:len(FEATURE_COLS)]

        # ── Fallback: live ONOS flows with packet/byte counters ───────────
        total_packets = sum(f.get("packets", 0) for f in flows)
        total_bytes   = sum(f.get("bytes", 0)   for f in flows)
        features = [
            float(n),
            total_packets,
            0.0,
            total_bytes,
            0.0,
            total_packets / n, 0.0, total_packets / n,
            0.0, 0.0, 0.0,
            total_bytes   / n,
            total_packets / n,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0, 0.0,
            total_bytes / max(total_packets, 1),
            0.0, 0.0,
        ]
        return features[:len(FEATURE_COLS)]

    def get_threat_score(self, flows: list) -> float:
        """
        Returns the raw threat probability (0.0 – 1.0) without triggering action.
        Used to inject the score into the RL observation vector (obs[10]).
        Ref: Eghtesad et al. (2020) — threat signal injected into Markov state.
        """
        if self.model is None:
            return 0.0
        prediction, confidence = self.predict_from_flows(flows)
        return confidence if prediction == 1 else 0.0

    def explain_prediction(self, flows: list) -> dict:
        """
        Uses SHAP (SHapley Additive exPlanations) to explain which network
        flow features most influenced the current prediction.

        SHAP values provide a unified measure of feature importance grounded
        in cooperative game theory. Each feature is assigned a Shapley value
        representing its marginal contribution to the prediction.

        This answers: "WHY did the model flag this traffic as a threat?"
        Useful for paper results (interpretability section) and debugging.

        Ref: Lundberg & Lee (2017) — 'A Unified Approach to Interpreting Model
             Predictions', NeurIPS 2017.
             TreeSHAP for tree-based models: Lundberg et al. (2020),
             'From local explanations to global understanding with explainable AI
              for trees', Nature Machine Intelligence.

        Returns:
            dict with keys:
                'feature_names'  — list of feature names
                'shap_values'    — list of SHAP values (positive = pushed toward threat)
                'base_value'     — model's expected output (baseline)
                'prediction'     — 0 (benign) or 1 (threat)
                'confidence'     — model confidence
        """
        if not SHAP_AVAILABLE:
            return {"error": "shap not installed. Run: pip install shap"}

        if self.ensemble.rf_model is None or self.scaler is None:
            return {"error": "Model not trained. Call train() first."}

        features = self._extract_features_from_flows(flows)
        features_scaled = self.scaler.transform([features])
        prediction, confidence = self.predict_from_flows(flows)

        # TreeSHAP — exact and fast for Random Forest and XGBoost
        # Ref: Lundberg et al. (2020) Nature MI — Section 2 TreeSHAP algorithm
        explainer = shap.TreeExplainer(self.ensemble.rf_model)
        shap_vals  = explainer.shap_values(features_scaled)

        # Handle both old (list of 2D arrays) and new (3D array) SHAP formats:
        #   Old shap <0.46:  list[ (n_samples,n_features), ... ]  — index by class first
        #   New shap >=0.46: ndarray (n_samples, n_features, n_classes) — last axis is class
        sv = np.array(shap_vals)
        if sv.ndim == 3 and sv.shape[0] == 1:
            # New format: (1, n_features, n_classes) — take class 1
            threat_shap = sv[0, :, 1]
        elif sv.ndim == 3 and sv.shape[2] == 2:
            # Also new format shape variant: (1, n_features, 2)
            threat_shap = sv[0, :, 1]
        elif isinstance(shap_vals, list) and len(shap_vals) == 2:
            # Old format: [class0_arr, class1_arr]
            threat_shap = np.array(shap_vals[1][0])
        else:
            threat_shap = sv.flatten()[:len(FEATURE_COLS)]

        # Sort by absolute importance
        sorted_pairs = sorted(
            zip(FEATURE_COLS[:len(threat_shap)], threat_shap.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        logging.info(
            f"SHAP top-3 features: "
            + ", ".join(f"{n}={v:+.3f}" for n, v in sorted_pairs[:3])
        )

        return {
            "feature_names": [p[0] for p in sorted_pairs],
            "shap_values":   [p[1] for p in sorted_pairs],
            "base_value":    float(explainer.expected_value[1]
                             if isinstance(explainer.expected_value, (list, np.ndarray))
                             else explainer.expected_value),
            "prediction":    prediction,
            "confidence":    round(confidence, 4),
        }
