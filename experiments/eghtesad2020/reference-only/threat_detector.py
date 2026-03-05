"""
Threat Detector — 3-Regime ML Ensemble
Adapts ensemble based on traffic regime (normal/recon/attack)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
from typing import Dict, Tuple


class ThreatDetector:
    """
    3-regime ensemble threat detector.
    - Regime A (low traffic): Random Forest
    - Regime B (high traffic): XGBoost
    - Regime C (anomaly spike): IsolationForest
    """
    
    # Feature columns used
    FEATURE_COLS = [
        'duration', 'bytes', 'packets', 'syn_flags', 'fin_flags', 'rst_flags', 'ack_flags',
        'bytes_per_sec', 'packets_per_sec', 'protocol_tcp', 'protocol_udp',
        'src_port_entropy', 'dst_port_entropy', 'flag_rate',
        'tcp_fraction', 'udp_fraction', 'icmp_fraction',
        'flow_iat_mean', 'flow_iat_std', 'fwd_packets_mean',
        'bwd_packets_mean', 'src_ports_unique', 'dst_ports_unique',
        'packet_size_mean', 'packet_size_std'
    ]
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        self.iso_model = IsolationForest(n_estimators=100, random_state=42)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = self.FEATURE_COLS
        
    def train(self, csv_path: str):
        """Train all ensemble models on labeled dataset."""
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Ensure all feature columns exist
        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        
        X = df[self.FEATURE_COLS].fillna(0).values
        y = df.get('label', np.zeros(len(df))).values
        
        # Handle label column variations
        if 'Label' in df.columns:
            y = (df['Label'].str.lower() != 'benign').astype(int).values
        
        print(f"Training on {len(df)} samples, {len(self.FEATURE_COLS)} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        print("  - Fitting Random Forest...")
        self.rf_model.fit(X_scaled, y)
        
        print("  - Fitting XGBoost...")
        self.xgb_model.fit(X_scaled, y)
        
        print("  - Fitting IsolationForest...")
        self.iso_model.fit(X_scaled)
        
        self.is_trained = True
        print("✓ Detector training complete")
    
    def predict(self, flow_dict: Dict) -> Tuple[int, float]:
        """
        Predict if flow is attack or benign.
        
        Returns:
            (prediction, confidence)
            prediction: 0=benign, 1=attack
            confidence: [0, 1] prediction confidence
        """
        if not self.is_trained:
            return 0, 0.5
        
        x = self._flow_to_feature_vector(flow_dict)
        x_scaled = self.scaler.transform([x])[0]
        
        # Select regime based on traffic intensity
        traffic_intensity = self._compute_traffic_intensity(flow_dict)
        
        if traffic_intensity < 0.3:
            # Regime A: low traffic → use RF
            pred = self.rf_model.predict([x_scaled])[0]
            confidence = self.rf_model.predict_proba([x_scaled]).max()
        elif traffic_intensity > 0.7:
            # Regime B: high traffic → use XGBoost
            pred = self.xgb_model.predict([x_scaled])[0]
            confidence = self.xgb_model.predict_proba([x_scaled]).max()
        else:
            # Regime C: medium/anomaly → use majority vote
            preds = [
                self.rf_model.predict([x_scaled])[0],
                self.xgb_model.predict([x_scaled])[0],
            ]
            pred = int(np.mean(preds) > 0.5)
            conf_rf = self.rf_model.predict_proba([x_scaled]).max()
            conf_xgb = self.xgb_model.predict_proba([x_scaled]).max()
            confidence = (conf_rf + conf_xgb) / 2
        
        return int(pred), float(confidence)
    
    def get_threat_score(self, flow_dict: Dict) -> float:
        """
        Get threat probability [0, 1].
        
        Returns:
            threat_score = confidence if prediction==1 else (1-confidence)
        """
        pred, confidence = self.predict(flow_dict)
        
        if pred == 1:  # Predicted as attack
            threat = confidence
        else:  # Predicted as benign
            threat = 1.0 - confidence
        
        return float(np.clip(threat, 0, 1))
    
    def explain_prediction(self, flow_dict: Dict) -> Dict[str, float]:
        """
        Get SHAP explanation for prediction.
        
        Returns:
            dict mapping feature names to SHAP values
        """
        if not self.is_trained:
            return {}
        
        x = self._flow_to_feature_vector(flow_dict)
        x_scaled = self.scaler.transform([x])
        
        # Use RF for tree SHAP
        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer.shap_values(x_scaled)
        
        # Handle both old and new SHAP formats
        if isinstance(shap_values, list):
            # Multi-class output [class0_shap, class1_shap]
            shap_vals = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            # 3D array: (samples, features, classes)
            shap_vals = shap_values[0, :, 1]
        else:
            # 2D array: (samples, features)
            shap_vals = shap_values[0]
        
        # Map to feature names
        explanation = {}
        for feature_name, shap_value in zip(self.FEATURE_COLS, shap_vals):
            explanation[feature_name] = float(shap_value)
        
        return explanation
    
    def select_regime(self, flow_dict: Dict) -> int:
        """
        Select active regime based on traffic.
        
        Returns:
            0=Regime A (low), 1=Regime B (high), 2=Regime C (anomaly)
        """
        intensity = self._compute_traffic_intensity(flow_dict)
        
        if intensity < 0.3:
            return 0
        elif intensity > 0.7:
            return 1
        else:
            return 2
    
    def _flow_to_feature_vector(self, flow_dict: Dict) -> np.ndarray:
        """Convert flow dictionary to feature vector."""
        features = {}
        
        # Extract raw features
        features['duration'] = float(flow_dict.get('duration', 0))
        features['bytes'] = float(flow_dict.get('bytes', 0))
        features['packets'] = float(flow_dict.get('packets', 0))
        features['syn_flags'] = float(flow_dict.get('syn_flags', 0))
        features['fin_flags'] = float(flow_dict.get('fin_flags', 0))
        features['rst_flags'] = float(flow_dict.get('rst_flags', 0))
        features['ack_flags'] = float(flow_dict.get('ack_flags', 0))
        
        # Derived features
        duration = max(features['duration'], 0.001)
        features['bytes_per_sec'] = features['bytes'] / duration
        features['packets_per_sec'] = features['packets'] / duration
        
        protocol = flow_dict.get('protocol', 'TCP').upper()
        features['protocol_tcp'] = 1.0 if protocol == 'TCP' else 0.0
        features['protocol_udp'] = 1.0 if protocol == 'UDP' else 0.0
        
        # Port entropy (simplified)
        features['src_port_entropy'] = 5.0  # Placeholder
        features['dst_port_entropy'] = 5.0
        features['flag_rate'] = (features['syn_flags'] + features['fin_flags'] + 
                                features['rst_flags'] + features['ack_flags']) / duration
        
        # Additional features
        features['tcp_fraction'] = features['protocol_tcp']
        features['udp_fraction'] = features['protocol_udp']
        features['icmp_fraction'] = 0.0
        features['flow_iat_mean'] = 0.1
        features['flow_iat_std'] = 0.01
        features['fwd_packets_mean'] = features['packets'] / 2
        features['bwd_packets_mean'] = features['packets'] / 2
        features['src_ports_unique'] = 1.0
        features['dst_ports_unique'] = 1.0
        features['packet_size_mean'] = features['bytes'] / max(features['packets'], 1)
        features['packet_size_std'] = 10.0
        
        # Build feature vector in column order
        vec = np.array([features.get(col, 0.0) for col in self.FEATURE_COLS], 
                      dtype=np.float32)
        
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _compute_traffic_intensity(self, flow_dict: Dict) -> float:
        """
        Compute traffic intensity [0, 1] to select regime.
        Low = normal, High = DoS/attack, Medium = mixed.
        """
        packets = float(flow_dict.get('packets', 0))
        duration = float(flow_dict.get('duration', 0.1))
        
        pps = packets / duration
        
        # Normalize: 0 pps → 0, 10000 pps → 1
        intensity = np.clip(pps / 10000, 0, 1)
        
        return intensity
