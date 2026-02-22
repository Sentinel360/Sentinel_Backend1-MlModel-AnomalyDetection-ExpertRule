"""
ML Model Inference Wrapper
Handles hybrid Ghana GB + Porto IF predictions
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
import os


class HybridMLModel:
    """
    Wrapper for hybrid ML model (Ghana GB + Porto IF)

    Fusion: 0.7 x Ghana GB + 0.3 x Porto IF
    """

    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir

        self.ghana_gb = self._load_model('ghana_gb_model.pkl')
        self.porto_if = self._load_model('porto_if_model.pkl')
        self.scaler_ghana = self._load_model('ghana_scaler.pkl')
        self.scaler_porto = self._load_model('porto_scaler.pkl')
        self.feature_names = self._load_model('feature_names.pkl')

        self.gb_weight = 0.7
        self.if_weight = 0.3

        print(f"\u2705 Loaded hybrid ML model")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Fusion: {self.gb_weight}\u00d7GB + {self.if_weight}\u00d7IF")

    def _load_model(self, filename: str):
        filepath = os.path.join(self.models_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)

    def predict(self, features: Dict[str, float]) -> Dict:
        for f in self.feature_names:
            if f not in features:
                raise ValueError(f"Missing feature: {f}")

        feature_array = pd.DataFrame(
            [[features[f] for f in self.feature_names]],
            columns=self.feature_names,
        )

        ghana_scaled = self.scaler_ghana.transform(feature_array)
        gb_score = float(self.ghana_gb.predict_proba(ghana_scaled)[0][1])

        porto_scaled = self.scaler_porto.transform(feature_array)
        if_raw = float(self.porto_if.decision_function(porto_scaled)[0])
        if_score = 1 / (1 + np.exp(-if_raw))

        hybrid_score = (self.gb_weight * gb_score) + (self.if_weight * if_score)

        if hybrid_score < 0.3:
            level, color = 'SAFE', 'green'
        elif hybrid_score < 0.7:
            level, color = 'MEDIUM', 'orange'
        else:
            level, color = 'HIGH RISK', 'red'

        return {
            'gb_score': gb_score,
            'if_score': if_score,
            'hybrid_score': hybrid_score,
            'level': level,
            'color': color,
            'components': {
                'ghana_gb': {
                    'weight': self.gb_weight,
                    'score': gb_score,
                    'contribution': self.gb_weight * gb_score,
                },
                'porto_if': {
                    'weight': self.if_weight,
                    'score': if_score,
                    'contribution': self.if_weight * if_score,
                },
            },
        }

    def batch_predict(self, feature_list: List[Dict]) -> List[Dict]:
        return [self.predict(f) for f in feature_list]

    def set_fusion_weights(self, gb_weight: float, if_weight: float):
        if not np.isclose(gb_weight + if_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
        self.gb_weight = gb_weight
        self.if_weight = if_weight
        print(f"Updated fusion weights: {gb_weight}\u00d7GB + {if_weight}\u00d7IF")
