"""Real-Time Hybrid Monitor - Ghana GB + Porto IF Fusion"""
import sys, os
import numpy as np
import joblib
import warnings
from collections import deque, defaultdict
from config import *

warnings.filterwarnings('ignore')

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci


class HybridMonitor:
    """Hybrid model: 0.7 x Ghana GB + 0.3 x Porto IF fusion"""

    def __init__(self):
        self.ghana_gb = None
        self.porto_if = None
        self.ghana_scaler = None
        self.porto_scaler = None
        self.features = None
        self.fusion_config = {}
        self.use_rule_based = False

        try:
            self.ghana_gb = joblib.load(GHANA_GB_FILE)
            print(f"\u2705 Ghana GB loaded: {type(self.ghana_gb).__name__}")
        except Exception as e:
            print(f"\u26a0\ufe0f  Ghana GB failed: {str(e)[:60]}")

        try:
            self.porto_if = joblib.load(PORTO_IF_FILE)
            print(f"\u2705 Porto IF loaded: {type(self.porto_if).__name__}")
        except Exception as e:
            print(f"\u26a0\ufe0f  Porto IF failed: {str(e)[:60]}")

        try:
            self.ghana_scaler = joblib.load(GHANA_SCALER_FILE)
            self.porto_scaler = joblib.load(PORTO_SCALER_FILE)
            print(f"\u2705 Scalers loaded (Ghana: {type(self.ghana_scaler).__name__}, Porto: {type(self.porto_scaler).__name__})")
        except Exception as e:
            print(f"\u26a0\ufe0f  Scaler loading failed: {str(e)[:60]}")

        try:
            self.features = joblib.load(FEATURES_FILE)
            self.fusion_config = joblib.load(FUSION_CONFIG_FILE)
            print(f"\u2705 Features loaded ({len(self.features)} features)")
        except Exception as e:
            print(f"\u26a0\ufe0f  Features/config failed: {str(e)[:60]}")
            self.fusion_config = {}

        self.weight_gb = self.fusion_config.get('weight_gb', 0.7)
        self.weight_if = self.fusion_config.get('weight_if', 0.3)

        if self.ghana_gb and self.porto_if:
            self.mode = 'hybrid'
            print(f"\u2705 Hybrid Monitor ready")
            print(f"   Mode: HYBRID FUSION")
        elif self.porto_if:
            self.mode = 'porto_only'
            self.weight_if = 1.0
            self.weight_gb = 0.0
            print(f"\u2705 Monitor ready (Porto IF only)")
        elif self.ghana_gb:
            self.mode = 'ghana_only'
            self.weight_gb = 1.0
            self.weight_if = 0.0
            print(f"\u2705 Monitor ready (Ghana GB only)")
        else:
            self.mode = 'rule_based'
            self.use_rule_based = True
            print(f"\u26a0\ufe0f  No ML models loaded, using rule-based")

        print(f"   Fusion: {self.weight_gb}\u00d7GB + {self.weight_if}\u00d7IF")

        self.vehicles = defaultdict(lambda: {
            'speeds': deque(maxlen=10), 'positions': deque(maxlen=10),
            'accels': deque(maxlen=10), 'distance': 0.0, 'stops': 0,
            'last_speed': 0, 'trip_start': None, 'risk_score': 0.0, 'risk_level': 'SAFE'
        })

    def haversine(self, pos1, pos2):
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [*pos1, *pos2])
        a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
        return 6371 * 2 * asin(sqrt(a))

    def update_vehicle(self, veh_id, speed, position, timestamp):
        data = self.vehicles[veh_id]
        if data['trip_start'] is None:
            data['trip_start'] = timestamp
        speed_kmh = speed * 3.6
        data['speeds'].append(speed_kmh)
        data['positions'].append(position)
        if len(data['speeds']) >= 2:
            data['accels'].append(data['speeds'][-1] - data['speeds'][-2])
        if len(data['positions']) >= 2:
            data['distance'] += self.haversine(data['positions'][-2], data['positions'][-1])
        if speed_kmh < 1.0 and data['last_speed'] > 2.0:
            data['stops'] += 1
        data['last_speed'] = speed_kmh

    def _build_features(self, data, trip_duration):
        return {
            'speed': data['speeds'][-1],
            'acceleration': data['accels'][-1] if data['accels'] else 0,
            'acceleration_variation': np.std(data['accels']) if len(data['accels']) > 1 else 0,
            'trip_duration': trip_duration,
            'trip_distance': data['distance'],
            'stop_events': data['stops'],
            'road_encoded': 0,
            'weather_encoded': 0,
            'traffic_encoded': 1,
            'hour': 10,
            'month': 2,
            'avg_speed': data['distance'] / (trip_duration / 3600 + 0.001),
            'stops_per_km': data['stops'] / (data['distance'] + 0.1),
            'accel_abs': abs(data['accels'][-1]) if data['accels'] else 0,
            'speed_normalized': data['speeds'][-1] / 100,
            'speed_squared': data['speeds'][-1] ** 2,
            'is_rush_hour': 0,
            'is_night': 0,
        }

    def predict_risk(self, veh_id, timestamp):
        data = self.vehicles[veh_id]
        if len(data['speeds']) < 5:
            return 0.0, 'SAFE', COLORS['SAFE']

        trip_duration = timestamp - data['trip_start']

        if self.use_rule_based:
            return self._rule_based_risk(data, trip_duration)

        try:
            features = self._build_features(data, trip_duration)
            feature_array = [features[f] for f in self.features]
            risk_score = 0.0

            if self.ghana_gb and self.ghana_scaler and self.weight_gb > 0:
                ghana_scaled = self.ghana_scaler.transform([feature_array])
                gb_score = self.ghana_gb.predict_proba(ghana_scaled)[0][1]
                risk_score += self.weight_gb * gb_score

            if self.porto_if and self.porto_scaler and self.weight_if > 0:
                porto_scaled = self.porto_scaler.transform([feature_array])
                if_raw = self.porto_if.decision_function(porto_scaled)[0]
                if_score = 1 / (1 + np.exp(-if_raw))
                risk_score += self.weight_if * if_score

        except Exception:
            return self._rule_based_risk(data, trip_duration)

        threshold_safe = self.fusion_config.get('threshold_safe', 0.3)
        threshold_high = self.fusion_config.get('threshold_high', 0.7)

        if risk_score < threshold_safe:
            risk_level, color = 'SAFE', COLORS['SAFE']
        elif risk_score < threshold_high:
            risk_level, color = 'MEDIUM', COLORS['MEDIUM']
        else:
            risk_level, color = 'HIGH', COLORS['HIGH']

        data['risk_score'] = risk_score
        data['risk_level'] = risk_level
        return risk_score, risk_level, color

    def _rule_based_risk(self, data, trip_duration):
        risk_score = self._rule_based_risk_score(data, trip_duration)
        threshold_safe = self.fusion_config.get('threshold_safe', 0.3) if self.fusion_config else 0.3
        threshold_high = self.fusion_config.get('threshold_high', 0.7) if self.fusion_config else 0.7

        if risk_score < threshold_safe:
            risk_level, color = 'SAFE', COLORS['SAFE']
        elif risk_score < threshold_high:
            risk_level, color = 'MEDIUM', COLORS['MEDIUM']
        else:
            risk_level, color = 'HIGH', COLORS['HIGH']

        data['risk_score'] = risk_score
        data['risk_level'] = risk_level
        return risk_score, risk_level, color

    def _rule_based_risk_score(self, data, trip_duration):
        risk_score = 0.0
        current_speed = data['speeds'][-1]
        if current_speed > 80:
            risk_score += 0.3
        elif current_speed > 60:
            risk_score += 0.15

        if data['accels']:
            recent_accel = abs(data['accels'][-1])
            if recent_accel > 5.0:
                risk_score += 0.25
            elif recent_accel > 3.0:
                risk_score += 0.15
            accel_std = np.std(data['accels'])
            if accel_std > 4.0:
                risk_score += 0.2
            elif accel_std > 2.5:
                risk_score += 0.1

        trip_duration_valid = max(1, trip_duration)
        stops_per_min = (data['stops'] / trip_duration_valid) * 60
        if stops_per_min > 2.0:
            risk_score += 0.15
        elif stops_per_min > 1.0:
            risk_score += 0.08

        return min(1.0, risk_score)

    def get_statistics(self):
        if not self.vehicles:
            return None
        risk_counts = {'SAFE': 0, 'MEDIUM': 0, 'HIGH': 0}
        for data in self.vehicles.values():
            risk_counts[data['risk_level']] += 1
        return {
            'total': len(self.vehicles),
            'safe': risk_counts['SAFE'],
            'medium': risk_counts['MEDIUM'],
            'high': risk_counts['HIGH'],
            'avg_risk': np.mean([d['risk_score'] for d in self.vehicles.values()])
        }
