"""
Risk Fusion Engine
Combines ML Model + Expert Rules + Route Anomaly Detection
"""

from typing import Dict, Optional
import numpy as np
from datetime import datetime

from core.ml_inference import HybridMLModel
from core.expert_rules import ExpertRulesEngine

try:
    from core.route_anomaly import RouteAnomalyDetector
    _HAS_ROUTE = True
except ImportError:
    _HAS_ROUTE = False


class RiskFusionEngine:
    """
    Master risk assessment system

    Combines:
    1. ML Model  (driving behaviour patterns)
    2. Expert Rules  (Ghana-contextualized safety rules)
    3. Route Anomaly Detection  (geographic behaviour)
    """

    def __init__(
        self,
        models_dir: str = 'models',
        google_api_key: Optional[str] = None,
    ):
        self.ml_model = HybridMLModel(models_dir=models_dir)
        self.expert_rules = ExpertRulesEngine()
        self.google_api_key = google_api_key
        self.route_detectors: Dict[str, Optional[object]] = {}

        print("\u2705 Risk Fusion Engine initialized")

    # --------------------------------------------------------- trip lifecycle

    def start_trip_monitoring(
        self, trip_id: str, origin: tuple, destination: tuple,
    ):
        if not self.google_api_key or not _HAS_ROUTE:
            if not _HAS_ROUTE:
                print("\u26a0\ufe0f Route monitoring unavailable (missing googlemaps/shapely/pyproj)")
            else:
                print("\u26a0\ufe0f No Google API key \u2014 route monitoring disabled")
            self.route_detectors[trip_id] = None
            return

        self.route_detectors[trip_id] = RouteAnomalyDetector(
            origin=origin, destination=destination,
            google_api_key=self.google_api_key,
        )
        print(f"\u2705 Started route monitoring for trip {trip_id}")

    def end_trip_monitoring(self, trip_id: str) -> Dict:
        summary: Dict = {'trip_id': trip_id, 'route_summary': None}
        det = self.route_detectors.pop(trip_id, None)
        if det is not None:
            summary['route_summary'] = det.get_trip_summary()
        return summary

    # ------------------------------------------------------------ assessment

    def assess_risk(self, trip_id: str, trip_data: Dict, context: Dict) -> Dict:
        ts = datetime.now().timestamp()

        # 1 ML
        features = self._extract_features(trip_data)
        ml_result = self.ml_model.predict(features)
        ml_score = ml_result['hybrid_score']

        # 2 Expert Rules
        rules_result = self.expert_rules.apply_all_rules(trip_data, context)
        rules_adj = rules_result['total_risk_adjustment']

        # 3 Route Anomaly
        route_result: Dict = {'status': 'NO_ROUTE_MONITORING', 'risk_adjustment': 0.0}
        det = self.route_detectors.get(trip_id)
        if det is not None:
            gps = context.get('current_location', (0, 0))
            route_result = det.update(gps, ts)
        route_adj = route_result.get('risk_adjustment', 0.0)

        # Fusion
        final = max(0.0, min(1.0, ml_score + route_adj + rules_adj))

        if final < 0.3:
            level, color = 'SAFE', 'green'
        elif final < 0.7:
            level, color = 'MEDIUM', 'orange'
        else:
            level, color = 'HIGH RISK', 'red'

        actions: list = list(rules_result.get('triggered_actions', []))
        if route_result.get('action'):
            actions.append(route_result['action'])
        if level == 'HIGH RISK' and not actions:
            actions.append('ALERT_USER')
        elif level == 'MEDIUM' and not actions:
            actions.append('MONITOR')

        return {
            'trip_id': trip_id, 'timestamp': ts,
            'final_score': final, 'final_level': level, 'final_color': color,
            'components': {
                'ml_model': {
                    'score': ml_score, 'level': ml_result['level'],
                    'gb_score': ml_result['gb_score'],
                    'if_score': ml_result['if_score'],
                },
                'expert_rules': {
                    'adjustment': rules_adj,
                    'critical_rules': rules_result['critical_rules'],
                    'severity_rules': rules_result['severity_rules'],
                    'context_adjustments': rules_result['context_adjustments'],
                },
                'route_anomaly': {
                    'adjustment': route_adj,
                    'status': route_result['status'],
                    'deviation_distance': route_result.get('deviation_distance', 0),
                },
            },
            'actions': list(set(actions)),
            'explanation': self._explain(ml_result, rules_result, route_result, final),
        }

    # ------------------------------------------------------------ features

    def _extract_features(self, trip_data: Dict) -> Dict[str, float]:
        defaults = {
            'avg_speed': 0.0, 'max_speed': 0.0, 'speed_std': 0.0,
            'avg_acceleration': 0.0, 'max_acceleration': 0.0,
            'harsh_accel_count': 0, 'harsh_brake_count': 0,
            'stop_count': 0, 'avg_stop_duration': 0.0,
            'total_distance': 0.0, 'distance_per_stop': 0.0,
            'time_of_day': 12, 'day_of_week': 3, 'trip_duration': 0.0,
            'speed_changes': 0, 'route_straightness': 0.85,
            'idle_time_ratio': 0.0, 'avg_trip_speed': 0.0,
        }
        return {**defaults, **trip_data.get('features', {})}

    # ------------------------------------------------------------ explain

    @staticmethod
    def _explain(ml, rules, route, final) -> str:
        parts = [f"Driving behaviour: {ml['level']} (score: {ml['hybrid_score']:.2f})"]
        for r in rules.get('critical_rules', []):
            parts.append(f"\u26a0\ufe0f {r['reason']}")
        if route.get('triggered'):
            parts.append(f"\U0001f5fa\ufe0f {route.get('message', route['status'])}")
        for r in rules.get('severity_rules', []):
            parts.append(f"\u26a1 {r['reason']}")
        tr = rules.get('context_adjustments', {}).get('time_risk', {})
        if tr.get('multiplier', 1.0) > 1.5:
            parts.append(f"\u23f0 High-risk time (x{tr['multiplier']})")
        parts.append(
            f"Final: {final:.2f} "
            f"(ML {ml['hybrid_score']:.2f} + Rules {rules['total_risk_adjustment']:.2f} "
            f"+ Route {route.get('risk_adjustment', 0):.2f})"
        )
        return " | ".join(parts)
