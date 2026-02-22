"""
Tests for core/risk_fusion.py

Validates RiskFusionEngine: component integration, score fusion,
action determination, trip lifecycle, and explanation generation.
ML models are fully mocked.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from tests.conftest import MODEL_FEATURE_NAMES, make_safe_features


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mock_ml_model(hybrid_score=0.2, level='SAFE', gb=0.1, if_s=0.4):
    ml = MagicMock()
    ml.feature_names = MODEL_FEATURE_NAMES
    ml.predict.return_value = {
        'hybrid_score': hybrid_score, 'level': level,
        'gb_score': gb, 'if_score': if_s,
        'color': 'green',
        'components': {
            'ghana_gb': {'weight': 0.7, 'score': gb, 'contribution': 0.7 * gb},
            'porto_if': {'weight': 0.3, 'score': if_s, 'contribution': 0.3 * if_s},
        },
    }
    return ml


@pytest.fixture
def fusion_engine(monkeypatch):
    """RiskFusionEngine with mocked ML model (no pkl files needed)."""
    mock_ml = _build_mock_ml_model()

    monkeypatch.setattr(
        'core.risk_fusion.HybridMLModel',
        lambda models_dir: mock_ml,
    )

    from core.risk_fusion import RiskFusionEngine
    engine = RiskFusionEngine(models_dir='fake')
    engine._mock_ml = mock_ml
    return engine


# ═══════════════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_creates_engine(self, fusion_engine):
        assert fusion_engine.ml_model is not None
        assert fusion_engine.expert_rules is not None

    def test_no_api_key_disables_route(self, fusion_engine):
        assert fusion_engine.google_api_key is None


# ═══════════════════════════════════════════════════════════════════════════════
# assess_risk: basic
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssessRisk:

    def test_returns_required_keys(self, fusion_engine):
        result = fusion_engine.assess_risk('trip1', {}, {})
        for key in ['trip_id', 'timestamp', 'final_score', 'final_level',
                     'final_color', 'components', 'actions', 'explanation']:
            assert key in result, f"Missing key: {key}"

    def test_safe_ml_gives_safe_level(self, fusion_engine):
        result = fusion_engine.assess_risk('trip1', {}, {'time_of_day': 12})
        assert result['final_level'] == 'SAFE'
        assert result['final_color'] == 'green'

    def test_high_ml_gives_high_risk(self, fusion_engine):
        fusion_engine._mock_ml.predict.return_value = {
            'hybrid_score': 0.85, 'level': 'HIGH RISK',
            'gb_score': 0.9, 'if_score': 0.7, 'color': 'red',
            'components': {
                'ghana_gb': {'weight': 0.7, 'score': 0.9, 'contribution': 0.63},
                'porto_if': {'weight': 0.3, 'score': 0.7, 'contribution': 0.21},
            },
        }
        result = fusion_engine.assess_risk('trip1', {}, {'time_of_day': 12})
        assert result['final_level'] == 'HIGH RISK'
        assert 'ALERT_USER' in result['actions']

    def test_score_clamped_to_zero_one(self, fusion_engine):
        fusion_engine._mock_ml.predict.return_value = {
            'hybrid_score': 0.9, 'level': 'HIGH RISK',
            'gb_score': 0.95, 'if_score': 0.8, 'color': 'red',
            'components': {
                'ghana_gb': {'weight': 0.7, 'score': 0.95, 'contribution': 0.665},
                'porto_if': {'weight': 0.3, 'score': 0.8, 'contribution': 0.24},
            },
        }
        result = fusion_engine.assess_risk(
            'trip1',
            {'current_speed': 140},
            {'speed_limit': 50, 'location_type': 'urban', 'time_of_day': 23},
        )
        assert 0 <= result['final_score'] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Expert rules integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpertRulesIntegration:

    def test_speeding_adds_to_final_score(self, fusion_engine):
        result = fusion_engine.assess_risk(
            'trip1',
            {'current_speed': 140},
            {'speed_limit': 50, 'location_type': 'urban', 'time_of_day': 12},
        )
        assert result['final_score'] > 0.2
        assert 'IMMEDIATE_ALERT' in result['actions']

    def test_late_night_time_risk_adds(self, fusion_engine):
        result_day = fusion_engine.assess_risk('t1', {}, {'time_of_day': 12})
        result_night = fusion_engine.assess_risk('t2', {}, {'time_of_day': 23})
        assert result_night['final_score'] > result_day['final_score']

    def test_professional_driver_reduces_score(self, fusion_engine):
        driver = {'trips_completed': 600, 'rating': 4.9, 'safety_incidents_90d': 0}
        r_std = fusion_engine.assess_risk('t1', {}, {'time_of_day': 12})
        r_pro = fusion_engine.assess_risk('t2', {}, {'time_of_day': 12, 'driver_history': driver})
        assert r_pro['final_score'] <= r_std['final_score']


# ═══════════════════════════════════════════════════════════════════════════════
# Route anomaly integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteAnomalyIntegration:

    def test_no_route_monitor_returns_status(self, fusion_engine):
        result = fusion_engine.assess_risk('trip1', {}, {})
        assert result['components']['route_anomaly']['status'] == 'NO_ROUTE_MONITORING'

    def test_start_trip_without_api_key(self, fusion_engine):
        fusion_engine.start_trip_monitoring('trip1', (5.65, -0.19), (5.60, -0.17))
        assert fusion_engine.route_detectors.get('trip1') is None


# ═══════════════════════════════════════════════════════════════════════════════
# Trip lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestTripLifecycle:

    def test_start_and_end(self, fusion_engine):
        fusion_engine.start_trip_monitoring('trip1', (5.65, -0.19), (5.60, -0.17))
        summary = fusion_engine.end_trip_monitoring('trip1')
        assert summary['trip_id'] == 'trip1'
        assert 'trip1' not in fusion_engine.route_detectors

    def test_end_nonexistent_trip(self, fusion_engine):
        summary = fusion_engine.end_trip_monitoring('nonexistent')
        assert summary['trip_id'] == 'nonexistent'
        assert summary['route_summary'] is None

    def test_multiple_trips(self, fusion_engine):
        for tid in ['t1', 't2', 't3']:
            fusion_engine.start_trip_monitoring(tid, (5.65, -0.19), (5.60, -0.17))
        assert len(fusion_engine.route_detectors) == 3
        fusion_engine.end_trip_monitoring('t2')
        assert 't2' not in fusion_engine.route_detectors
        assert 't1' in fusion_engine.route_detectors


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_features
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractFeatures:

    def test_defaults_when_no_features(self, fusion_engine):
        features = fusion_engine._extract_features({})
        assert features['avg_speed'] == 0.0
        assert features['time_of_day'] == 12
        assert features['route_straightness'] == 0.85

    def test_overrides_with_provided_data(self, fusion_engine):
        trip = {'features': {'avg_speed': 55.0, 'time_of_day': 8}}
        features = fusion_engine._extract_features(trip)
        assert features['avg_speed'] == 55.0
        assert features['time_of_day'] == 8
        assert features['trip_duration'] == 0.0

    def test_returns_18_keys(self, fusion_engine):
        features = fusion_engine._extract_features({})
        assert len(features) == 18


# ═══════════════════════════════════════════════════════════════════════════════
# Explanation
# ═══════════════════════════════════════════════════════════════════════════════

class TestExplanation:

    def test_explanation_is_string(self, fusion_engine):
        result = fusion_engine.assess_risk('trip1', {}, {'time_of_day': 12})
        assert isinstance(result['explanation'], str)
        assert len(result['explanation']) > 10

    def test_explanation_contains_ml_score(self, fusion_engine):
        result = fusion_engine.assess_risk('trip1', {}, {'time_of_day': 12})
        assert 'Driving behaviour' in result['explanation']

    def test_explanation_contains_final(self, fusion_engine):
        result = fusion_engine.assess_risk('trip1', {}, {'time_of_day': 12})
        assert 'Final' in result['explanation']

    def test_critical_rule_appears_in_explanation(self, fusion_engine):
        result = fusion_engine.assess_risk(
            'trip1',
            {'current_speed': 140},
            {'speed_limit': 50, 'location_type': 'urban', 'time_of_day': 12},
        )
        assert 'Extreme speed' in result['explanation'] or 'Excessive' in result['explanation']


# ═══════════════════════════════════════════════════════════════════════════════
# Action determination
# ═══════════════════════════════════════════════════════════════════════════════

class TestActions:

    def test_medium_score_gets_monitor(self, fusion_engine):
        fusion_engine._mock_ml.predict.return_value = {
            'hybrid_score': 0.5, 'level': 'MEDIUM',
            'gb_score': 0.5, 'if_score': 0.5, 'color': 'orange',
            'components': {
                'ghana_gb': {'weight': 0.7, 'score': 0.5, 'contribution': 0.35},
                'porto_if': {'weight': 0.3, 'score': 0.5, 'contribution': 0.15},
            },
        }
        result = fusion_engine.assess_risk('trip1', {}, {'time_of_day': 12})
        assert 'MONITOR' in result['actions']

    def test_safe_score_no_alert(self, fusion_engine):
        result = fusion_engine.assess_risk('trip1', {}, {'time_of_day': 12})
        assert 'ALERT_USER' not in result['actions']
        assert 'MONITOR' not in result['actions']
