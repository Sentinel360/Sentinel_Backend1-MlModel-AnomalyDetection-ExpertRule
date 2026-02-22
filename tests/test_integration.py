"""
End-to-end integration tests for Sentinel360.

Simulates realistic trip scenarios through the full
RiskFusionEngine pipeline (ML + ExpertRules), validating that
the system produces coherent, expected outputs for various
real-world driving patterns.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from tests.conftest import MODEL_FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_ml(hybrid_score, gb_score, if_score, level):
    ml = MagicMock()
    ml.feature_names = MODEL_FEATURE_NAMES
    color = {'SAFE': 'green', 'MEDIUM': 'orange', 'HIGH RISK': 'red'}[level]
    ml.predict.return_value = {
        'hybrid_score': hybrid_score, 'level': level,
        'gb_score': gb_score, 'if_score': if_score, 'color': color,
        'components': {
            'ghana_gb': {'weight': 0.7, 'score': gb_score, 'contribution': 0.7 * gb_score},
            'porto_if': {'weight': 0.3, 'score': if_score, 'contribution': 0.3 * if_score},
        },
    }
    return ml


@pytest.fixture
def safe_engine(monkeypatch):
    ml = _mock_ml(0.15, 0.05, 0.38, 'SAFE')
    monkeypatch.setattr('core.risk_fusion.HybridMLModel', lambda models_dir: ml)
    from core.risk_fusion import RiskFusionEngine
    return RiskFusionEngine(models_dir='fake')


@pytest.fixture
def medium_engine(monkeypatch):
    ml = _mock_ml(0.50, 0.55, 0.38, 'MEDIUM')
    monkeypatch.setattr('core.risk_fusion.HybridMLModel', lambda models_dir: ml)
    from core.risk_fusion import RiskFusionEngine
    return RiskFusionEngine(models_dir='fake')


@pytest.fixture
def high_engine(monkeypatch):
    ml = _mock_ml(0.85, 0.95, 0.62, 'HIGH RISK')
    monkeypatch.setattr('core.risk_fusion.HybridMLModel', lambda models_dir: ml)
    from core.risk_fusion import RiskFusionEngine
    return RiskFusionEngine(models_dir='fake')


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Normal safe commute
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalSafeTrip:

    def test_safe_commute_classification(self, safe_engine):
        """Weekday morning commute at normal speed should be SAFE."""
        trip = {'current_speed': 40}
        ctx = {
            'speed_limit': 50, 'location_type': 'urban',
            'time_of_day': 10, 'day_of_week': 2,
            'road_type': 'arterial',
        }
        result = safe_engine.assess_risk('safe_trip', trip, ctx)
        assert result['final_level'] == 'SAFE'
        assert result['final_score'] < 0.3

    def test_no_critical_rules_triggered(self, safe_engine):
        trip = {'current_speed': 40}
        ctx = {'speed_limit': 50, 'time_of_day': 10}
        result = safe_engine.assess_risk('safe_trip', trip, ctx)
        assert len(result['components']['expert_rules']['critical_rules']) == 0

    def test_no_alert_actions(self, safe_engine):
        result = safe_engine.assess_risk('safe_trip', {}, {'time_of_day': 10})
        assert 'ALERT_USER' not in result['actions']
        assert 'IMMEDIATE_ALERT' not in result['actions']


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Speeding violation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedingViolation:

    def test_extreme_speed_triggers_alert(self, high_engine):
        trip = {'current_speed': 140}
        ctx = {'speed_limit': 50, 'location_type': 'urban', 'time_of_day': 12}
        result = high_engine.assess_risk('speeding', trip, ctx)
        assert result['final_level'] == 'HIGH RISK'
        assert 'IMMEDIATE_ALERT' in result['actions']

    def test_school_zone_speeding(self, medium_engine):
        trip = {'current_speed': 40}
        ctx = {'speed_limit': 20, 'location_type': 'school_zone', 'time_of_day': 10}
        result = medium_engine.assess_risk('school', trip, ctx)
        assert len(result['components']['expert_rules']['critical_rules']) >= 1
        assert result['final_score'] > 0.5

    def test_highway_speed_acceptable(self, safe_engine):
        """90 km/h on motorway (limit 100) should remain safe."""
        trip = {'current_speed': 90}
        ctx = {'speed_limit': 100, 'location_type': 'motorway', 'time_of_day': 14}
        result = safe_engine.assess_risk('highway', trip, ctx)
        assert len(result['components']['expert_rules']['critical_rules']) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Late-night risk
# ═══════════════════════════════════════════════════════════════════════════════

class TestLateNightRisk:

    def test_late_night_elevates_score(self, safe_engine):
        """Even safe ML score at 11 PM should be nudged upward."""
        day = safe_engine.assess_risk('day', {}, {'time_of_day': 12})
        night = safe_engine.assess_risk('night', {}, {'time_of_day': 23})
        assert night['final_score'] > day['final_score']

    def test_late_night_multiplier_in_context(self, safe_engine):
        result = safe_engine.assess_risk('night', {}, {'time_of_day': 23})
        time_ctx = result['components']['expert_rules']['context_adjustments']['time_risk']
        assert time_ctx['multiplier'] == 3.2


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: Crash pattern
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrashPattern:

    def test_emergency_brake_plus_stop(self, safe_engine):
        accel_history = [0.0] * 9 + [-9.0]
        trip = {
            'current_speed': 0,
            'acceleration_history': accel_history,
        }
        ctx = {'speed_limit': 50, 'time_of_day': 14}
        result = safe_engine.assess_risk('crash', trip, ctx)
        crits = result['components']['expert_rules']['critical_rules']
        assert any(r['rule_name'] == 'CRASH_PATTERN' for r in crits)
        assert 'IMMEDIATE_ALERT_AND_CALL_USER' in result['actions']


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 5: Professional driver whitelist
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfessionalDriver:

    def test_pro_driver_gets_lower_score(self, medium_engine):
        ctx_std = {'time_of_day': 12}
        ctx_pro = {
            'time_of_day': 12,
            'driver_history': {
                'trips_completed': 800, 'rating': 4.85,
                'safety_incidents_90d': 0,
            },
        }
        r_std = medium_engine.assess_risk('std', {}, ctx_std)
        r_pro = medium_engine.assess_risk('pro', {}, ctx_pro)
        assert r_pro['final_score'] < r_std['final_score']


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 6: Rush-hour congestion
# ═══════════════════════════════════════════════════════════════════════════════

class TestRushHourCongestion:

    def test_rush_hour_damper_applied(self, medium_engine):
        ctx = {
            'time_of_day': 8, 'day_of_week': 1,
            'location': 'Circle', 'road_type': 'arterial',
        }
        result = medium_engine.assess_risk('rush', {}, ctx)
        rush = result['components']['expert_rules']['context_adjustments']['rush_hour']
        assert rush['context'] == 'RUSH_HOUR_CONGESTION'


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 7: Combined violations
# ═══════════════════════════════════════════════════════════════════════════════

class TestCombinedViolations:

    def test_speeding_plus_late_night_in_nima(self, high_engine):
        """Multiple concurrent risk factors should push score toward 1.0."""
        nima = (5.5750, -0.2050)
        trip = {'current_speed': 140}
        ctx = {
            'speed_limit': 50, 'location_type': 'urban',
            'time_of_day': 23, 'current_location': nima,
        }
        result = high_engine.assess_risk('combined', trip, ctx)
        assert result['final_score'] >= 0.9
        assert result['final_level'] == 'HIGH RISK'
        assert len(result['actions']) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 8: Full trip lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullTripLifecycle:

    def test_start_assess_end(self, safe_engine):
        safe_engine.start_trip_monitoring('lifecycle', (5.65, -0.19), (5.60, -0.17))

        for step in range(5):
            result = safe_engine.assess_risk(
                'lifecycle', {'current_speed': 40},
                {'speed_limit': 50, 'time_of_day': 10},
            )
            assert 'final_score' in result

        summary = safe_engine.end_trip_monitoring('lifecycle')
        assert summary['trip_id'] == 'lifecycle'

    def test_multiple_concurrent_trips(self, safe_engine):
        for tid in ['t1', 't2', 't3']:
            safe_engine.start_trip_monitoring(tid, (5.65, -0.19), (5.60, -0.17))

        for tid in ['t1', 't2', 't3']:
            r = safe_engine.assess_risk(tid, {}, {'time_of_day': 12})
            assert r['trip_id'] == tid

        for tid in ['t1', 't2', 't3']:
            s = safe_engine.end_trip_monitoring(tid)
            assert s['trip_id'] == tid


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 9: Unpaved road context
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnpavedRoadContext:

    def test_unpaved_reduces_final_score(self, medium_engine):
        ctx_paved = {'road_type': 'arterial', 'time_of_day': 12}
        ctx_unpaved = {'road_type': 'unpaved', 'time_of_day': 12}
        r_paved = medium_engine.assess_risk('paved', {}, ctx_paved)
        r_unpaved = medium_engine.assess_risk('unpaved', {}, ctx_unpaved)
        assert r_unpaved['final_score'] < r_paved['final_score']


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 10: Explanation quality
# ═══════════════════════════════════════════════════════════════════════════════

class TestExplanationQuality:

    def test_safe_explanation_short(self, safe_engine):
        r = safe_engine.assess_risk('t', {}, {'time_of_day': 12})
        assert 'SAFE' in r['explanation']

    def test_critical_explanation_includes_reason(self, high_engine):
        r = high_engine.assess_risk(
            't', {'current_speed': 140},
            {'speed_limit': 50, 'location_type': 'urban', 'time_of_day': 12},
        )
        assert 'Extreme speed' in r['explanation'] or 'Excessive' in r['explanation']

    def test_late_night_explanation_includes_time(self, safe_engine):
        r = safe_engine.assess_risk('t', {}, {'time_of_day': 23})
        assert 'High-risk time' in r['explanation']
