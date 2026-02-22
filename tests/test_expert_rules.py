"""
Tests for core/expert_rules.py

Validates every rule category in ExpertRulesEngine:
critical, severity, context, correlation, whitelist, and the aggregate.
"""

import time
import pytest
from core.expert_rules import ExpertRulesEngine

NIMA_CENTER = (5.5750, -0.2050)
OUTSIDE_ZONES = (5.70, -0.10)


@pytest.fixture
def engine():
    return ExpertRulesEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: rule_extreme_speeding
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleExtremeSpeeding:

    def test_above_130_triggers_critical(self, engine):
        r = engine.rule_extreme_speeding(135, 100, 'motorway')
        assert r['triggered'] is True
        assert r['severity'] == 'CRITICAL'
        assert r['rule_name'] == 'EXTREME_SPEEDING'
        assert r['risk_adjustment'] == 0.50

    def test_50_over_limit_triggers_critical(self, engine):
        r = engine.rule_extreme_speeding(100, 50, 'urban')
        assert r['triggered'] is True
        assert r['rule_name'] == 'EXCESSIVE_SPEEDING'

    def test_school_zone_above_35_triggers(self, engine):
        r = engine.rule_extreme_speeding(40, 20, 'school_zone')
        assert r['triggered'] is True
        assert r['rule_name'] == 'SCHOOL_ZONE_SPEEDING'
        assert r['risk_adjustment'] == 0.45

    def test_school_zone_at_35_does_not_trigger(self, engine):
        r = engine.rule_extreme_speeding(35, 20, 'school_zone')
        assert r['triggered'] is False

    def test_normal_speed_does_not_trigger(self, engine):
        r = engine.rule_extreme_speeding(45, 50, 'urban')
        assert r['triggered'] is False

    def test_slightly_over_limit_no_trigger(self, engine):
        r = engine.rule_extreme_speeding(65, 50, 'urban')
        assert r['triggered'] is False

    @pytest.mark.parametrize("speed,limit,loc,expected_triggered", [
        (131, 100, 'motorway', True),
        (129, 100, 'motorway', False),
        (110, 50, 'urban', True),
        (90, 50, 'urban', False),
        (36, 20, 'school_zone', True),
    ])
    def test_boundary_conditions(self, engine, speed, limit, loc, expected_triggered):
        r = engine.rule_extreme_speeding(speed, limit, loc)
        assert r['triggered'] is expected_triggered


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: rule_crash_pattern
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleCrashPattern:

    def test_harsh_brake_plus_stop_triggers(self, engine):
        history = [0.0] * 9 + [-8.0]
        r = engine.rule_crash_pattern(history, current_speed=0)
        assert r['triggered'] is True
        assert r['severity'] == 'CRITICAL'
        assert r['risk_adjustment'] == 0.60

    def test_harsh_brake_but_still_moving_no_trigger(self, engine):
        history = [0.0] * 9 + [-8.0]
        r = engine.rule_crash_pattern(history, current_speed=30)
        assert r['triggered'] is False

    def test_mild_brake_plus_stop_no_trigger(self, engine):
        history = [0.0] * 9 + [-3.0]
        r = engine.rule_crash_pattern(history, current_speed=0)
        assert r['triggered'] is False

    def test_insufficient_history_no_trigger(self, engine):
        r = engine.rule_crash_pattern([-10.0] * 5, current_speed=0)
        assert r['triggered'] is False

    def test_empty_history(self, engine):
        r = engine.rule_crash_pattern([], current_speed=0)
        assert r['triggered'] is False

    def test_exactly_minus_7_boundary(self, engine):
        """At exactly -7.0 it does NOT trigger (< -7.0 required)."""
        history = [0.0] * 9 + [-7.0]
        r = engine.rule_crash_pattern(history, current_speed=0)
        assert r['triggered'] is False


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: rule_geofence_violation
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleGeofenceViolation:

    def test_nima_at_midnight_triggers(self, engine):
        r = engine.rule_geofence_violation(NIMA_CENTER, time_of_day=0)
        assert r['triggered'] is True
        assert r['rule_name'] == 'GEOFENCE_VIOLATION'
        assert r['risk_adjustment'] == 0.35

    def test_nima_at_23_triggers(self, engine):
        r = engine.rule_geofence_violation(NIMA_CENTER, time_of_day=23)
        assert r['triggered'] is True

    def test_nima_at_midday_does_not_trigger(self, engine):
        """Nima risk hours are 22-5. Hour 12 should not trigger."""
        r = engine.rule_geofence_violation(NIMA_CENTER, time_of_day=12)
        assert r['triggered'] is False

    def test_outside_all_zones(self, engine):
        r = engine.rule_geofence_violation(OUTSIDE_ZONES, time_of_day=0)
        assert r['triggered'] is False

    def test_chorkor_at_night_triggers(self, engine):
        chorkor_center = (5.5450, -0.2550)
        r = engine.rule_geofence_violation(chorkor_center, time_of_day=23)
        assert r['triggered'] is True


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY: rule_sustained_speeding
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleSustainedSpeeding:

    def test_10_seconds_all_over_triggers(self, engine):
        history = [80.0] * 10
        r = engine.rule_sustained_speeding(history, speed_limit=50)
        assert r['triggered'] is True
        assert r['risk_adjustment'] == 0.25

    def test_9_of_10_over_does_not_trigger(self, engine):
        history = [80.0] * 9 + [50.0]
        r = engine.rule_sustained_speeding(history, speed_limit=50)
        assert r['triggered'] is False

    def test_only_slightly_over_does_not_trigger(self, engine):
        """Must be >20 km/h over limit, not just over."""
        history = [65.0] * 10
        r = engine.rule_sustained_speeding(history, speed_limit=50)
        assert r['triggered'] is False

    def test_short_history(self, engine):
        r = engine.rule_sustained_speeding([90] * 5, speed_limit=50)
        assert r['triggered'] is False


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY: rule_harsh_event_cluster
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleHarshEventCluster:

    def test_three_recent_events_triggers(self, engine):
        now = time.time()
        events = [{'timestamp': now - i} for i in range(3)]
        r = engine.rule_harsh_event_cluster(events, time_window=60)
        assert r['triggered'] is True
        assert r['risk_adjustment'] == 0.20

    def test_two_events_does_not_trigger(self, engine):
        now = time.time()
        events = [{'timestamp': now - i} for i in range(2)]
        r = engine.rule_harsh_event_cluster(events)
        assert r['triggered'] is False

    def test_old_events_outside_window(self, engine):
        now = time.time()
        events = [{'timestamp': now - 120} for _ in range(5)]
        r = engine.rule_harsh_event_cluster(events, time_window=60)
        assert r['triggered'] is False

    def test_empty_events(self, engine):
        r = engine.rule_harsh_event_cluster([])
        assert r['triggered'] is False


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT: rule_rush_hour_context
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleRushHourContext:

    def test_morning_rush_weekday_circle(self, engine):
        r = engine.rule_rush_hour_context(time_of_day=8, day_of_week=1, location='Circle')
        assert r['context'] == 'RUSH_HOUR_CONGESTION'
        assert r['adjustments']['ml_score_damper'] == -0.15

    def test_evening_rush_weekday_kaneshie(self, engine):
        r = engine.rule_rush_hour_context(time_of_day=18, day_of_week=3, location='Kaneshie')
        assert r['context'] == 'RUSH_HOUR_CONGESTION'

    def test_weekend_no_rush(self, engine):
        r = engine.rule_rush_hour_context(time_of_day=8, day_of_week=6, location='Circle')
        assert r['context'] == 'NORMAL'

    def test_midday_no_rush(self, engine):
        r = engine.rule_rush_hour_context(time_of_day=12, day_of_week=1, location='Circle')
        assert r['context'] == 'NORMAL'

    def test_non_rush_location(self, engine):
        r = engine.rule_rush_hour_context(time_of_day=8, day_of_week=1, location='Legon')
        assert r['context'] == 'NORMAL'


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT: rule_time_risk_multiplier
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleTimeRiskMultiplier:

    def test_late_night_highest_risk(self, engine):
        r = engine.rule_time_risk_multiplier(23)
        assert r['multiplier'] == 3.2
        assert r['ml_adjustment'] == pytest.approx((3.2 - 1.0) * 0.15)

    def test_early_morning_high_risk(self, engine):
        r = engine.rule_time_risk_multiplier(3)
        assert r['multiplier'] == 2.5

    def test_midday_baseline(self, engine):
        r = engine.rule_time_risk_multiplier(11)
        assert r['multiplier'] == 1.0
        assert r['ml_adjustment'] == pytest.approx(0.0)

    def test_early_afternoon_safest(self, engine):
        r = engine.rule_time_risk_multiplier(15)
        assert r['multiplier'] == 0.9
        assert r['ml_adjustment'] < 0


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION RULES
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrelationRules:

    def test_fast_in_residential_triggers(self, engine):
        r = engine.rule_speed_location_correlation(50, 'residential', 12)
        assert r['triggered'] is True
        assert r['risk_adjustment'] >= 0.15

    def test_fast_in_school_zone_during_hours(self, engine):
        r = engine.rule_speed_location_correlation(40, 'school_zone', 10)
        assert r['triggered'] is True
        assert r['risk_adjustment'] >= 0.25

    def test_slow_on_motorway_triggers(self, engine):
        r = engine.rule_speed_location_correlation(30, 'motorway', 12)
        assert r['triggered'] is True
        assert r['risk_adjustment'] >= 0.10

    def test_normal_speed_no_trigger(self, engine):
        r = engine.rule_speed_location_correlation(45, 'urban', 12)
        assert r['triggered'] is False

    def test_route_deviation_plus_long_duration(self, engine):
        r = engine.rule_route_deviation_duration(
            route_deviation=500, trip_duration=1500, expected_duration=800,
        )
        assert r['triggered'] is True
        assert r['risk_adjustment'] == 0.20


# ═══════════════════════════════════════════════════════════════════════════════
# WHITELIST RULES
# ═══════════════════════════════════════════════════════════════════════════════

class TestWhitelistRules:

    def test_professional_driver_gets_adjustment(self, engine):
        history = {'trips_completed': 600, 'rating': 4.9, 'safety_incidents_90d': 0}
        r = engine.rule_professional_driver_profile(history)
        assert r['profile'] == 'PROFESSIONAL'
        assert r['ml_adjustment'] == -0.10

    def test_new_driver_standard(self, engine):
        history = {'trips_completed': 10, 'rating': 4.5, 'safety_incidents_90d': 0}
        r = engine.rule_professional_driver_profile(history)
        assert r['profile'] == 'STANDARD'
        assert r['ml_adjustment'] == 0.0

    def test_known_safe_route_reduces_score(self, engine):
        route_history = {'legon_to_airport': {'count': 20, 'incidents': 0}}
        r = engine.rule_known_safe_route('legon_to_airport', 'driver1', route_history)
        assert r['context'] == 'KNOWN_SAFE_ROUTE'
        assert r['ml_adjustment'] == -0.05

    def test_unknown_route_no_adjustment(self, engine):
        r = engine.rule_known_safe_route('random_route', 'driver1', {})
        assert r['context'] == 'NORMAL'
        assert r['ml_adjustment'] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATE: apply_all_rules
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyAllRules:

    def test_returns_required_keys(self, engine):
        r = engine.apply_all_rules({}, {})
        for key in ['critical_rules', 'severity_rules', 'context_adjustments',
                     'correlation_rules', 'whitelist_adjustments',
                     'total_risk_adjustment', 'triggered_actions']:
            assert key in r, f"Missing key: {key}"

    def test_speeding_triggers_critical_in_aggregate(self, engine):
        trip = {'current_speed': 140}
        ctx = {'speed_limit': 50, 'location_type': 'urban'}
        r = engine.apply_all_rules(trip, ctx)
        assert len(r['critical_rules']) >= 1
        assert r['total_risk_adjustment'] >= 0.50
        assert 'IMMEDIATE_ALERT' in r['triggered_actions']

    def test_safe_driving_minimal_adjustment(self, engine):
        trip = {'current_speed': 40}
        ctx = {'speed_limit': 50, 'location_type': 'urban', 'time_of_day': 12}
        r = engine.apply_all_rules(trip, ctx)
        assert len(r['critical_rules']) == 0
        assert len(r['severity_rules']) == 0
        assert abs(r['total_risk_adjustment']) < 0.2

    def test_professional_driver_reduces_total(self, engine):
        trip = {'current_speed': 40}
        ctx = {
            'speed_limit': 50, 'time_of_day': 12,
            'driver_history': {'trips_completed': 600, 'rating': 4.9, 'safety_incidents_90d': 0},
        }
        r = engine.apply_all_rules(trip, ctx)
        assert r['whitelist_adjustments']['driver_profile']['ml_adjustment'] == -0.10

    def test_unpaved_road_damper_applied(self, engine):
        trip = {'current_speed': 30}
        ctx = {'road_type': 'unpaved', 'time_of_day': 12}
        r = engine.apply_all_rules(trip, ctx)
        assert r['total_risk_adjustment'] < 0, "Unpaved damper should reduce score"

    def test_rush_hour_damper_applied(self, engine):
        trip = {'current_speed': 30}
        ctx = {'time_of_day': 8, 'day_of_week': 1, 'location': 'Circle', 'road_type': 'arterial'}
        r = engine.apply_all_rules(trip, ctx)
        rush = r['context_adjustments']['rush_hour']
        assert rush['context'] == 'RUSH_HOUR_CONGESTION'
