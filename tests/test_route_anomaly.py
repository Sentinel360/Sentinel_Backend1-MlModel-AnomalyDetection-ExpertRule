"""
Tests for core/route_anomaly.py

Validates static geo helpers (haversine, bearing, closest-point),
deviation evaluation logic, and trip summary.
Google Maps / Shapely / pyproj interactions are fully mocked.
"""

import math
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from core.route_anomaly import RouteAnomalyDetector

LEGON = (5.6519, -0.1873)
AIRPORT = (5.6052, -0.1668)


# ═══════════════════════════════════════════════════════════════════════════════
# Static helpers (no external deps)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHaversineDistance:

    def test_same_point(self):
        d = RouteAnomalyDetector.haversine_distance(5.65, -0.19, 5.65, -0.19)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_known_distance(self):
        d = RouteAnomalyDetector.haversine_distance(*LEGON, *AIRPORT)
        assert 5000 < d < 6500

    def test_symmetry(self):
        d1 = RouteAnomalyDetector.haversine_distance(*LEGON, *AIRPORT)
        d2 = RouteAnomalyDetector.haversine_distance(*AIRPORT, *LEGON)
        assert d1 == pytest.approx(d2, rel=1e-9)


class TestCalculateBearing:

    def test_north(self):
        b = RouteAnomalyDetector.calculate_bearing((0, 0), (1, 0))
        assert b == pytest.approx(0.0, abs=1.0)

    def test_east(self):
        b = RouteAnomalyDetector.calculate_bearing((0, 0), (0, 1))
        assert b == pytest.approx(90.0, abs=1.0)

    def test_south(self):
        b = RouteAnomalyDetector.calculate_bearing((1, 0), (0, 0))
        assert b == pytest.approx(180.0, abs=1.0)

    def test_range(self):
        for lat_off in [-1, 0, 1]:
            for lon_off in [-1, 0, 1]:
                if lat_off == 0 and lon_off == 0:
                    continue
                b = RouteAnomalyDetector.calculate_bearing((0, 0), (lat_off, lon_off))
                assert 0 <= b < 360


class TestClosestPointOnSegment:

    def test_midpoint_projection(self):
        cp = RouteAnomalyDetector._closest_point_on_segment(
            (5, 0), (0, 0), (10, 0),
        )
        assert cp[0] == pytest.approx(5.0)
        assert cp[1] == pytest.approx(0.0)

    def test_before_segment_clamps_to_start(self):
        cp = RouteAnomalyDetector._closest_point_on_segment(
            (-5, 0), (0, 0), (10, 0),
        )
        assert cp == (0, 0)

    def test_after_segment_clamps_to_end(self):
        cp = RouteAnomalyDetector._closest_point_on_segment(
            (15, 0), (0, 0), (10, 0),
        )
        assert cp[0] == pytest.approx(10.0)

    def test_degenerate_segment(self):
        cp = RouteAnomalyDetector._closest_point_on_segment(
            (3, 4), (5, 5), (5, 5),
        )
        assert cp == (5, 5)


# ═══════════════════════════════════════════════════════════════════════════════
# distance_from_route (uses polyline, no external deps)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDistanceFromRoute:

    @pytest.fixture
    def detector(self):
        """Build a bare detector with attributes set manually (no __init__)."""
        det = object.__new__(RouteAnomalyDetector)
        det.origin = LEGON
        det.destination = AIRPORT
        det.routes = []
        det.route_corridors = []
        det.gps_breadcrumbs = []
        det.deviation_events = []
        det.consecutive_deviations = 0
        return det

    def test_on_route_returns_zero_distance(self, detector):
        polyline = [(0, 0), (0, 0.001), (0, 0.002)]
        result = detector.distance_from_route((0, 0.001), polyline)
        assert result['distance'] < 1.0

    def test_off_route_returns_positive(self, detector):
        polyline = [(0, 0), (1, 0)]
        result = detector.distance_from_route((0.5, 0.5), polyline)
        assert result['distance'] > 1000

    def test_result_has_progress(self, detector):
        polyline = [(0, 0), (0.5, 0), (1, 0)]
        result = detector.distance_from_route((0.5, 0), polyline)
        assert 0 <= result['progress'] <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# _evaluate_deviation (pure logic, no deps)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateDeviation:

    @pytest.fixture
    def detector(self):
        det = object.__new__(RouteAnomalyDetector)
        return det

    def test_wrong_direction_critical(self, detector):
        r = detector._evaluate_deviation(400, 3, wrong_direction=True, timestamp=0)
        assert r['status'] == 'WRONG_DIRECTION'
        assert r['risk_adjustment'] == 0.60
        assert r['triggered'] is True

    def test_wrong_direction_close_not_critical(self, detector):
        r = detector._evaluate_deviation(100, 3, wrong_direction=True, timestamp=0)
        assert r['status'] != 'WRONG_DIRECTION'

    def test_large_sustained_deviation(self, detector):
        r = detector._evaluate_deviation(600, 5, wrong_direction=False, timestamp=0)
        assert r['status'] == 'CRITICAL_DEVIATION'
        assert r['risk_adjustment'] == 0.40

    def test_moderate_sustained_deviation(self, detector):
        r = detector._evaluate_deviation(250, 10, wrong_direction=False, timestamp=0)
        assert r['status'] == 'HIGH_DEVIATION'
        assert r['risk_adjustment'] == 0.25

    def test_prolonged_minor_deviation(self, detector):
        r = detector._evaluate_deviation(160, 20, wrong_direction=False, timestamp=0)
        assert r['status'] == 'PROLONGED_DEVIATION'
        assert r['risk_adjustment'] == 0.15

    def test_minor_deviation(self, detector):
        r = detector._evaluate_deviation(50, 1, wrong_direction=False, timestamp=0)
        assert r['status'] == 'MINOR_DEVIATION'
        assert r['risk_adjustment'] == 0.05
        assert r['triggered'] is False

    @pytest.mark.parametrize("dist,consec,wrong,expected_status", [
        (600, 5, False, 'CRITICAL_DEVIATION'),
        (600, 4, False, 'MINOR_DEVIATION'),
        (499, 5, False, 'MINOR_DEVIATION'),
        (300, 10, False, 'HIGH_DEVIATION'),
        (250, 10, False, 'HIGH_DEVIATION'),
        (199, 10, False, 'MINOR_DEVIATION'),
    ])
    def test_boundary_conditions(self, detector, dist, consec, wrong, expected_status):
        r = detector._evaluate_deviation(dist, consec, wrong, 0)
        assert r['status'] == expected_status


# ═══════════════════════════════════════════════════════════════════════════════
# get_trip_summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTripSummary:

    @pytest.fixture
    def detector(self):
        det = object.__new__(RouteAnomalyDetector)
        det.gps_breadcrumbs = []
        det.deviation_events = []
        return det

    def test_empty_returns_none(self, detector):
        assert detector.get_trip_summary() is None

    def test_no_deviations(self, detector):
        detector.gps_breadcrumbs = [{'position': (0, 0), 'timestamp': i} for i in range(10)]
        s = detector.get_trip_summary()
        assert s['total_points'] == 10
        assert s['deviation_events'] == 0
        assert s['deviation_ratio'] == 0
        assert s['was_anomalous'] is False

    def test_high_deviation_ratio_is_anomalous(self, detector):
        detector.gps_breadcrumbs = [{'position': (0, 0), 'timestamp': i} for i in range(10)]
        detector.deviation_events = [{'distance': 100, 'timestamp': i, 'position': (0, 0)} for i in range(5)]
        s = detector.get_trip_summary()
        assert s['deviation_ratio'] == 0.5
        assert s['was_anomalous'] is True

    def test_large_max_deviation_is_anomalous(self, detector):
        detector.gps_breadcrumbs = [{'position': (0, 0), 'timestamp': 0}]
        detector.deviation_events = [{'distance': 1500, 'timestamp': 0, 'position': (0, 0)}]
        s = detector.get_trip_summary()
        assert s['max_deviation'] == 1500
        assert s['was_anomalous'] is True
