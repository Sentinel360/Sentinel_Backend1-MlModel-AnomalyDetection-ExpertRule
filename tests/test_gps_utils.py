"""
Tests for utils/gps_utils.py

Validates Haversine distance, bearing calculations, point-in-polygon,
route straightness, and GPS smoothing against known geographic data.
"""

import math
import pytest
from utils.gps_utils import (
    haversine_distance,
    calculate_bearing,
    point_in_polygon,
    calculate_route_straightness,
    smooth_gps_trace,
)

ACCRA_LEGON = (5.6519, -0.1873)
ACCRA_AIRPORT = (5.6052, -0.1668)
ACCRA_CIRCLE = (5.5700, -0.2051)


# ── haversine_distance ──────────────────────────────────────────────────────

class TestHaversineDistance:

    def test_same_point_returns_zero(self):
        d = haversine_distance(5.65, -0.19, 5.65, -0.19)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_legon_to_airport_approx_5_7km(self):
        """Known real-world distance between University of Ghana and Kotoka Airport."""
        d = haversine_distance(*ACCRA_LEGON, *ACCRA_AIRPORT)
        assert 5000 < d < 6500, f"Expected ~5.7 km, got {d:.0f} m"

    def test_legon_to_circle_approx_9_2km(self):
        d = haversine_distance(*ACCRA_LEGON, *ACCRA_CIRCLE)
        assert 8500 < d < 10000, f"Expected ~9.2 km, got {d:.0f} m"

    def test_symmetry(self):
        d1 = haversine_distance(*ACCRA_LEGON, *ACCRA_AIRPORT)
        d2 = haversine_distance(*ACCRA_AIRPORT, *ACCRA_LEGON)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_short_distance_accuracy(self):
        """~111 m for 0.001 degree latitude near equator."""
        d = haversine_distance(5.65, -0.19, 5.651, -0.19)
        assert 100 < d < 120

    @pytest.mark.parametrize("lat1,lon1,lat2,lon2", [
        (0, 0, 0, 0),
        (90, 0, -90, 0),
    ])
    def test_extreme_coordinates(self, lat1, lon1, lat2, lon2):
        d = haversine_distance(lat1, lon1, lat2, lon2)
        assert d >= 0

    def test_poles_half_circumference(self):
        """North pole to south pole ≈ 20,015 km."""
        d = haversine_distance(90, 0, -90, 0)
        assert 19_900_000 < d < 20_100_000


# ── calculate_bearing ────────────────────────────────────────────────────────

class TestCalculateBearing:

    def test_due_north(self):
        bearing = calculate_bearing((0.0, 0.0), (1.0, 0.0))
        assert bearing == pytest.approx(0.0, abs=0.5)

    def test_due_east(self):
        bearing = calculate_bearing((0.0, 0.0), (0.0, 1.0))
        assert bearing == pytest.approx(90.0, abs=0.5)

    def test_due_south(self):
        bearing = calculate_bearing((1.0, 0.0), (0.0, 0.0))
        assert bearing == pytest.approx(180.0, abs=0.5)

    def test_due_west(self):
        bearing = calculate_bearing((0.0, 0.0), (0.0, -1.0))
        assert bearing == pytest.approx(270.0, abs=0.5)

    def test_bearing_range_0_to_360(self):
        for lon in range(-180, 180, 30):
            b = calculate_bearing((0, 0), (0.01, lon / 100))
            assert 0 <= b < 360

    def test_same_point_bearing_is_zero(self):
        b = calculate_bearing((5.65, -0.19), (5.65, -0.19))
        assert 0 <= b < 360

    def test_northeast_roughly_45_degrees(self):
        b = calculate_bearing((0.0, 0.0), (1.0, 1.0))
        assert 40 < b < 50


# ── point_in_polygon ─────────────────────────────────────────────────────────

class TestPointInPolygon:

    @pytest.fixture
    def square(self):
        return [(0, 0), (0, 10), (10, 10), (10, 0)]

    def test_center_inside(self, square):
        assert point_in_polygon((5, 5), square) is True

    def test_outside_point(self, square):
        assert point_in_polygon((15, 15), square) is False

    def test_negative_outside(self, square):
        assert point_in_polygon((-1, -1), square) is False

    def test_triangle(self):
        tri = [(0, 0), (5, 10), (10, 0)]
        assert point_in_polygon((5, 3), tri) is True
        assert point_in_polygon((0, 10), tri) is False

    def test_accra_bounding_box(self):
        """Rough Accra bounding polygon."""
        accra_box = [
            (5.50, -0.30), (5.50, -0.10),
            (5.70, -0.10), (5.70, -0.30),
        ]
        assert point_in_polygon((5.60, -0.20), accra_box) is True
        assert point_in_polygon((6.00, -0.20), accra_box) is False


# ── calculate_route_straightness ─────────────────────────────────────────────

class TestRouteStrightness:

    def test_straight_line_returns_one(self):
        origin = (0.0, 0.0)
        dest = (1.0, 0.0)
        path = [origin, (0.5, 0.0), dest]
        s = calculate_route_straightness(path, origin, dest)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_detour_less_than_one(self):
        origin = (0.0, 0.0)
        dest = (1.0, 0.0)
        path = [origin, (0.5, 0.5), dest]
        s = calculate_route_straightness(path, origin, dest)
        assert 0 < s < 1.0

    def test_empty_path_returns_one(self):
        origin = (0.0, 0.0)
        dest = (1.0, 0.0)
        s = calculate_route_straightness([], origin, dest)
        assert s == 1.0

    def test_single_point_path(self):
        origin = (0.0, 0.0)
        dest = (1.0, 0.0)
        s = calculate_route_straightness([origin], origin, dest)
        assert s == 1.0

    def test_highly_winding_route(self):
        origin = (0.0, 0.0)
        dest = (0.01, 0.0)
        path = [origin, (0.005, 0.02), (0.01, -0.02), dest]
        s = calculate_route_straightness(path, origin, dest)
        assert s < 0.5


# ── smooth_gps_trace ─────────────────────────────────────────────────────────

class TestSmoothGpsTrace:

    def test_short_trace_returned_unchanged(self):
        pts = [(1.0, 2.0), (1.1, 2.1)]
        assert smooth_gps_trace(pts, window_size=5) == pts

    def test_preserves_length(self):
        pts = [(i * 0.001, 0.0) for i in range(20)]
        smoothed = smooth_gps_trace(pts, window_size=3)
        assert len(smoothed) == len(pts)

    def test_reduces_noise(self):
        pts = [(0.0, 0.0), (0.001, 0.0), (0.0, 0.0), (0.001, 0.0), (0.0, 0.0)]
        smoothed = smooth_gps_trace(pts, window_size=3)
        lats = [p[0] for p in smoothed]
        orig_lats = [p[0] for p in pts]
        assert max(lats) - min(lats) <= max(orig_lats) - min(orig_lats)

    def test_constant_trace_unchanged(self):
        pts = [(5.0, -0.2)] * 10
        smoothed = smooth_gps_trace(pts, window_size=5)
        for s in smoothed:
            assert s[0] == pytest.approx(5.0, abs=1e-10)
            assert s[1] == pytest.approx(-0.2, abs=1e-10)

    def test_window_size_one(self):
        pts = [(1.0, 2.0), (3.0, 4.0)]
        smoothed = smooth_gps_trace(pts, window_size=1)
        assert len(smoothed) == 2
