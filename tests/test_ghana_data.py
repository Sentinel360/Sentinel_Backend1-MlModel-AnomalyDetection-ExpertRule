"""
Tests for utils/ghana_data.py

Validates Ghana-specific constants, location classification,
speed limits, and congestion zone detection.
"""

import pytest
from utils.ghana_data import (
    ACCRA_LANDMARKS,
    ACCRA_MAJOR_ROADS,
    CONGESTION_ZONES,
    KNOWN_SHORTCUTS,
    get_location_type,
    get_speed_limit,
    is_in_congestion_zone,
)


# ── Constants ────────────────────────────────────────────────────────────────

class TestAccraLandmarks:

    def test_all_landmarks_are_tuples(self):
        for name, coord in ACCRA_LANDMARKS.items():
            assert isinstance(coord, tuple), f"{name} is not a tuple"
            assert len(coord) == 2

    def test_landmarks_within_greater_accra(self):
        """All landmarks should be roughly within Greater Accra bounds."""
        for name, (lat, lon) in ACCRA_LANDMARKS.items():
            assert 5.4 < lat < 5.8, f"{name} lat {lat} out of Accra range"
            assert -0.4 < lon < 0.1, f"{name} lon {lon} out of Accra range"

    @pytest.mark.parametrize("name", [
        'legon', 'airport', 'circle', 'madina', 'osu',
        'kaneshie', 'achimota', 'tema', 'tetteh_quarshie',
    ])
    def test_expected_landmarks_present(self, name):
        assert name in ACCRA_LANDMARKS

    def test_legon_coordinates(self):
        lat, lon = ACCRA_LANDMARKS['legon']
        assert lat == pytest.approx(5.6519, abs=0.01)
        assert lon == pytest.approx(-0.1873, abs=0.01)


class TestAccraMajorRoads:

    def test_motorway_speed_limit_100(self):
        assert ACCRA_MAJOR_ROADS['accra_tema_motorway']['speed_limit'] == 100

    def test_all_roads_have_required_keys(self):
        for name, road in ACCRA_MAJOR_ROADS.items():
            assert 'type' in road, f"{name} missing 'type'"
            assert 'speed_limit' in road, f"{name} missing 'speed_limit'"
            assert 'lanes' in road, f"{name} missing 'lanes'"


class TestCongestionZones:

    def test_three_zones_defined(self):
        assert len(CONGESTION_ZONES) == 3

    def test_zones_have_peak_hours(self):
        for name, zone in CONGESTION_ZONES.items():
            assert 'peak_hours' in zone
            assert len(zone['peak_hours']) >= 1


class TestKnownShortcuts:

    def test_shortcuts_are_lists(self):
        for route, options in KNOWN_SHORTCUTS.items():
            assert isinstance(options, list)
            assert len(options) >= 2


# ── get_location_type ────────────────────────────────────────────────────────

class TestGetLocationType:

    def test_near_airport_is_motorway(self):
        assert get_location_type(5.6052, -0.1668) == 'motorway'

    def test_near_tetteh_quarshie_is_motorway(self):
        assert get_location_type(5.6086, -0.1768) == 'motorway'

    def test_legon_is_urban(self):
        assert get_location_type(5.6519, -0.1873) == 'urban'

    def test_far_from_motorway_is_urban(self):
        assert get_location_type(5.5, -0.3) == 'urban'

    def test_osu_is_urban(self):
        lat, lon = ACCRA_LANDMARKS['osu']
        assert get_location_type(lat, lon) == 'urban'


# ── get_speed_limit ──────────────────────────────────────────────────────────

class TestGetSpeedLimit:

    def test_motorway_area_returns_100(self):
        assert get_speed_limit(5.6052, -0.1668) == 100

    def test_urban_area_returns_50(self):
        assert get_speed_limit(5.6519, -0.1873) == 50

    def test_returns_positive_integer(self):
        limit = get_speed_limit(5.55, -0.20)
        assert isinstance(limit, int)
        assert limit > 0


# ── is_in_congestion_zone ────────────────────────────────────────────────────

class TestIsInCongestionZone:

    def test_circle_during_morning_rush(self):
        lat, lon = CONGESTION_ZONES['circle']['location']
        is_cong, name = is_in_congestion_zone(lat, lon, 8)
        assert is_cong is True
        assert name == 'circle'

    def test_circle_during_evening_rush(self):
        lat, lon = CONGESTION_ZONES['circle']['location']
        is_cong, name = is_in_congestion_zone(lat, lon, 18)
        assert is_cong is True

    def test_circle_at_midday_not_congested(self):
        lat, lon = CONGESTION_ZONES['circle']['location']
        is_cong, _ = is_in_congestion_zone(lat, lon, 12)
        assert is_cong is False

    def test_kaneshie_during_rush(self):
        lat, lon = CONGESTION_ZONES['kaneshie']['location']
        is_cong, name = is_in_congestion_zone(lat, lon, 7)
        assert is_cong is True
        assert name == 'kaneshie'

    def test_far_from_zones_never_congested(self):
        is_cong, _ = is_in_congestion_zone(6.0, -1.0, 8)
        assert is_cong is False

    def test_returns_empty_string_when_not_congested(self):
        _, name = is_in_congestion_zone(6.0, -1.0, 8)
        assert name == ''
