"""
Shared fixtures for Sentinel360 test suite.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACCRA_LEGON = (5.6519, -0.1873)
ACCRA_AIRPORT = (5.6052, -0.1668)
ACCRA_CIRCLE = (5.5700, -0.2051)
ACCRA_MADINA = (5.6852, -0.1691)

NIMA_CENTER = (5.5750, -0.2050)

MODEL_FEATURE_NAMES = [
    'speed', 'acceleration', 'acceleration_variation', 'trip_duration',
    'trip_distance', 'stop_events', 'road_encoded', 'weather_encoded',
    'traffic_encoded', 'hour', 'month', 'avg_speed', 'stops_per_km',
    'accel_abs', 'speed_normalized', 'speed_squared', 'is_rush_hour',
    'is_night',
]


def make_safe_features() -> dict:
    """Normal city commute features."""
    return {
        'speed': 40.0, 'acceleration': 0.5, 'acceleration_variation': 1.2,
        'trip_duration': 900, 'trip_distance': 8.5, 'stop_events': 3,
        'road_encoded': 0, 'weather_encoded': 0, 'traffic_encoded': 1,
        'hour': 10, 'month': 2, 'avg_speed': 34.0, 'stops_per_km': 0.35,
        'accel_abs': 0.5, 'speed_normalized': 0.4, 'speed_squared': 1600,
        'is_rush_hour': 0, 'is_night': 0,
    }


def make_risky_features() -> dict:
    """Reckless late-night speeding features."""
    return {
        'speed': 130.0, 'acceleration': 6.0, 'acceleration_variation': 12.0,
        'trip_duration': 720, 'trip_distance': 18.0, 'stop_events': 2,
        'road_encoded': 0, 'weather_encoded': 0, 'traffic_encoded': 1,
        'hour': 1, 'month': 2, 'avg_speed': 90.0, 'stops_per_km': 0.11,
        'accel_abs': 6.0, 'speed_normalized': 1.3, 'speed_squared': 16900,
        'is_rush_hour': 0, 'is_night': 1,
    }


def _build_mock_gb(safe_prob=0.85):
    """Build a mock Gradient Boosting model."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[safe_prob, 1 - safe_prob]])
    return m


def _build_mock_if(raw_score=-0.5):
    """Build a mock Isolation Forest model."""
    m = MagicMock()
    m.decision_function.return_value = np.array([raw_score])
    return m


def _build_mock_scaler():
    """Build a mock scaler that returns input unchanged."""
    s = MagicMock()
    s.transform.side_effect = lambda x: x
    return s


@pytest.fixture
def mock_models(monkeypatch):
    """
    Patch joblib.load and os.path.exists so HybridMLModel.__init__
    can run without real .pkl files on disk.
    """
    gb = _build_mock_gb(safe_prob=0.85)
    if_model = _build_mock_if(raw_score=-0.5)
    scaler_gh = _build_mock_scaler()
    scaler_po = _build_mock_scaler()

    load_map = {
        'ghana_gb_model.pkl': gb,
        'porto_if_model.pkl': if_model,
        'ghana_scaler.pkl': scaler_gh,
        'porto_scaler.pkl': scaler_po,
        'feature_names.pkl': MODEL_FEATURE_NAMES,
    }

    def fake_load(path):
        for key, val in load_map.items():
            if path.endswith(key):
                return val
        raise FileNotFoundError(path)

    monkeypatch.setattr('os.path.exists', lambda p: True)
    monkeypatch.setattr('joblib.load', fake_load)

    return {
        'gb': gb, 'if': if_model,
        'scaler_gh': scaler_gh, 'scaler_po': scaler_po,
        'feature_names': MODEL_FEATURE_NAMES,
    }


@pytest.fixture
def expert_engine():
    """Fresh ExpertRulesEngine instance."""
    from core.expert_rules import ExpertRulesEngine
    return ExpertRulesEngine()


@pytest.fixture
def safe_features():
    return make_safe_features()


@pytest.fixture
def risky_features():
    return make_risky_features()
