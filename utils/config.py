"""
Configuration constants
"""

import os

# Model configuration
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
FUSION_GB_WEIGHT = float(os.getenv('FUSION_GB_WEIGHT', '0.7'))
FUSION_IF_WEIGHT = float(os.getenv('FUSION_IF_WEIGHT', '0.3'))

# Risk thresholds
SAFE_THRESHOLD = 0.3
MEDIUM_THRESHOLD = 0.7

# Route anomaly detection
ROUTE_BUFFER_DISTANCE = int(os.getenv('ROUTE_BUFFER_DISTANCE', '100'))
ROUTE_DEVIATION_CRITICAL = int(os.getenv('ROUTE_DEVIATION_CRITICAL', '500'))
ROUTE_REROUTE_CHECK_INTERVAL = int(os.getenv('ROUTE_REROUTE_CHECK_INTERVAL', '30'))

# Google Maps API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

# Alert thresholds
CONSECUTIVE_HIGH_RISK_THRESHOLD = int(os.getenv('CONSECUTIVE_HIGH_RISK', '3'))
ALERT_COOLDOWN_PERIOD = int(os.getenv('ALERT_COOLDOWN', '30'))

# Feature extraction
FEATURE_WINDOW_SIZE = int(os.getenv('FEATURE_WINDOW_SIZE', '10'))

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Feature names (must match training)
FEATURE_NAMES = [
    'avg_speed',
    'max_speed',
    'speed_std',
    'avg_acceleration',
    'max_acceleration',
    'harsh_accel_count',
    'harsh_brake_count',
    'stop_count',
    'avg_stop_duration',
    'total_distance',
    'distance_per_stop',
    'time_of_day',
    'day_of_week',
    'trip_duration',
    'speed_changes',
    'route_straightness',
    'idle_time_ratio',
    'avg_trip_speed',
]
