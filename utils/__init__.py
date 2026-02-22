"""
Utility functions
"""

from utils.gps_utils import (
    haversine_distance,
    calculate_bearing,
    point_in_polygon,
    calculate_route_straightness,
    smooth_gps_trace,
)

from utils.ghana_data import (
    ACCRA_LANDMARKS,
    get_location_type,
    get_speed_limit,
    is_in_congestion_zone,
)

from utils import config

__all__ = [
    'haversine_distance',
    'calculate_bearing',
    'point_in_polygon',
    'calculate_route_straightness',
    'smooth_gps_trace',
    'ACCRA_LANDMARKS',
    'get_location_type',
    'get_speed_limit',
    'is_in_congestion_zone',
    'config',
]
