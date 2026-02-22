"""
Ghana-specific data and constants
"""

from typing import Tuple

ACCRA_LANDMARKS = {
    'legon': (5.6519, -0.1873),
    'airport': (5.6052, -0.1668),
    'circle': (5.5700, -0.2051),
    'madina': (5.6852, -0.1691),
    'osu': (5.5597, -0.1826),
    'kaneshie': (5.5619, -0.2313),
    'achimota': (5.6253, -0.2314),
    'tema': (5.6698, 0.0117),
    'tetteh_quarshie': (5.6086, -0.1768),
}

ACCRA_MAJOR_ROADS = {
    'accra_tema_motorway': {
        'type': 'motorway',
        'speed_limit': 100,
        'lanes': 3,
    },
    'ring_road_west': {
        'type': 'arterial',
        'speed_limit': 50,
        'lanes': 2,
    },
    'spintex_road': {
        'type': 'arterial',
        'speed_limit': 50,
        'lanes': 2,
    },
    'independence_avenue': {
        'type': 'urban',
        'speed_limit': 50,
        'lanes': 2,
    },
}

CONGESTION_ZONES = {
    'circle': {
        'location': (5.5700, -0.2051),
        'radius': 500,
        'peak_hours': [(6, 9), (17, 20)],
    },
    'kaneshie': {
        'location': (5.5619, -0.2313),
        'radius': 600,
        'peak_hours': [(6, 9), (17, 20)],
    },
    'madina': {
        'location': (5.6852, -0.1691),
        'radius': 400,
        'peak_hours': [(6, 9), (17, 20)],
    },
}

KNOWN_SHORTCUTS = {
    'airport_to_legon': [
        'via Madina (Google primary)',
        'via Shiashie (faster during rush)',
        'via Dome (avoid motorway)',
    ],
    'circle_to_tema': [
        'via Motorway (fastest)',
        'via Spintex (avoid tolls)',
        'via Lashibi (scenic route)',
    ],
}


def get_location_type(lat: float, lon: float) -> str:
    """
    Classify location type based on coordinates.

    Returns 'motorway', 'arterial', 'residential', etc.
    """
    from utils.gps_utils import haversine_distance

    motorway_coords = [
        (5.6052, -0.1668),
        (5.6086, -0.1768),
    ]

    for coord in motorway_coords:
        if haversine_distance(lat, lon, coord[0], coord[1]) < 1000:
            return 'motorway'

    return 'urban'


def get_speed_limit(lat: float, lon: float) -> int:
    """Get speed limit for location (km/h)."""
    location_type = get_location_type(lat, lon)
    speed_limits = {
        'motorway': 100,
        'highway': 80,
        'arterial': 50,
        'urban': 50,
        'residential': 30,
        'school_zone': 20,
    }
    return speed_limits.get(location_type, 50)


def is_in_congestion_zone(
    lat: float,
    lon: float,
    time_of_day: int,
) -> Tuple[bool, str]:
    """
    Check if location is in known congestion zone during peak hours.

    Returns (is_congested, zone_name).
    """
    from utils.gps_utils import haversine_distance

    for zone_name, zone_data in CONGESTION_ZONES.items():
        zone_lat, zone_lon = zone_data['location']
        distance = haversine_distance(lat, lon, zone_lat, zone_lon)

        if distance < zone_data['radius']:
            for start, end in zone_data['peak_hours']:
                if start <= time_of_day < end:
                    return (True, zone_name)

    return (False, '')
