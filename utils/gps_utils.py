"""
GPS and geometric utility functions
"""

import math
import numpy as np
from typing import Tuple, List


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calculate distance between two GPS points (meters)

    Uses Haversine formula - standard in GPS/GIS applications
    """
    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_bearing(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
) -> float:
    """
    Calculate compass bearing from point1 to point2.

    Returns bearing in degrees (0-360).
    0 = North, 90 = East, 180 = South, 270 = West.
    """
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])

    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]],
) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def calculate_route_straightness(
    actual_path: List[Tuple[float, float]],
    origin: Tuple[float, float],
    destination: Tuple[float, float],
) -> float:
    """
    Calculate route straightness ratio.

    Returns ratio of direct distance to actual path distance (0-1).
    1.0 = perfectly straight, 0.0 = very winding.
    """
    direct_distance = haversine_distance(
        origin[0], origin[1], destination[0], destination[1],
    )

    path_distance = 0.0
    for i in range(len(actual_path) - 1):
        path_distance += haversine_distance(
            actual_path[i][0], actual_path[i][1],
            actual_path[i + 1][0], actual_path[i + 1][1],
        )

    if path_distance == 0:
        return 1.0

    return min(1.0, direct_distance / path_distance)


def smooth_gps_trace(
    gps_points: List[Tuple[float, float]],
    window_size: int = 3,
) -> List[Tuple[float, float]]:
    """
    Smooth GPS trace using moving average to reduce noise/jitter.
    """
    if len(gps_points) < window_size:
        return gps_points

    smoothed = []
    for i in range(len(gps_points)):
        start = max(0, i - window_size // 2)
        end = min(len(gps_points), i + window_size // 2 + 1)
        window = gps_points[start:end]
        avg_lat = np.mean([p[0] for p in window])
        avg_lon = np.mean([p[1] for p in window])
        smoothed.append((avg_lat, avg_lon))

    return smoothed
