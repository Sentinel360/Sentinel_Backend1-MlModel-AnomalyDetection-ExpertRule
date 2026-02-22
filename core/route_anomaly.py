"""
Route Anomaly Detection System
Detects when driver deviates from expected routes

Requires optional dependencies: googlemaps, shapely, pyproj
Install with:  pip install googlemaps shapely pyproj
"""

import math
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional

try:
    import googlemaps
    import pyproj
    from shapely.geometry import Point, LineString, Polygon
    from shapely.ops import transform
    _HAS_GEO_DEPS = True
except ImportError:
    _HAS_GEO_DEPS = False


class RouteAnomalyDetector:
    """
    Real-time route deviation detection.

    Requires googlemaps, shapely, and pyproj at runtime.
    """

    def __init__(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        google_api_key: str,
        buffer_distance: int = 100,
    ):
        if not _HAS_GEO_DEPS:
            raise ImportError(
                "RouteAnomalyDetector requires googlemaps, shapely, and pyproj. "
                "Install with: pip install googlemaps shapely pyproj"
            )

        self.origin = origin
        self.destination = destination
        self.api_key = google_api_key
        self.buffer_distance = buffer_distance

        self.gmaps = googlemaps.Client(key=self.api_key)

        self.routes = self._fetch_all_routes()
        self.primary_route = self.routes[0] if self.routes else None
        self.alternative_routes = self.routes[1:] if len(self.routes) > 1 else []

        self.route_corridors = self._create_route_corridors()

        self.trip_start_time = datetime.now().timestamp()
        self.last_reroute_check = self.trip_start_time
        self.gps_breadcrumbs: List[Dict] = []
        self.deviation_events: List[Dict] = []
        self.consecutive_deviations = 0

    # ------------------------------------------------------------------ fetch

    def _fetch_all_routes(self) -> List[Dict]:
        try:
            directions = self.gmaps.directions(
                origin=self.origin, destination=self.destination,
                mode='driving', departure_time='now',
                alternatives=True, traffic_model='best_guess',
            )
            if not directions:
                raise Exception("No routes found from Google Maps")

            routes = []
            for rd in directions:
                poly = googlemaps.convert.decode_polyline(rd['overview_polyline']['points'])
                routes.append({
                    'polyline': poly,
                    'distance': rd['legs'][0]['distance']['value'],
                    'duration': rd['legs'][0]['duration']['value'],
                    'summary': rd['summary'],
                    'bounds': rd['bounds'],
                })
            return routes
        except Exception as e:
            print(f"Error fetching routes: {e}")
            return []

    # -------------------------------------------------------------- corridors

    def _create_route_corridors(self) -> List:
        corridors = []
        wgs84 = pyproj.CRS('EPSG:4326')
        utm = pyproj.CRS('EPSG:32630')
        to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
        to_wgs = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True)

        for route in self.routes:
            line = LineString([(p[1], p[0]) for p in route['polyline']])
            line_utm = transform(to_utm.transform, line)
            buf = line_utm.buffer(self.buffer_distance)
            corridors.append(transform(to_wgs.transform, buf))
        return corridors

    # ---------------------------------------------------------------- reroute

    def check_for_reroutes(self, current_position: Tuple[float, float]):
        now = datetime.now().timestamp()
        if now - self.last_reroute_check < 30:
            return {'rerouted': False}
        try:
            data = self.gmaps.directions(
                origin=current_position, destination=self.destination,
                mode='driving', departure_time='now',
                alternatives=True, traffic_model='best_guess',
            )
            if data:
                updated = []
                for rd in data:
                    poly = googlemaps.convert.decode_polyline(rd['overview_polyline']['points'])
                    updated.append({
                        'polyline': poly,
                        'distance': rd['legs'][0]['distance']['value'],
                        'duration': rd['legs'][0]['duration']['value'],
                        'summary': rd['summary'],
                    })
                self.routes = updated
                self.route_corridors = self._create_route_corridors()
                self.last_reroute_check = now
                return {'rerouted': True, 'new_route': updated[0]['summary']}
        except Exception as e:
            print(f"Reroute check failed: {e}")
        self.last_reroute_check = now
        return {'rerouted': False}

    # ----------------------------------------------------------- geo helpers

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2) -> float:
        R = 6371000
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def distance_from_route(self, current_pos, route_polyline) -> Dict:
        min_d = float('inf')
        closest = None
        seg_idx = 0
        clat, clon = current_pos
        for i in range(len(route_polyline) - 1):
            cp = self._closest_point_on_segment(current_pos, route_polyline[i], route_polyline[i + 1])
            d = self.haversine_distance(clat, clon, cp[0], cp[1])
            if d < min_d:
                min_d, closest, seg_idx = d, cp, i
        return {
            'distance': min_d, 'closest_point': closest,
            'segment_index': seg_idx,
            'progress': seg_idx / max(len(route_polyline) - 1, 1),
        }

    @staticmethod
    def _closest_point_on_segment(point, seg_start, seg_end):
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return seg_start
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        return (x1 + t * dx, y1 + t * dy)

    @staticmethod
    def calculate_bearing(p1, p2) -> float:
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dl = lon2 - lon1
        x = math.sin(dl) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dl)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def is_within_any_corridor(self, current_pos) -> Tuple[bool, Optional[int]]:
        pt = Point(current_pos[1], current_pos[0])
        for idx, corridor in enumerate(self.route_corridors):
            if corridor.contains(pt):
                return (True, idx)
        return (False, None)

    # ---------------------------------------------------------------- update

    def update(self, current_gps: Tuple[float, float], timestamp: float) -> Dict:
        self.gps_breadcrumbs.append({'position': current_gps, 'timestamp': timestamp})
        self.check_for_reroutes(current_gps)

        within, ridx = self.is_within_any_corridor(current_gps)
        if within:
            self.consecutive_deviations = 0
            name = self.routes[ridx]['summary']
            return {
                'status': 'ON_PRIMARY_ROUTE' if ridx == 0 else 'ON_ALTERNATIVE_ROUTE',
                'route': name, 'deviation_distance': 0.0,
                'risk_adjustment': 0.0, 'triggered': False,
            }

        min_info, min_dist = None, float('inf')
        for route in self.routes:
            info = self.distance_from_route(current_gps, route['polyline'])
            if info['distance'] < min_dist:
                min_dist = info['distance']
                min_info = info

        self.consecutive_deviations += 1
        self.deviation_events.append({
            'timestamp': timestamp, 'distance': min_dist, 'position': current_gps,
        })

        wrong_dir = False
        if len(self.gps_breadcrumbs) >= 2:
            bearing_dest = self.calculate_bearing(current_gps, self.destination)
            heading = self.calculate_bearing(self.gps_breadcrumbs[-2]['position'], current_gps)
            diff = abs(bearing_dest - heading)
            if diff > 180:
                diff = 360 - diff
            wrong_dir = diff > 90

        return self._evaluate_deviation(min_dist, self.consecutive_deviations, wrong_dir, timestamp)

    def _evaluate_deviation(self, distance, consecutive, wrong_direction, timestamp) -> Dict:
        if wrong_direction and distance > 300:
            return {
                'status': 'WRONG_DIRECTION', 'deviation_distance': distance,
                'consecutive_seconds': consecutive, 'risk_adjustment': 0.60,
                'triggered': True, 'severity': 'CRITICAL',
                'action': 'ALERT_USER_IMMEDIATELY',
                'message': 'Driver heading away from destination!',
            }
        if distance > 500 and consecutive >= 5:
            return {
                'status': 'CRITICAL_DEVIATION', 'deviation_distance': distance,
                'consecutive_seconds': consecutive, 'risk_adjustment': 0.40,
                'triggered': True, 'severity': 'CRITICAL',
                'action': 'ALERT_USER_IMMEDIATELY',
                'message': f'{distance:.0f}m off route for {consecutive}s',
            }
        if distance > 200 and consecutive >= 10:
            return {
                'status': 'HIGH_DEVIATION', 'deviation_distance': distance,
                'consecutive_seconds': consecutive, 'risk_adjustment': 0.25,
                'triggered': True, 'severity': 'HIGH',
                'action': 'CHECK_IN_WITH_USER',
                'message': 'Driver taking different route. Are you okay?',
            }
        if distance > 150 and consecutive >= 20:
            return {
                'status': 'PROLONGED_DEVIATION', 'deviation_distance': distance,
                'consecutive_seconds': consecutive, 'risk_adjustment': 0.15,
                'triggered': True, 'severity': 'MEDIUM',
                'action': 'MONITOR', 'message': 'Route deviation detected',
            }
        return {
            'status': 'MINOR_DEVIATION', 'deviation_distance': distance,
            'consecutive_seconds': consecutive, 'risk_adjustment': 0.05,
            'triggered': False, 'severity': 'LOW', 'action': None,
        }

    def get_trip_summary(self) -> Optional[Dict]:
        if not self.gps_breadcrumbs:
            return None
        total = len(self.gps_breadcrumbs)
        dev_time = len(self.deviation_events)
        ratio = dev_time / total if total else 0
        max_dev = max((e['distance'] for e in self.deviation_events), default=0)
        return {
            'total_points': total, 'deviation_events': dev_time,
            'deviation_ratio': ratio, 'max_deviation': max_dev,
            'was_anomalous': ratio > 0.3 or max_dev > 1000,
        }
