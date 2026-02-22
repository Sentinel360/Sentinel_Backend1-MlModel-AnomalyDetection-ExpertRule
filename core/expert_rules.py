"""
Expert Rules System for Sentinel360
Ghana-contextualized safety rules backed by research
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np


class ExpertRulesEngine:
    """
    Research-backed expert rules for passenger safety

    Sources:
    - Ghana Road Traffic Regulations (2012) - LI 2180
    - WHO Global Status Report on Road Safety (2024)
    - NRSA Ghana road safety statistics
    - Uber/Bolt safety system research
    """

    GHANA_SPEED_LIMITS = {
        'motorway': 100,
        'highway': 80,
        'urban': 50,
        'residential': 30,
        'school_zone': 20,
    }

    TIME_RISK_MULTIPLIERS = {
        (0, 6): 2.5,
        (6, 9): 1.1,
        (9, 14): 1.0,
        (14, 17): 0.9,
        (17, 20): 1.3,
        (20, 22): 1.5,
        (22, 24): 3.2,
    }

    HIGH_RISK_ZONES = {
        'nima': {
            'bounds': {'north': 5.5850, 'south': 5.5650, 'east': -0.1950, 'west': -0.2150},
            'risk_hours': (22, 5),
            'risk_level': 'HIGH',
        },
        'chorkor': {
            'bounds': {'north': 5.5550, 'south': 5.5350, 'east': -0.2450, 'west': -0.2650},
            'risk_hours': (22, 6),
            'risk_level': 'MEDIUM',
        },
    }

    ROAD_CONTEXTS = {
        'motorway': {
            'speed_limit': 100,
            'acceptable_range': (60, 110),
            'max_stops_per_10km': 2,
            'harsh_brake_tolerance': 1,
        },
        'arterial': {
            'speed_limit': 50,
            'acceptable_range': (20, 65),
            'max_stops_per_10km': 8,
            'harsh_brake_tolerance': 3,
        },
        'residential': {
            'speed_limit': 30,
            'acceptable_range': (10, 40),
            'max_stops_per_10km': 15,
            'harsh_brake_tolerance': 5,
        },
        'unpaved': {
            'speed_limit': 40,
            'acceptable_range': (5, 50),
            'max_stops_per_10km': 10,
            'harsh_brake_tolerance': 8,
            'harsh_accel_tolerance': 6,
            'ml_score_damper': -0.20,
        },
    }

    RUSH_HOUR_ZONES = [
        'Circle', 'Kaneshie', 'Achimota', 'Madina',
        'Tema Station', 'Tetteh Quarshie', 'Airport Area',
    ]

    def __init__(self):
        self.triggered_rules = []

    # ====================================================================
    # CRITICAL RULES
    # ====================================================================

    def rule_extreme_speeding(
        self, current_speed: float, speed_limit: int, location_type: str,
    ) -> Dict:
        excess = current_speed - speed_limit

        if current_speed > 130:
            return {
                'triggered': True, 'severity': 'CRITICAL',
                'rule_name': 'EXTREME_SPEEDING',
                'reason': f'Extreme speed: {current_speed:.0f} km/h (limit: {speed_limit})',
                'risk_adjustment': 0.50, 'action': 'IMMEDIATE_ALERT',
            }

        if excess >= 50:
            return {
                'triggered': True, 'severity': 'CRITICAL',
                'rule_name': 'EXCESSIVE_SPEEDING',
                'reason': f'Excessive speeding: {excess:.0f} km/h over limit',
                'risk_adjustment': 0.50, 'action': 'IMMEDIATE_ALERT',
            }

        if location_type == 'school_zone' and current_speed > 35:
            return {
                'triggered': True, 'severity': 'CRITICAL',
                'rule_name': 'SCHOOL_ZONE_SPEEDING',
                'reason': f'Speeding in school zone: {current_speed:.0f} km/h',
                'risk_adjustment': 0.45, 'action': 'IMMEDIATE_ALERT',
            }

        return {'triggered': False}

    def rule_crash_pattern(
        self, acceleration_history: List[float], current_speed: float,
    ) -> Dict:
        if len(acceleration_history) < 10:
            return {'triggered': False}

        harsh_brake = min(acceleration_history[-10:])
        stopped = current_speed < 5

        if harsh_brake < -7.0 and stopped:
            return {
                'triggered': True, 'severity': 'CRITICAL',
                'rule_name': 'CRASH_PATTERN',
                'reason': f'Possible crash: Emergency brake ({harsh_brake:.1f} m/s\u00b2) + stop',
                'risk_adjustment': 0.60, 'action': 'IMMEDIATE_ALERT_AND_CALL_USER',
            }

        return {'triggered': False}

    def rule_geofence_violation(
        self, current_location: Tuple[float, float], time_of_day: int,
    ) -> Dict:
        lat, lon = current_location

        for zone_name, zone_data in self.HIGH_RISK_ZONES.items():
            bounds = zone_data['bounds']
            if (bounds['south'] <= lat <= bounds['north']
                    and bounds['west'] <= lon <= bounds['east']):
                start_hour, end_hour = zone_data['risk_hours']
                if time_of_day >= start_hour or time_of_day < end_hour:
                    return {
                        'triggered': True, 'severity': 'HIGH',
                        'rule_name': 'GEOFENCE_VIOLATION',
                        'reason': f'Entered {zone_name} during high-risk hours',
                        'risk_adjustment': 0.35,
                        'action': 'ALERT_USER_AND_SHARE_LOCATION',
                    }

        return {'triggered': False}

    # ====================================================================
    # SEVERITY RULES
    # ====================================================================

    def rule_sustained_speeding(
        self, speed_history: List[float], speed_limit: int,
    ) -> Dict:
        if len(speed_history) < 10:
            return {'triggered': False}

        recent = speed_history[-10:]
        over = [s > (speed_limit + 20) for s in recent]

        if sum(over) >= 10:
            avg_excess = float(np.mean([s - speed_limit for s in recent]))
            return {
                'triggered': True, 'severity': 'MEDIUM',
                'rule_name': 'SUSTAINED_SPEEDING',
                'reason': f'Sustained speeding: {avg_excess:.0f} km/h over for 10+ seconds',
                'risk_adjustment': 0.25, 'escalate_to': 'HIGH',
            }

        return {'triggered': False}

    def rule_harsh_event_cluster(
        self, harsh_events: List[Dict], time_window: int = 60,
    ) -> Dict:
        now = datetime.now().timestamp()
        recent = [e for e in harsh_events if now - e['timestamp'] < time_window]

        if len(recent) >= 3:
            return {
                'triggered': True, 'severity': 'MEDIUM',
                'rule_name': 'HARSH_EVENT_CLUSTER',
                'reason': f'{len(recent)} harsh events in {time_window}s',
                'risk_adjustment': 0.20,
            }

        return {'triggered': False}

    def rule_suspicious_stop_pattern(
        self,
        stop_count: int,
        stop_locations: List[Tuple[float, float]],
        time_of_day: int,
        expected_route: List[Tuple[float, float]],
    ) -> Dict:
        is_night = (22 <= time_of_day or time_of_day < 5)
        if not is_night or not expected_route:
            return {'triggered': False}

        off_route = 0
        for loc in stop_locations:
            min_dist = min(self._haversine_distance(loc, rp) for rp in expected_route)
            if min_dist > 500:
                off_route += 1

        if off_route >= 3:
            return {
                'triggered': True, 'severity': 'MEDIUM',
                'rule_name': 'SUSPICIOUS_STOPS',
                'reason': f'Night hours with {off_route} off-route stops',
                'risk_adjustment': 0.30,
                'action': 'CHECK_IN_WITH_USER', 'escalate_to': 'HIGH',
            }

        return {'triggered': False}

    # ====================================================================
    # CONTEXT RULES
    # ====================================================================

    def rule_rush_hour_context(
        self, time_of_day: int, day_of_week: int, location: str,
    ) -> Dict:
        is_weekday = day_of_week < 5
        is_morning = 6 <= time_of_day < 9.5
        is_evening = 16.5 <= time_of_day < 20

        if is_weekday and (is_morning or is_evening) and location in self.RUSH_HOUR_ZONES:
            return {
                'context': 'RUSH_HOUR_CONGESTION',
                'adjustments': {
                    'max_acceptable_stops': 15,
                    'max_idle_ratio': 0.60,
                    'min_expected_speed': 10,
                    'ml_score_damper': -0.15,
                },
                'reason': 'Peak hour congestion - normal stop-and-go traffic',
            }

        return {'context': 'NORMAL', 'adjustments': {}}

    def rule_road_type_context(self, road_type: str) -> Dict:
        ctx = self.ROAD_CONTEXTS.get(road_type, self.ROAD_CONTEXTS['arterial'])
        return {
            'context': f'ROAD_TYPE_{road_type.upper()}',
            'adjustments': ctx,
            'reason': f'Road-specific thresholds for {road_type}',
        }

    def rule_time_risk_multiplier(self, time_of_day: int) -> Dict:
        for (start, end), mult in self.TIME_RISK_MULTIPLIERS.items():
            if start <= time_of_day < end:
                return {
                    'context': 'TIME_RISK',
                    'multiplier': mult,
                    'ml_adjustment': (mult - 1.0) * 0.15,
                    'reason': f'Time-of-day risk factor: {mult}x',
                }
        return {'context': 'NORMAL', 'multiplier': 1.0, 'ml_adjustment': 0.0}

    # ====================================================================
    # CORRELATION RULES
    # ====================================================================

    def rule_speed_location_correlation(
        self, current_speed: float, location_type: str, time_of_day: int,
    ) -> Dict:
        adj = 0.0
        reasons = []

        if location_type == 'residential' and current_speed > 45:
            adj += 0.15
            reasons.append(f'{current_speed:.0f} km/h in residential area')
        if location_type == 'school_zone' and 7 <= time_of_day < 17 and current_speed > 35:
            adj += 0.25
            reasons.append(f'{current_speed:.0f} km/h near school during school hours')
        if location_type == 'motorway' and current_speed < 50:
            adj += 0.10
            reasons.append(f'Unusually slow ({current_speed:.0f} km/h) on motorway')

        if adj > 0:
            return {
                'triggered': True, 'rule_name': 'SPEED_LOCATION_CORRELATION',
                'risk_adjustment': adj, 'reason': '; '.join(reasons),
            }
        return {'triggered': False}

    def rule_route_deviation_duration(
        self, route_deviation: float, trip_duration: float, expected_duration: float,
    ) -> Dict:
        dev_ratio = route_deviation / 1000 if route_deviation > 0 else 0
        dur_excess = ((trip_duration - expected_duration) / expected_duration
                      if expected_duration > 0 else 0)

        if dev_ratio > 0.30 and dur_excess > 0.50:
            return {
                'triggered': True, 'rule_name': 'ROUTE_DEVIATION_DURATION',
                'risk_adjustment': 0.20,
                'reason': f'{dev_ratio*100:.0f}% route deviation + {dur_excess*100:.0f}% longer',
                'action': 'CHECK_IN_WITH_USER',
            }
        return {'triggered': False}

    # ====================================================================
    # WHITELIST RULES
    # ====================================================================

    def rule_professional_driver_profile(self, driver_history: Dict) -> Dict:
        trips = driver_history.get('trips_completed', 0)
        rating = driver_history.get('rating', 0)
        incidents = driver_history.get('safety_incidents_90d', 0)

        if trips >= 500 and rating >= 4.75 and incidents == 0:
            return {
                'profile': 'PROFESSIONAL', 'ml_adjustment': -0.10,
                'reason': 'Verified professional driver with excellent record',
            }
        return {'profile': 'STANDARD', 'ml_adjustment': 0.0}

    def rule_known_safe_route(
        self, route_signature: str, driver_id: str, route_history: Dict,
    ) -> Dict:
        if route_signature in route_history:
            h = route_history[route_signature]
            if h['count'] >= 10 and h['incidents'] == 0:
                return {
                    'context': 'KNOWN_SAFE_ROUTE', 'ml_adjustment': -0.05,
                    'reason': f'Driver has completed this route {h["count"]} times safely',
                }
        return {'context': 'NORMAL', 'ml_adjustment': 0.0}

    # ====================================================================
    # HELPERS
    # ====================================================================

    def _haversine_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float],
    ) -> float:
        import math
        R = 6371000
        lat1, lon1 = point1
        lat2, lon2 = point2
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # ====================================================================
    # AGGREGATE
    # ====================================================================

    def apply_all_rules(self, trip_data: Dict, context: Dict) -> Dict:
        results = {
            'critical_rules': [], 'severity_rules': [],
            'context_adjustments': {}, 'correlation_rules': [],
            'whitelist_adjustments': {},
            'total_risk_adjustment': 0.0, 'triggered_actions': [],
        }

        # Critical
        for check in [
            self.rule_extreme_speeding(
                trip_data.get('current_speed', 0),
                context.get('speed_limit', 50),
                context.get('location_type', 'urban')),
            self.rule_crash_pattern(
                trip_data.get('acceleration_history', []),
                trip_data.get('current_speed', 0)),
            self.rule_geofence_violation(
                context.get('current_location', (0, 0)),
                context.get('time_of_day', 12)),
        ]:
            if check.get('triggered'):
                results['critical_rules'].append(check)
                results['total_risk_adjustment'] += check.get('risk_adjustment', 0)
                if check.get('action'):
                    results['triggered_actions'].append(check['action'])

        # Severity
        for check in [
            self.rule_sustained_speeding(
                trip_data.get('speed_history', []),
                context.get('speed_limit', 50)),
            self.rule_harsh_event_cluster(trip_data.get('harsh_events', [])),
            self.rule_suspicious_stop_pattern(
                trip_data.get('stop_count', 0),
                trip_data.get('stop_locations', []),
                context.get('time_of_day', 12),
                context.get('expected_route', [])),
        ]:
            if check.get('triggered'):
                results['severity_rules'].append(check)
                results['total_risk_adjustment'] += check.get('risk_adjustment', 0)

        # Context
        results['context_adjustments']['rush_hour'] = self.rule_rush_hour_context(
            context.get('time_of_day', 12), context.get('day_of_week', 3),
            context.get('location', ''))
        results['context_adjustments']['road_type'] = self.rule_road_type_context(
            context.get('road_type', 'arterial'))

        time_risk = self.rule_time_risk_multiplier(context.get('time_of_day', 12))
        results['context_adjustments']['time_risk'] = time_risk
        results['total_risk_adjustment'] += time_risk.get('ml_adjustment', 0)

        road_adj = results['context_adjustments']['road_type'].get('adjustments', {})
        if 'ml_score_damper' in road_adj:
            results['total_risk_adjustment'] += road_adj['ml_score_damper']
        rush_adj = results['context_adjustments']['rush_hour'].get('adjustments', {})
        if 'ml_score_damper' in rush_adj:
            results['total_risk_adjustment'] += rush_adj['ml_score_damper']

        # Correlation
        for check in [
            self.rule_speed_location_correlation(
                trip_data.get('current_speed', 0),
                context.get('location_type', 'urban'),
                context.get('time_of_day', 12)),
            self.rule_route_deviation_duration(
                trip_data.get('route_deviation', 0),
                trip_data.get('trip_duration', 0),
                context.get('expected_duration', 0)),
        ]:
            if check.get('triggered'):
                results['correlation_rules'].append(check)
                results['total_risk_adjustment'] += check.get('risk_adjustment', 0)

        # Whitelist
        prof = self.rule_professional_driver_profile(context.get('driver_history', {}))
        results['whitelist_adjustments']['driver_profile'] = prof
        results['total_risk_adjustment'] += prof.get('ml_adjustment', 0)

        route_w = self.rule_known_safe_route(
            context.get('route_signature', ''), context.get('driver_id', ''),
            context.get('route_history', {}))
        results['whitelist_adjustments']['known_route'] = route_w
        results['total_risk_adjustment'] += route_w.get('ml_adjustment', 0)

        return results
