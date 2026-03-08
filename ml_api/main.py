"""
Sentinel360 ML API
FastAPI service for Cloud Run deployment.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.risk_fusion import RiskFusionEngine  # noqa: E402

app = FastAPI(title="Sentinel360 ML API", version="1.0.0")


class TripStartRequest(BaseModel):
    trip_id: str
    origin: Dict
    destination: Dict


class SensorDataRequest(BaseModel):
    trip_id: str
    gps: Dict[str, float]
    acceleration: Dict[str, float] = Field(default_factory=dict)
    timestamp: int
    source: str = "PHONE"


class TripEndRequest(BaseModel):
    trip_id: str


class RiskAssessmentResponse(BaseModel):
    trip_id: str
    timestamp: int
    final_score: float
    final_level: str
    final_color: str
    active_sensor: str
    components: Dict
    actions: List[str]
    explanation: str
    model_version: str = "1.0.0"


MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
WINDOW_SIZE = int(os.getenv("FEATURE_WINDOW_SIZE", "10"))

print("Loading Risk Fusion Engine...")
risk_engine = RiskFusionEngine(models_dir=MODEL_DIR, google_api_key=GOOGLE_API_KEY)
print("Risk Fusion Engine ready")

# In-memory rolling history per trip (sufficient for MVP + capstone demos)
trip_sensor_history: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))


@app.get("/")
def root() -> Dict:
    return {
        "status": "online",
        "service": "Sentinel360 ML API",
        "version": "1.0.0",
        "models_loaded": True,
    }


@app.post("/trip/start")
def start_trip(request: TripStartRequest) -> Dict:
    try:
        origin = normalize_lat_lon(request.origin, "origin")
        destination = normalize_lat_lon(request.destination, "destination")
        route_status = risk_engine.start_trip_monitoring(
            trip_id=request.trip_id,
            origin=origin,
            destination=destination,
        )
        trip_sensor_history[request.trip_id].clear()
        return {
            "trip_id": request.trip_id,
            "status": "monitoring_started",
            "route_loaded": bool(route_status.get("enabled")),
            "route_status": route_status,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict", response_model=RiskAssessmentResponse)
def predict_risk(request: SensorDataRequest) -> RiskAssessmentResponse:
    try:
        history = trip_sensor_history[request.trip_id]
        history.append(
            {
                "timestamp": request.timestamp,
                "speed": float(request.gps.get("speed", 0.0)),
                "lat": float(request.gps.get("lat", 0.0)),
                "lon": float(request.gps.get("lon", 0.0)),
                "ax": float(request.acceleration.get("x", 0.0)),
                "ay": float(request.acceleration.get("y", 0.0)),
                "az": float(request.acceleration.get("z", 0.0)),
            }
        )

        features = extract_features_for_model(request.trip_id, history, risk_engine.ml_model.feature_names)
        event_dt = datetime.fromtimestamp(request.timestamp / 1000)

        trip_data = {
            "current_speed": float(request.gps.get("speed", 0.0)),
            "acceleration_history": [h["ax"] for h in history],
            "speed_history": [h["speed"] for h in history],
            "trip_duration": max(
                0.0,
                (history[-1]["timestamp"] - history[0]["timestamp"]) / 1000 if len(history) > 1 else 0.0,
            ),
            "harsh_events": build_harsh_events(history),
            "stop_count": sum(1 for h in history if h["speed"] < 5),
            "stop_locations": [(h["lat"], h["lon"]) for h in history if h["speed"] < 5],
            "features": features,
        }

        context = {
            "current_location": (
                float(request.gps.get("lat", 0.0)),
                float(request.gps.get("lon", 0.0)),
            ),
            "time_of_day": event_dt.hour,
            "day_of_week": event_dt.weekday(),
            "speed_limit": 50,
            "location_type": "urban",
            "road_type": "arterial",
            "location": "Accra",
        }

        result = risk_engine.assess_risk(
            trip_id=request.trip_id,
            trip_data=trip_data,
            context=context,
        )

        return RiskAssessmentResponse(
            trip_id=request.trip_id,
            timestamp=request.timestamp,
            final_score=result["final_score"],
            final_level=result["final_level"],
            final_color=result["final_color"],
            active_sensor=request.source,
            components=result["components"],
            actions=result["actions"],
            explanation=result["explanation"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/trip/end")
def end_trip(request: TripEndRequest) -> Dict:
    try:
        summary = risk_engine.end_trip_monitoring(request.trip_id)
        trip_sensor_history.pop(request.trip_id, None)
        return {"trip_id": request.trip_id, "status": "ended", "summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def build_harsh_events(history: Deque[Dict]) -> List[Dict]:
    events: List[Dict] = []
    for h in history:
        ax = h["ax"]
        if ax > 4.0 or ax < -4.0:
            events.append({"timestamp": h["timestamp"] / 1000, "value": ax})
    return events


def normalize_lat_lon(value: Dict, field_name: str) -> Tuple[float, float]:
    """
    Accept multiple coordinate shapes from app/function payloads.
    Supported keys:
    - lat/lon
    - latitude/longitude
    - _latitude/_longitude (Firestore GeoPoint JSON-like)
    """
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object with coordinates")

    lat = value.get("lat", value.get("latitude", value.get("_latitude")))
    lon = value.get("lon", value.get("lng", value.get("longitude", value.get("_longitude"))))

    if lat is None or lon is None:
        raise ValueError(
            f"{field_name} is missing coordinates. Provide lat/lon (or latitude/longitude)."
        )
    return float(lat), float(lon)


def extract_features_for_model(trip_id: str, history: Deque[Dict], expected_features: List[str]) -> Dict[str, float]:
    speeds = np.array([h["speed"] for h in history], dtype=float) if history else np.array([0.0])
    accel_x = np.array([h["ax"] for h in history], dtype=float) if history else np.array([0.0])
    timestamps = np.array([h["timestamp"] for h in history], dtype=float) if history else np.array([0.0])

    duration_s = float((timestamps[-1] - timestamps[0]) / 1000) if len(timestamps) > 1 else 0.0
    duration_h = max(duration_s / 3600.0, 1e-6)

    stop_count = int(np.sum(speeds < 5))
    harsh_accel_count = int(np.sum(accel_x > 4.0))
    harsh_brake_count = int(np.sum(accel_x < -4.0))

    # Approximate total distance from speed samples (speed in km/h).
    # dt is unknown when sparse, so we use average period over observed points.
    if len(timestamps) > 1:
        avg_dt_h = float(np.mean(np.diff(timestamps)) / 1000.0 / 3600.0)
    else:
        avg_dt_h = 0.0
    total_distance_km = float(np.sum(speeds * avg_dt_h))

    now = datetime.now()
    time_of_day = now.hour
    day_of_week = now.weekday()

    candidates: Dict[str, float] = {
        # Original risk_fusion-style feature set
        "avg_speed": float(np.mean(speeds)),
        "max_speed": float(np.max(speeds)),
        "speed_std": float(np.std(speeds)),
        "avg_acceleration": float(np.mean(np.abs(accel_x))),
        "max_acceleration": float(np.max(np.abs(accel_x))),
        "harsh_accel_count": float(harsh_accel_count),
        "harsh_brake_count": float(harsh_brake_count),
        "stop_count": float(stop_count),
        "avg_stop_duration": 0.0,
        "total_distance": total_distance_km,
        "distance_per_stop": float(total_distance_km / max(stop_count, 1)),
        "time_of_day": float(time_of_day),
        "day_of_week": float(day_of_week),
        "trip_duration": duration_s,
        "speed_changes": float(np.sum(np.abs(np.diff(speeds)) > 10.0)) if len(speeds) > 1 else 0.0,
        "route_straightness": 0.85,
        "idle_time_ratio": float(stop_count / max(len(speeds), 1)),
        "avg_trip_speed": float(total_distance_km / duration_h) if duration_h > 0 else float(np.mean(speeds)),
        # Simulation-style feature set
        "speed": float(speeds[-1]),
        "acceleration": float(accel_x[-1]),
        "acceleration_variation": float(np.std(accel_x)),
        "trip_distance": total_distance_km,
        "stop_events": float(stop_count),
        "road_encoded": 0.0,
        "weather_encoded": 0.0,
        "traffic_encoded": 1.0,
        "hour": float(time_of_day),
        "month": float(now.month),
        "stops_per_km": float(stop_count / max(total_distance_km, 0.1)),
        "accel_abs": float(abs(accel_x[-1])),
        "speed_normalized": float(speeds[-1] / 100.0),
        "speed_squared": float(speeds[-1] ** 2),
        "is_rush_hour": float(1 if (7 <= time_of_day < 10 or 16 <= time_of_day < 19) else 0),
        "is_night": float(1 if (time_of_day >= 22 or time_of_day <= 5) else 0),
    }

    features = {name: float(candidates.get(name, 0.0)) for name in expected_features}
    return features


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
