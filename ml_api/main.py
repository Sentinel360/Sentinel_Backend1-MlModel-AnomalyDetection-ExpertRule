"""
Sentinel360 ML API
FastAPI service for Cloud Run deployment.
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# When running locally: ml_api/ is inside SUMO_Hybrid_Package/, so parent.parent
# reaches the repo root where core/ and models/ live.
# When running in Docker: main.py is at /app/main.py and core/ is at /app/core/,
# so we also add the parent (working dir) to cover both cases.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent  # /app in container
for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.risk_fusion import RiskFusionEngine  # noqa: E402

# ---------------------------------------------------------------------------
# App & Config
# ---------------------------------------------------------------------------

app = FastAPI(title="Sentinel360 ML API", version="1.0.0")

# MODEL_DIR: Docker image has models at /app/models. If env is wrong (e.g. /models) or empty, fall back.
_app_models = APP_DIR / "models"
_default_model_dir = str(_app_models) if _app_models.is_dir() else str(PROJECT_ROOT / "models")
_model_dir_env = os.getenv("MODEL_DIR", "").strip()
if _model_dir_env:
    _cand = Path(_model_dir_env)
    if not _cand.is_absolute():
        _cand = (APP_DIR / _model_dir_env).resolve()
    if (_cand / "ghana_gb_model.pkl").is_file():
        MODEL_DIR = str(_cand)
    else:
        print(
            f"WARNING: MODEL_DIR={_model_dir_env!r} has no model files; using {_default_model_dir}",
        )
        MODEL_DIR = _default_model_dir
else:
    MODEL_DIR = _default_model_dir
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
WINDOW_SIZE = int(os.getenv("FEATURE_WINDOW_SIZE", "10"))
ML_API_KEY = os.getenv("ML_API_KEY", "")  # optional auth
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

print("Loading Risk Fusion Engine...")
risk_engine = RiskFusionEngine(models_dir=MODEL_DIR, google_api_key=GOOGLE_API_KEY)
print("Risk Fusion Engine ready")

# In-memory rolling history per trip (sufficient for MVP + capstone demos)
trip_sensor_history: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

# Simple weather cache: (lat_rounded, lon_rounded) -> (timestamp_s, encoded_value)
_weather_cache: Dict[Tuple[float, float], Tuple[float, float]] = {}
_WEATHER_CACHE_TTL = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Auth middleware (optional — skipped when ML_API_KEY is empty)
# ---------------------------------------------------------------------------

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Reject requests without a valid API key (if ML_API_KEY is configured)."""
    if not ML_API_KEY:
        return await call_next(request)

    # Health check is always open
    if request.url.path == "/":
        return await call_next(request)

    # Accept via header or query param
    key = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if key != ML_API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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

        features = extract_features_for_model(
            request.trip_id, history, risk_engine.ml_model.feature_names,
        )
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Haversine distance (meters) — replaces inaccurate speed*dt approximation
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS points in metres."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _total_distance_km(history: Deque[Dict]) -> float:
    """Sum haversine segment distances across the history window."""
    if len(history) < 2:
        return 0.0
    total = 0.0
    pts = list(history)
    for i in range(1, len(pts)):
        total += _haversine_m(pts[i - 1]["lat"], pts[i - 1]["lon"], pts[i]["lat"], pts[i]["lon"])
    return total / 1000.0


# ---------------------------------------------------------------------------
# Weather encoding via OpenWeatherMap
# ---------------------------------------------------------------------------

def _get_weather_encoded(lat: float, lon: float) -> float:
    """
    Fetch current weather and return an encoded risk factor:
    0.0 = clear/clouds, 0.3 = drizzle/mist, 0.6 = rain, 1.0 = heavy/storm
    Falls back to 0.0 on any error or missing API key.
    """
    if not OPENWEATHER_API_KEY:
        return 0.0

    # Round to 1 decimal for cache key (city-level granularity)
    cache_key = (round(lat, 1), round(lon, 1))
    now_s = datetime.now().timestamp()
    cached = _weather_cache.get(cache_key)
    if cached and (now_s - cached[0]) < _WEATHER_CACHE_TTL:
        return cached[1]

    try:
        resp = httpx.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY},
            timeout=3.0,
        )
        resp.raise_for_status()
        data = resp.json()
        weather_id = data.get("weather", [{}])[0].get("id", 800)

        # WMO-style grouping
        if weather_id >= 800:
            encoded = 0.0   # clear / clouds
        elif weather_id >= 700:
            encoded = 0.3   # mist / fog / haze
        elif weather_id >= 600:
            encoded = 0.6   # snow (rare in Accra but handle it)
        elif weather_id >= 500:
            # 500-504 rain, 511 freezing rain, 520-531 shower
            encoded = 0.6 if weather_id < 502 else 1.0
        elif weather_id >= 300:
            encoded = 0.3   # drizzle
        elif weather_id >= 200:
            encoded = 1.0   # thunderstorm
        else:
            encoded = 0.0

        _weather_cache[cache_key] = (now_s, encoded)
        return encoded
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Road type and traffic encoding proxies
# ---------------------------------------------------------------------------

def _road_encoded_proxy(avg_speed: float, max_speed: float) -> float:
    """
    Approximate road type from observed speeds:
    0 = residential (<30), 1 = arterial (30-60), 2 = highway (>60)
    """
    ref_speed = max(avg_speed, max_speed * 0.8)
    if ref_speed > 60:
        return 2.0
    elif ref_speed > 30:
        return 1.0
    return 0.0


def _traffic_encoded_proxy(speeds: np.ndarray) -> float:
    """
    Approximate traffic density from speed variance:
    0 = free-flow, 1 = moderate, 2 = congested
    """
    if len(speeds) < 2:
        return 1.0  # unknown → moderate default
    avg = float(np.mean(speeds))
    std = float(np.std(speeds))
    if avg < 15 and std < 5:
        return 2.0  # slow + steady = congestion
    elif avg < 30 or std > 15:
        return 1.0  # moderate / stop-and-go
    return 0.0  # free-flow


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_for_model(
    trip_id: str, history: Deque[Dict], expected_features: List[str],
) -> Dict[str, float]:
    speeds = np.array([h["speed"] for h in history], dtype=float) if history else np.array([0.0])
    accel_x = np.array([h["ax"] for h in history], dtype=float) if history else np.array([0.0])
    timestamps = np.array([h["timestamp"] for h in history], dtype=float) if history else np.array([0.0])

    duration_s = float((timestamps[-1] - timestamps[0]) / 1000) if len(timestamps) > 1 else 0.0
    duration_h = max(duration_s / 3600.0, 1e-6)

    stop_count = int(np.sum(speeds < 5))
    harsh_accel_count = int(np.sum(accel_x > 4.0))
    harsh_brake_count = int(np.sum(accel_x < -4.0))

    # GPS haversine distance (accurate)
    total_distance_km = _total_distance_km(history)

    # Use EVENT timestamp, not server time
    last_ts = timestamps[-1]
    event_dt = datetime.fromtimestamp(last_ts / 1000)
    time_of_day = event_dt.hour
    day_of_week = event_dt.weekday()
    month = event_dt.month

    # Context-aware encodings
    avg_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    road_enc = _road_encoded_proxy(avg_speed, max_speed)
    traffic_enc = _traffic_encoded_proxy(speeds)

    # Weather (uses last known GPS position)
    lat = float(history[-1]["lat"]) if history else 0.0
    lon = float(history[-1]["lon"]) if history else 0.0
    weather_enc = _get_weather_encoded(lat, lon)

    candidates: Dict[str, float] = {
        # Original risk_fusion-style feature set
        "avg_speed": avg_speed,
        "max_speed": max_speed,
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
        "avg_trip_speed": float(total_distance_km / duration_h) if duration_h > 0 else avg_speed,
        # Simulation-style feature set
        "speed": float(speeds[-1]),
        "acceleration": float(accel_x[-1]),
        "acceleration_variation": float(np.std(accel_x)),
        "trip_distance": total_distance_km,
        "stop_events": float(stop_count),
        "road_encoded": road_enc,
        "weather_encoded": weather_enc,
        "traffic_encoded": traffic_enc,
        "hour": float(time_of_day),
        "month": float(month),
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
