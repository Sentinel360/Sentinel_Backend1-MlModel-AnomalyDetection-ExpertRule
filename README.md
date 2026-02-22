# Sentinel360 - Hybrid Driver Behavior Monitoring

Real-time driver risk prediction using a hybrid ML model (Ghana Gradient Boosting + Porto Isolation Forest) on real Accra road networks in Eclipse SUMO.

## Overview

Sentinel360 monitors simulated vehicles on real OpenStreetMap roads from Accra, Ghana and classifies each driver's risk level in real time using a **hybrid fusion** of two ML models:

- **Ghana GB** (Gradient Boosting Classifier, 70% weight) -- trained on Ghana driving patterns
- **Porto IF** (Isolation Forest, 30% weight) -- anomaly detector trained on 1M+ Porto taxi trips

Risk classification: **SAFE** (< 0.3) | **MEDIUM** (0.3-0.7) | **HIGH** (>= 0.7)

## Quick Start

### Prerequisites

- Python 3.10+
- [Eclipse SUMO](https://sumo.dlr.de/) 1.20+
- Dependencies: `numpy`, `scikit-learn`, `joblib`, `pandas`

### Run the simulation

```bash
export SUMO_HOME='/opt/homebrew/opt/sumo/share/sumo'

# Headless (fast, terminal output only)
python run_simulation.py

# With SUMO GUI (visual, see vehicles on real Accra map)
python run_simulation.py --gui
```

### Run the Streamlit model tester

```bash
pip install streamlit plotly
streamlit run test_model.py
```

### Verify setup

```bash
python quick_start.py
```

## How It Works

1. **SUMO loads** a real Accra road network (10,570 nodes, 4,740 road segments from OpenStreetMap)
2. **200 vehicles** spawn with realistic types (cars, taxis, trotros, trucks, motorbikes, aggressive drivers)
3. **Every simulation second**, TraCI extracts each vehicle's speed, position, and acceleration
4. **18 features** are computed from the telemetry (speed, acceleration variation, stops per km, etc.)
5. **Both models** independently score the driver, then scores are fused: `0.7 × GB + 0.3 × IF`
6. **Vehicle color** updates in real time: green (safe), yellow (medium), red (high risk)

## Project Structure

```
Sentinel_ML_Model/
├── run_simulation.py          # Main SUMO simulation runner
├── monitor.py                 # HybridMonitor - ML inference engine
├── config.py                  # Configuration constants
├── test_model.py              # Streamlit model testing dashboard
├── quick_start.py             # Environment verification script
├── generate_accra_network.py  # OSM network generation script
├── accra.net.xml              # Real Accra road network (OSM)
├── vehicles.rou.xml           # 200 vehicles with validated routes
├── simulation.sumocfg         # SUMO configuration
└── models/
    ├── ghana_gb_model.pkl     # Gradient Boosting classifier
    ├── porto_if_model.pkl     # Isolation Forest anomaly detector
    ├── ghana_scaler.pkl       # RobustScaler for Ghana GB
    ├── porto_scaler.pkl       # RobustScaler for Porto IF
    ├── feature_names.pkl      # 18 feature names
    └── fusion_config.pkl      # Fusion weights and thresholds
```

## Model Features (18)

| Feature | Description |
|---|---|
| `speed` | Current speed (km/h) |
| `acceleration` | Speed delta between timesteps |
| `acceleration_variation` | Std dev of recent accelerations |
| `trip_duration` | Time since trip start (s) |
| `trip_distance` | Cumulative distance (km) |
| `stop_events` | Number of full stops |
| `avg_speed` | Average trip speed (km/h) |
| `stops_per_km` | Stop frequency |
| `accel_abs` | Absolute acceleration |
| `speed_normalized` | Speed / 100 |
| `speed_squared` | Speed^2 (emphasizes high-speed risk) |
| `road_encoded` | Road type (urban/highway/rural) |
| `weather_encoded` | Weather condition |
| `traffic_encoded` | Traffic density |
| `hour` | Hour of day |
| `month` | Month of year |
| `is_rush_hour` | Rush hour flag |
| `is_night` | Nighttime flag |

## Streamlit Dashboard

The Streamlit app (`test_model.py`) provides three testing modes:

- **Pre-defined Scenarios** -- 15 edge-case test scenarios with expected outcomes
- **Manual Input** -- Custom feature values with auto-computed derived features
- **Batch Analysis** -- Run all tests, view accuracy, confusion matrix, and score distributions

## Network Coverage

Real OpenStreetMap data for Accra, Ghana:
- **Area**: Legon, Airport, Osu, Circle, Madina
- **Coordinates**: 5.545°N - 5.630°N, 0.225°W - 0.165°W
- **Intersections**: 10,570 nodes
- **Road segments**: 4,740 edges

## Regenerating the Network

To regenerate the Accra network from OpenStreetMap:

```bash
pip install osmnx
python generate_accra_network.py
```

This downloads fresh OSM data, extracts the largest connected component, converts to SUMO format with `netconvert`, and generates validated routes with `duarouter`.
