# Sentinel360 Backend

**AI-Powered Passenger Safety Monitoring System for Ghana**

Sentinel360 combines machine learning, expert safety rules, and route anomaly detection to protect ride-hailing passengers in real-time.

---

## Overview

Sentinel360 monitors trips in real-time using:

1. **Hybrid ML Model** (0.7 x Ghana GB + 0.3 x Porto IF)
   - Detects unsafe driving patterns from 18 telemetry features
   - Ghana Gradient Boosting trained on synthetic Ghana driving data
   - Porto Isolation Forest validated on 1M+ Porto taxi trips
   - 99.7% accuracy on cross-geography validation

2. **Expert Rules System**
   - Ghana-contextualized safety rules backed by research
   - Based on NRSA road safety data, WHO standards, Ghana Road Traffic Regulations (LI 2180)
   - Considers rush hours, road types, time-of-day risk, geofence zones

3. **Route Anomaly Detection**
   - Real-time route deviation monitoring via Google Maps API
   - Corridor buffering with Shapely/pyproj
   - Detects wrong direction, prolonged deviation, geographic anomalies

Risk classification: **SAFE** (< 0.3) | **MEDIUM** (0.3-0.7) | **HIGH RISK** (>= 0.7)

---

## Architecture

```
+--------------+      +--------------+      +--------------+
|  IoT Device  |----->|    Cloud     |----->|  Mobile App  |
|  (ESP32)     |      |  (Firebase)  |      |  (Flutter)   |
|              |      |              |      |              |
| GPS + IMU    |      | ML + Rules   |      | Real-time    |
| Cellular 4G  |      | + Route Mon  |      | Monitoring   |
+--------------+      +--------------+      +--------------+
```

---

## Project Structure

```
sentinel360-backend/
├── core/                       # Core risk assessment
│   ├── ml_inference.py         # HybridMLModel (Ghana GB + Porto IF)
│   ├── expert_rules.py         # ExpertRulesEngine (Ghana safety rules)
│   ├── route_anomaly.py        # RouteAnomalyDetector (Google Maps)
│   └── risk_fusion.py          # RiskFusionEngine (combines all)
│
├── utils/                      # Helper functions
│   ├── gps_utils.py            # Haversine, bearing, point-in-polygon
│   ├── ghana_data.py           # Accra landmarks, congestion zones
│   └── config.py               # Environment-based configuration
│
├── models/                     # ML models (not in git)
│   ├── ghana_gb_model.pkl      # Gradient Boosting classifier
│   ├── porto_if_model.pkl      # Isolation Forest anomaly detector
│   ├── ghana_scaler.pkl        # RobustScaler for Ghana GB
│   ├── porto_scaler.pkl        # RobustScaler for Porto IF
│   ├── feature_names.pkl       # 18 feature names
│   └── fusion_config.pkl       # Fusion weights and thresholds
│
├── tests/                      # Pytest unit tests (207 tests)
│   ├── test_ml_inference.py
│   ├── test_expert_rules.py
│   ├── test_route_anomaly.py
│   ├── test_risk_fusion.py
│   ├── test_gps_utils.py
│   ├── test_ghana_data.py
│   └── test_integration.py
│
├── simulation/                 # SUMO traffic simulation
│   ├── sumo_integration.py     # Real-time SUMO + ML monitoring
│   ├── streamlit_test_app.py   # Streamlit model testing dashboard
│   ├── accra.net.xml           # Real Accra OSM road network
│   ├── vehicles.rou.xml        # 200 vehicles with validated routes
│   └── simulation.sumocfg      # SUMO configuration
│
├── cloud_functions/            # Firebase deployment
├── deployment/                 # Docker & deploy scripts
├── requirements.txt
├── setup.py
├── pytest.ini
├── .env.example
└── .gitignore
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- Google Maps API key (for route monitoring)
- Eclipse SUMO 1.20+ (for simulation only)

### Installation

```bash
git clone https://github.com/Sentinel360/Sentinel_ML_Model.git
cd Sentinel_ML_Model

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Download ML Models

Models are stored separately (too large for git). Place them in the `models/` directory:

```
models/
├── ghana_gb_model.pkl
├── porto_if_model.pkl
├── ghana_scaler.pkl
├── porto_scaler.pkl
├── feature_names.pkl
└── fusion_config.pkl
```

---

## Usage

### Risk Assessment (Production API)

```python
from core.risk_fusion import RiskFusionEngine

engine = RiskFusionEngine(
    models_dir='models',
    google_api_key='YOUR_API_KEY'
)

engine.start_trip_monitoring(
    trip_id='trip_123',
    origin=(5.6519, -0.1873),      # Legon
    destination=(5.6052, -0.1668)   # Airport
)

result = engine.assess_risk(
    trip_id='trip_123',
    trip_data={
        'current_speed': 65,
        'acceleration_history': [0.5, 1.2, -2.1],
        'features': { ... }
    },
    context={
        'current_location': (5.64, -0.18),
        'time_of_day': 18,
        'speed_limit': 50,
        'location_type': 'urban'
    }
)

print(f"Risk: {result['final_level']}")    # SAFE / MEDIUM / HIGH RISK
print(f"Score: {result['final_score']:.2f}")
print(f"Actions: {result['actions']}")
```

### SUMO Simulation

```bash
export SUMO_HOME='/opt/homebrew/opt/sumo/share/sumo'

# Headless
python -m simulation.sumo_integration

# With SUMO GUI
python -m simulation.sumo_integration --gui
```

### Streamlit Model Tester

```bash
pip install streamlit plotly
streamlit run simulation/streamlit_test_app.py
```

---

## Testing

```bash
# Run all 207 tests
pytest tests/

# Run with coverage
pytest --cov=core --cov=utils tests/

# Run specific test file
pytest tests/test_expert_rules.py -v
```

---

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

---

## Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| ML Model | Cross-geography accuracy | 99.7% |
| ML Model | Ghana baseline accuracy | 90% |
| Hybrid Fusion | Safe trip classification | 99.7% |
| Route Detection | False positive rate | <5% |
| System Latency | ML inference | ~20ms |
| System Latency | End-to-end | ~60ms |

---

## Ghana Context

### Speed Limits (LI 2180)
- Motorway: 100 km/h
- Urban: 50 km/h
- Residential: 30 km/h
- School Zone: 20 km/h

### High-Risk Times (NRSA Data)
- Late night (10 PM - 2 AM): 3.2x baseline risk
- Evening rush (5-8 PM): 1.3x baseline risk

### Congestion Zones
- Circle, Kaneshie, Achimota, Madina
- Rush hours: 6-9 AM, 5-8 PM (Mon-Fri)

### SUMO Network Coverage
Real OpenStreetMap data for Accra, Ghana:
- **Area**: Legon, Airport, Osu, Circle, Madina
- **Intersections**: 10,570 nodes
- **Road segments**: 4,740 edges

---

## Configuration

Environment variables (`.env`):

```bash
GOOGLE_API_KEY=your_key_here
MODEL_DIR=models
FUSION_GB_WEIGHT=0.7
FUSION_IF_WEIGHT=0.3
SAFE_THRESHOLD=0.3
MEDIUM_THRESHOLD=0.7
ROUTE_BUFFER_DISTANCE=100
ROUTE_DEVIATION_CRITICAL=500
```

---

## Research & Citations

1. **Liu, F. T., et al. (2008).** Isolation Forest. IEEE ICDM.
2. **Ghana Road Traffic Regulations (2012).** LI 2180.
3. **WHO Global Status Report on Road Safety (2024).**
4. **NRSA Ghana Road Safety Statistics (2023).**

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Authors

- **Caleb Okwesie Arthur** - Ashesi University - caleb.arthur@ashesi.edu.gh
- **Frances Seyram Fiahagbe** - Ashesi University - frances.fiahagbe@ashesi.edu.gh

---

## Acknowledgments

- Ghana National Road Safety Authority (NRSA)
- Ashesi University Computer Science Department
- Uber/Bolt safety research teams
- OpenStreetMap Ghana community

---

## Contact

- Caleb: caleb.arthur@ashesi.edu.gh
- Frances: frances.fiahagbe@ashesi.edu.gh
- Project: https://github.com/Sentinel360/Sentinel_ML_Model
