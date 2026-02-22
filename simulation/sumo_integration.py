"""
SUMO Real-Time Traffic Simulation with Hybrid Driver Behavior Monitoring

Run from project root:
    python -m simulation.sumo_integration          (headless)
    python -m simulation.sumo_integration --gui    (with SUMO GUI)
"""
import sys
import os
import time
import subprocess
import math
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── SUMO simulation config (separate from production utils.config) ────────────
SIM_DIR = _PROJECT_ROOT / 'simulation'
NETWORK_FILE = str(SIM_DIR / 'accra.net.xml')
ROUTE_FILE = str(SIM_DIR / 'vehicles.rou.xml')
CONFIG_FILE = str(SIM_DIR / 'simulation.sumocfg')

MODEL_DIR = str(_PROJECT_ROOT / 'models')
GHANA_GB_FILE = str(_PROJECT_ROOT / 'models' / 'ghana_gb_model.pkl')
GHANA_SCALER_FILE = str(_PROJECT_ROOT / 'models' / 'ghana_scaler.pkl')
PORTO_IF_FILE = str(_PROJECT_ROOT / 'models' / 'porto_if_model.pkl')
PORTO_SCALER_FILE = str(_PROJECT_ROOT / 'models' / 'porto_scaler.pkl')
FEATURES_FILE = str(_PROJECT_ROOT / 'models' / 'feature_names.pkl')
FUSION_CONFIG_FILE = str(_PROJECT_ROOT / 'models' / 'fusion_config.pkl')

STEP_LENGTH = 1.0
MAX_STEPS = 1000
UPDATE_INTERVAL = 5
COLORS = {'SAFE': (0, 255, 0), 'MEDIUM': (255, 255, 0), 'HIGH': (255, 0, 0)}

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set. Run: export SUMO_HOME='/opt/homebrew/opt/sumo/share/sumo'")

try:
    import traci
except ImportError as e:
    sys.exit(f"Failed to import traci: {e}")

import numpy as np

USE_GUI = '--gui' in sys.argv or '-g' in sys.argv
SUMO_BINARY = 'sumo-gui' if USE_GUI else 'sumo'
TRACI_RETRIES = 3
TRACI_RETRY_DELAY = 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# SUMO-specific HybridMonitor — thin adapter around core.ml_inference
# ═══════════════════════════════════════════════════════════════════════════════

class HybridMonitor:
    """Per-vehicle risk monitor for the SUMO simulation loop."""

    def __init__(self):
        from core.ml_inference import HybridMLModel

        self.colors = COLORS
        self.ml_model = None
        self.use_rule_based = False

        try:
            self.ml_model = HybridMLModel(models_dir=MODEL_DIR)
            self.mode = 'hybrid'
        except Exception as e:
            print(f"\u26a0\ufe0f ML model failed to load: {e}")
            self.mode = 'rule_based'
            self.use_rule_based = True
            print("\u26a0\ufe0f Falling back to rule-based scoring")

        self.vehicles = defaultdict(lambda: {
            'speeds': deque(maxlen=10),
            'positions': deque(maxlen=10),
            'accels': deque(maxlen=10),
            'distance': 0.0,
            'stops': 0,
            'last_speed': 0.0,
            'trip_start': None,
            'risk_score': 0.0,
            'risk_level': 'SAFE',
        })

    # ---------------------------------------------------------------- update

    def update_vehicle(self, veh_id, speed, position, timestamp):
        v = self.vehicles[veh_id]
        if v['trip_start'] is None:
            v['trip_start'] = timestamp

        speed_kmh = speed * 3.6
        v['speeds'].append(speed_kmh)

        if v['positions']:
            prev = v['positions'][-1]
            dx, dy = position[0] - prev[0], position[1] - prev[1]
            v['distance'] += math.sqrt(dx * dx + dy * dy)
        v['positions'].append(position)

        if v['last_speed'] > 0 and speed_kmh == 0:
            v['stops'] += 1

        if len(v['speeds']) >= 2:
            accel = (speed - v['last_speed'] / 3.6)
            v['accels'].append(accel)

        v['last_speed'] = speed_kmh

    # ---------------------------------------------------------------- predict

    def predict_risk(self, veh_id, timestamp):
        v = self.vehicles[veh_id]
        trip_dur = max(1, timestamp - (v['trip_start'] or timestamp))

        if self.use_rule_based:
            return self._rule_based_risk(v, trip_dur)

        features = self._build_features(v, trip_dur)
        try:
            result = self.ml_model.predict(features)
            score = result['hybrid_score']
            level = result['level']
        except Exception:
            return self._rule_based_risk(v, trip_dur)

        if level == 'HIGH RISK':
            level_key = 'HIGH'
        elif level == 'MEDIUM':
            level_key = 'MEDIUM'
        else:
            level_key = 'SAFE'

        v['risk_score'] = score
        v['risk_level'] = level_key
        color = self.colors.get(level_key, (0, 255, 0))
        return score, level_key, (*color, 255)

    # ---------------------------------------------------------- features

    def _build_features(self, v, trip_dur):
        spd = v['speeds'][-1] if v['speeds'] else 0
        accel = v['accels'][-1] if v['accels'] else 0
        accel_var = float(np.std(v['accels'])) if len(v['accels']) > 1 else 0
        dist_km = v['distance'] / 1000
        now = datetime.now()
        hour = now.hour

        return {
            'speed': spd,
            'acceleration': accel,
            'acceleration_variation': accel_var,
            'trip_duration': trip_dur,
            'trip_distance': dist_km,
            'stop_events': v['stops'],
            'road_encoded': 0,
            'weather_encoded': 0,
            'traffic_encoded': 1,
            'hour': hour,
            'month': now.month,
            'avg_speed': dist_km / (trip_dur / 3600 + 0.001),
            'stops_per_km': v['stops'] / (dist_km + 0.1),
            'accel_abs': abs(accel),
            'speed_normalized': spd / 100,
            'speed_squared': spd ** 2,
            'is_rush_hour': int(7 <= hour < 10 or 16 <= hour < 19),
            'is_night': int(hour >= 22 or hour <= 5),
        }

    # --------------------------------------------------------- rule fallback

    def _rule_based_risk(self, v, trip_dur):
        spd = v['speeds'][-1] if v['speeds'] else 0
        accel_var = float(np.std(v['accels'])) if len(v['accels']) > 1 else 0

        score = 0.0
        if spd > 80:
            score += 0.4
        elif spd > 50:
            score += 0.2
        if accel_var > 3:
            score += 0.3
        elif accel_var > 1.5:
            score += 0.15
        dist_km = v['distance'] / 1000
        if dist_km > 0.5 and v['stops'] / dist_km > 3:
            score += 0.2

        score = min(1.0, score)
        if score < 0.3:
            lvl = 'SAFE'
        elif score < 0.7:
            lvl = 'MEDIUM'
        else:
            lvl = 'HIGH'

        v['risk_score'] = score
        v['risk_level'] = lvl
        color = self.colors.get(lvl, (0, 255, 0))
        return score, lvl, (*color, 255)

    # ---------------------------------------------------------- stats

    def get_statistics(self):
        if not self.vehicles:
            return None
        total = len(self.vehicles)
        safe = sum(1 for v in self.vehicles.values() if v['risk_level'] == 'SAFE')
        med = sum(1 for v in self.vehicles.values() if v['risk_level'] == 'MEDIUM')
        high = sum(1 for v in self.vehicles.values() if v['risk_level'] == 'HIGH')
        avg = sum(v['risk_score'] for v in self.vehicles.values()) / total
        return {'total': total, 'safe': safe, 'medium': med, 'high': high, 'avg_risk': avg}


# ═══════════════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def validate_environment():
    print("\n\U0001f50d Validating environment...")
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        return False, "SUMO_HOME not set"
    print(f"   \u2705 SUMO_HOME: {sumo_home}")
    try:
        result = subprocess.run([SUMO_BINARY, '--version'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return False, f"SUMO binary failed: {result.stderr}"
        print(f"   \u2705 SUMO: {result.stdout.split(chr(10))[0]}")
    except FileNotFoundError:
        return False, f"'{SUMO_BINARY}' not found in PATH"
    except Exception as e:
        return False, f"SUMO check failed: {e}"
    return True, "OK"


def validate_files():
    print("\n\U0001f4c1 Validating simulation files...")
    required = {
        'Network': NETWORK_FILE, 'Routes': ROUTE_FILE, 'Config': CONFIG_FILE,
        'Porto IF Model': PORTO_IF_FILE, 'Porto Scaler': PORTO_SCALER_FILE,
        'Features': FEATURES_FILE, 'Fusion Config': FUSION_CONFIG_FILE,
    }
    missing = []
    for name, fp in required.items():
        if not Path(fp).exists():
            missing.append(name)
            print(f"   \u274c {name}: {fp}")
        else:
            print(f"   \u2705 {name}: {fp}")

    optional = {'Ghana GB Model': GHANA_GB_FILE, 'Ghana Scaler': GHANA_SCALER_FILE}
    for name, fp in optional.items():
        status = "\u2705" if Path(fp).exists() else "\u26a0\ufe0f "
        print(f"   {status} {name}: {fp}")

    if missing:
        return False, f"Missing: {', '.join(missing)}"
    print(f"   \u2705 All model files present...")
    return True, "OK"


def validate_network():
    print("\n\U0001f5fa\ufe0f  Validating network file...")
    try:
        with open(NETWORK_FILE, 'r') as f:
            content = f.read()
        if '<net' not in content:
            return False, "Missing <net> element"
        if '<edge' not in content:
            return False, "No edges"
        if '<junction' not in content:
            return False, "No junctions"
        print(f"   \u2705 Network file valid")
        return True, "OK"
    except Exception as e:
        return False, str(e)


def validate_routes():
    print("\n\U0001f697 Validating route file...")
    try:
        with open(ROUTE_FILE, 'r') as f:
            content = f.read()
        if '<routes' not in content:
            return False, "Missing <routes> element"
        if '<route' not in content and '<flow' not in content:
            return False, "No routes/flows"
        print(f"   \u2705 Route file valid")
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMO lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

def start_sumo():
    print(f"\n\U0001f680 Starting SUMO...")
    sumo_cmd = [
        SUMO_BINARY, '-c', CONFIG_FILE,
        '--step-length', str(STEP_LENGTH),
        '--no-warnings', '--start', '--quit-on-end',
    ]
    for attempt in range(1, TRACI_RETRIES + 1):
        try:
            try:
                traci.close()
            except Exception:
                pass
            print(f"   Attempt {attempt}/{TRACI_RETRIES}...")
            traci.start(sumo_cmd)
            print(f"   \u2705 TraCI connected successfully")
            return True, "Connected"
        except Exception as e:
            print(f"   \u26a0\ufe0f  Attempt {attempt} failed: {e}")
            if attempt < TRACI_RETRIES:
                time.sleep(TRACI_RETRY_DELAY)
            else:
                return False, str(e)
    return False, "Failed"


def run_loop(monitor):
    print("\n\u25b6\ufe0f  Running simulation...")
    print("=" * 80)

    step = 0
    last_update = 0
    peak_vehicles = 0
    errors = []

    try:
        expected = traci.simulation.getMinExpectedNumber()
        if expected == 0:
            return False, "No vehicles in simulation"
        print(f"\U0001f4ca Expected vehicles: {expected}")
        print("=" * 80)

        while step < MAX_STEPS:
            try:
                traci.simulationStep()
                vehs = traci.vehicle.getIDList()
                if vehs:
                    peak_vehicles = max(peak_vehicles, len(vehs))

                for vid in vehs:
                    try:
                        spd = traci.vehicle.getSpeed(vid)
                        pos = traci.vehicle.getPosition(vid)
                        monitor.update_vehicle(vid, spd, pos, step)
                        score, level, color = monitor.predict_risk(vid, step)
                        traci.vehicle.setColor(vid, color)
                    except traci.exceptions.TraCIException:
                        continue
                    except Exception as e:
                        if len(errors) < 5:
                            errors.append(str(e))
                        continue

                if step - last_update >= UPDATE_INTERVAL:
                    stats = monitor.get_statistics()
                    if stats:
                        print(f"\u23f1\ufe0f  Step {step:4d}s | "
                              f"Vehicles: {stats['total']:3d} | "
                              f"\U0001f7e2 {stats['safe']:3d} | "
                              f"\U0001f7e1 {stats['medium']:3d} | "
                              f"\U0001f534 {stats['high']:3d} | "
                              f"Avg Risk: {stats['avg_risk']:.3f}")
                    last_update = step

                if traci.simulation.getMinExpectedNumber() <= 0:
                    print(f"\n\u2705 All vehicles completed (step {step})")
                    break

                step += 1

            except traci.exceptions.FatalTraCIError as e:
                return False, f"Fatal TraCI error at step {step}: {e}"
            except KeyboardInterrupt:
                print("\n\n\u26a0\ufe0f  Interrupted")
                break

        print("=" * 80)
        print(f"\n\u2705 Simulation completed!")
        print(f"   Total steps: {step}")
        print(f"   Peak vehicles: {peak_vehicles}")
        return True, "Completed"

    except Exception as e:
        return False, f"Loop failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("\U0001f1ec\U0001f1ed GHANA DRIVER BEHAVIOR MONITORING - SUMO HYBRID SIMULATION")
    print("=" * 80)

    start_time = time.time()

    for validator in [validate_environment, validate_files, validate_network, validate_routes]:
        ok, msg = validator()
        if not ok:
            print(f"\n\u274c FAILED: {msg}")
            sys.exit(1)

    print("\n\U0001f4e6 Loading hybrid monitor (Ghana GB + Porto IF)...")
    try:
        monitor = HybridMonitor()
    except Exception as e:
        print(f"\n\u274c Monitor init failed: {e}")
        sys.exit(1)

    ok, msg = start_sumo()
    if not ok:
        print(f"\n\u274c SUMO failed: {msg}")
        sys.exit(1)

    try:
        ok, msg = run_loop(monitor)
        if ok:
            stats = monitor.get_statistics()
            if stats:
                print("\n" + "=" * 80)
                print("\U0001f4ca FINAL STATISTICS")
                print("=" * 80)
                print(f"Total vehicles monitored: {stats['total']}")
                print(f"  \U0001f7e2 Safe drivers:        {stats['safe']} ({stats['safe']/stats['total']*100:.1f}%)")
                print(f"  \U0001f7e1 Medium risk:         {stats['medium']} ({stats['medium']/stats['total']*100:.1f}%)")
                print(f"  \U0001f534 High risk:           {stats['high']} ({stats['high']/stats['total']*100:.1f}%)")
                print(f"  \U0001f4c8 Average risk score:  {stats['avg_risk']:.3f}")
        else:
            print(f"\n\u274c FAILED: {msg}")
    finally:
        try:
            if traci.isLoaded():
                traci.close()
                print("\u2705 TraCI closed")
        except Exception:
            pass

    elapsed = time.time() - start_time
    print(f"\n\u23f1\ufe0f  Total runtime: {elapsed:.2f}s")
    print("=" * 80 + "\n")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
