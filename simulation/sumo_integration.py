"""
SUMO Real-Time Traffic Simulation with Hybrid Driver Behavior Monitoring

Run: python run_simulation.py          (headless)
     python run_simulation.py --gui    (with SUMO GUI)
"""
import sys
import os
import time
import subprocess
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from config import *

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set. Run: export SUMO_HOME='/opt/homebrew/opt/sumo/share/sumo'")

try:
    import traci
    from monitor import HybridMonitor
except ImportError as e:
    sys.exit(f"Failed to import: {e}")

USE_GUI = '--gui' in sys.argv or '-g' in sys.argv
SUMO_BINARY = 'sumo-gui' if USE_GUI else 'sumo'
TRACI_RETRIES = 3
TRACI_RETRY_DELAY = 2.0


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
