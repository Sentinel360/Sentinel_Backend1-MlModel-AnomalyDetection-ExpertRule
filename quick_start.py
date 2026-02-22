#!/usr/bin/env python3
"""Quick-start verification for SUMO Hybrid Simulation"""
import sys, os, subprocess
from pathlib import Path

def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")

def check_python():
    section("1. Python Version")
    v = sys.version_info
    print(f"   Python {v.major}.{v.minor}.{v.micro}")
    ok = v.major == 3 and v.minor >= 10
    print(f"   {'OK' if ok else 'Python 3.10+ required'}\n")
    return ok

def check_packages():
    section("2. Python Packages")
    ok = True
    for alias, name in {'numpy': 'numpy', 'sklearn': 'scikit-learn', 'joblib': 'joblib', 'traci': 'traci'}.items():
        try:
            mod = __import__(alias)
            print(f"   {name}: {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            print(f"   {name}: MISSING")
            ok = False
    print()
    return ok

def check_sumo():
    section("3. SUMO Installation")
    try:
        r = subprocess.run(['sumo', '--version'], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            print(f"   {r.stdout.split(chr(10))[0]}\n")
            return True
    except Exception:
        pass
    print("   SUMO not found\n")
    return False

def check_files():
    section("4. Required Files")
    ok = True
    groups = {
        'Scripts': ['run_simulation.py', 'monitor.py', 'config.py'],
        'Network': ['accra.net.xml', 'vehicles.rou.xml', 'simulation.sumocfg'],
        'Models': ['models/porto_if_model.pkl', 'models/porto_scaler.pkl',
                    'models/ghana_gb_model.pkl', 'models/ghana_scaler.pkl',
                    'models/feature_names.pkl', 'models/fusion_config.pkl']
    }
    for cat, files in groups.items():
        print(f"   {cat}:")
        for f in files:
            p = Path(f)
            if p.exists():
                sz = p.stat().st_size
                s = f"{sz/1024:.1f}KB" if sz < 1048576 else f"{sz/1048576:.1f}MB"
                print(f"      {f} ({s})")
            else:
                print(f"      {f} (MISSING)")
                ok = False
    print()
    return ok

def check_models():
    section("5. Model Loading")
    try:
        from monitor import HybridMonitor
        m = HybridMonitor()
        results = []
        if m.ghana_gb:
            results.append(f"Ghana GB: {type(m.ghana_gb).__name__}")
        if m.porto_if:
            results.append(f"Porto IF: {type(m.porto_if).__name__}")
        for r in results:
            print(f"   {r}")
        print(f"   Mode: {m.mode}")
        print(f"   Fusion: {m.weight_gb}xGB + {m.weight_if}xIF\n")
        return True
    except Exception as e:
        print(f"   Error: {e}\n")
        return False

def check_env():
    section("6. Environment")
    sh = os.environ.get('SUMO_HOME', '/opt/homebrew/opt/sumo/share/sumo')
    print(f"   SUMO_HOME: {sh}")
    ok = Path(sh).exists()
    print(f"   {'Exists' if ok else 'NOT FOUND'}\n")
    return ok

def main():
    print("\n" + "=" * 70)
    print("  SUMO Hybrid Traffic Simulation - Quick Start")
    print("=" * 70)

    checks = [
        ("Python", check_python), ("Packages", check_packages),
        ("SUMO", check_sumo), ("Files", check_files),
        ("Models", check_models), ("Environment", check_env),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"   Error: {e}\n")
            results[name] = False

    section("Summary")
    for name, ok in results.items():
        print(f"   {'PASS' if ok else 'FAIL'} {name}")
    print()

    if not all(results.values()):
        print("   Fix errors above before running.\n")
        sys.exit(1)

    if '--full' in sys.argv:
        subprocess.run([sys.executable, 'run_simulation.py'])
    else:
        print("   All systems ready!\n")
        print("   python run_simulation.py       # Run simulation")
        print("   python quick_start.py --full   # Run after checks\n")

if __name__ == '__main__':
    main()
