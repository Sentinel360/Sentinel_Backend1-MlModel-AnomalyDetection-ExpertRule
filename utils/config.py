"""SUMO Configuration - Hybrid Model"""
import os

SUMO_HOME = os.environ.get('SUMO_HOME', '/opt/homebrew/opt/sumo/share/sumo')
NETWORK_FILE = 'accra.net.xml'
ROUTE_FILE = 'vehicles.rou.xml'
CONFIG_FILE = 'simulation.sumocfg'
MODEL_DIR = 'models'

# Model files
GHANA_GB_FILE = os.path.join(MODEL_DIR, 'ghana_gb_model.pkl')
PORTO_IF_FILE = os.path.join(MODEL_DIR, 'porto_if_model.pkl')
GHANA_SCALER_FILE = os.path.join(MODEL_DIR, 'ghana_scaler.pkl')
PORTO_SCALER_FILE = os.path.join(MODEL_DIR, 'porto_scaler.pkl')
FEATURES_FILE = os.path.join(MODEL_DIR, 'feature_names.pkl')
FUSION_CONFIG_FILE = os.path.join(MODEL_DIR, 'fusion_config.pkl')

STEP_LENGTH = 1.0
MAX_STEPS = 1000
UPDATE_INTERVAL = 5

COLORS = {'SAFE': (0, 255, 0), 'MEDIUM': (255, 255, 0), 'HIGH': (255, 0, 0)}
RISK_THRESHOLDS = {'SAFE': 0.3, 'MEDIUM': 0.7}
