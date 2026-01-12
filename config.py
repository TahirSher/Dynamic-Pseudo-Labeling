import os

# ============================================================================
# THRESHOLDS AND SUBGROUPS
# ============================================================================
THRESHOLDS = [60, 84, 120] 

# TPS Subgroup definitions
SUBGROUPS = {
    'Overall': None,  # Overall population 
    # Uncomment to enable TPS subgroups:
    # 'C1': {'TPS_L1': 1, 'TPS_1_49': 0, 'TPS_50L': 0},  # TPS < 1%
    # 'C2': {'TPS_L1': 0, 'TPS_1_49': 1, 'TPS_50L': 0},  # TPS 1-49%
    # 'C3': {'TPS_L1': 0, 'TPS_1_49': 0, 'TPS_50L': 1}   # TPS >= 50%
}

# ============================================================================
# DIRECTORIES
# ============================================================================
BASE_OUTPUT_DIR = "outputs/multi_threshold_survival"
BASE_WEIGHTS_DIR = "weights/multi_threshold_survival"

# ============================================================================
# DATA COLUMNS
# ============================================================================
TIME_COL = 'Time'
EVENT_COL = 'Event'

EXCLUDE_COLS = ['ID', 'year', 'Age at Diagnosis', 'TPS_L1', 'TPS_1_49', 'TPS_50L']

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
MAX_ITER = 100
TARGET_R2 = 0.95
TARGET_C_INDEX = 0.85
C_INDEX_TOLERANCE = 0.03
MIN_DELTA = 0.001
PATIENCE = 15
EARLY_STOPPING_PATIENCE = 5 

# Model parameters
ENSEMBLE_SIZE = 3
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

# Neural network parameters
NN_LAYERS = [128, 64, 32, 16]
DROPOUT_RATES = [0.4, 0.3, 0.2, 0.1]
L2_REGULARIZATION = 1e-5
INITIAL_LEARNING_RATE = 0.0005
LEARNING_RATE_DECAY = 0.95
BATCH_SIZE = 32
EPOCHS = 100

# Calibration parameters
ISOTONIC_MIN = 0.001
ISOTONIC_MAX = 0.999
BLEND_FACTOR = 0.7

# Constraint parameters
MAX_DECEASED_PROB_BASE = 0.15
MAX_DECEASED_PROB_DECAY = 0.01
MIN_CENSORED_PROB = 0.20

# ============================================================================
# PLOTTING PARAMETERS
# ============================================================================
PLOT_INTERVAL = 10
PLOT_DPI = 600
FIGURE_SIZE_CURVES = (20, 8)
FIGURE_SIZE_IMPORTANCE = (20, 16)

# ============================================================================
# ENVIRONMENT SETTINGS
# ============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

# ============================================================================
# SUBGROUP DISPLAY NAMES
# ============================================================================
SUBGROUP_DISPLAY_NAMES = {
    'Overall': 'Overall Population',
    'C1': 'TPS < 1%',
    'C2': 'TPS 1-49%', 
    'C3': 'TPS = 50%'
}

# ============================================================================
# TIME FEATURE KEYWORDS
# ============================================================================
TIME_FEATURE_KEYWORDS = [
    'time', 'log_time', 'squared', 'cubic', 'sqrt', 'reciprocal', 
    'normalized', 'progression', 'remaining', 'ratio', 'decay', 'bins',
    'event_interaction'
]