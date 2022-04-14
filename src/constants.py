import torch
from pathlib import Path
from src.utils import ensure_dir

################################################################################
# Global variables
################################################################################

# static directories
SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
DATA_DIR = Path(__file__).parent.parent / 'data'
CACHE_DIR = Path(__file__).parent.parent / 'cache'
MODEL_DIR = Path(__file__).parent.parent / 'pretrained'
RUNS_DIR = Path(__file__).parent.parent / 'runs'

ensure_dir(SCRIPTS_DIR)
ensure_dir(DATA_DIR)
ensure_dir(CACHE_DIR)
ensure_dir(MODEL_DIR)
ensure_dir(RUNS_DIR)

# data properties
SR = 16000
SIG_LEN = 4.0
LIBRISPEECH_EXT = "flac"
RIR_NOISE_EXT = "wav"

# optimization hyperparameters
MAX_ITER = 1000
BATCH_SIZE = 40
LOG_ITER = 200
RAND_SEED = 123
N_PER_CLASS = 1
TARGET_CLASS = None
EXCLUDE_CLASS = None
EOT_ITER = 1
N_EVALS = 2000
CONFIDENCE = 0.5

# path to ResNetSE34V2 weights
MODEL_PATH = MODEL_DIR / 'resnetse34v2.pt'

# speaker verification hyperparameters
N_SEGMENTS = 0
DISTANCE_FN = 'cosine'
THRESHOLD = 0.5846

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
