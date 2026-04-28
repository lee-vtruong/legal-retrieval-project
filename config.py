import os
from pathlib import Path

# --- BASE PATHS ---
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "ALQAC_2025"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = BASE_DIR / "experiments" / "models"

# --- DATA FILES ---
CORPUS_FILE = RAW_DIR / "law.json"
TRAIN_FILE = RAW_DIR / "train.json"
TEST_FILE = RAW_DIR / "private_test_GOLD.json"

CHUNK_FILE = PROCESSED_DIR / "chunks" / "alqac25_law_chunks.json"

# --- MODEL PATHS ---
CROSS_ENCODER_PATH = MODELS_DIR / "reranker_model"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_ranker.json"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# --- HYPERPARAMETERS ---
DEVICE = "cuda:0"
TOP_K_HYBRID = 12
CHUNK_OVERLAP_RATIO = 0.5
HARD_NEGATIVE_COUNT = 15
CE_EPOCHS = 10
XGB_MAX_DEPTH = 6

def ensure_dirs():
    """Tự động tạo các thư mục nếu chưa tồn tại"""
    for d in [PROCESSED_DIR / "chunks", EMBEDDINGS_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

ensure_dirs()