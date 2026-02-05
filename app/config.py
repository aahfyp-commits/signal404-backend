import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_model.h5"))
METADATA_PATH = os.getenv("METADATA_PATH", str(BASE_DIR / "metadata" / "metadata.pkl"))
UPLOAD_DIR = BASE_DIR / "uploads"

# Audio processing config
TARGET_SR = 16000
TARGET_DURATION = 3.0
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
MAX_PAD_LEN = 94  # Adjust based on your metadata (3 sec * 16000 / 512 + 1)

# API config
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)