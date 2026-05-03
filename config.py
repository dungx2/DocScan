"""
config.py
---------
Central configuration for the DocScan application.
Values can be overridden via environment variables.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "outputs"
MODELS_DIR  = BASE_DIR / "models"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Model ────────────────────────────────────────────────────────────────────
# Set this to the path of your downloaded .pth weights file.
# Leave as None to run Phase 1 (traditional CV) only.
MODEL_PATH: str | None = os.getenv(
    "DOCSCAN_MODEL_PATH",
    str(MODELS_DIR / "dewarpnet.pth")   # expected filename
)

# PyTorch device: "auto" | "cpu" | "cuda" | "cuda:0"
DEVICE: str = os.getenv("DOCSCAN_DEVICE", "auto")

# ── File upload ───────────────────────────────────────────────────────────────
MAX_CONTENT_LENGTH: int = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png"}

# ── Processing defaults ───────────────────────────────────────────────────────
# output_mode: "binary" | "grayscale" | "color"
DEFAULT_OUTPUT_MODE: str = os.getenv("DOCSCAN_OUTPUT_MODE", "binary")

# binarize_method: "adaptive" | "sauvola" | "otsu"
DEFAULT_BINARIZE_METHOD: str = os.getenv("DOCSCAN_BINARIZE", "adaptive")

# ── Flask ─────────────────────────────────────────────────────────────────────
SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
PORT: int = int(os.getenv("PORT", 5000))
