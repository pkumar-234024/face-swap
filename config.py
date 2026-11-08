# AI Face Replacement & Animation Studio Configuration
from __future__ import annotations

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_VIDEOS_DIR = UPLOADS_DIR / "videos"
UPLOADS_IMAGES_DIR = UPLOADS_DIR / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = PROJECT_ROOT / "assets"
CSS_DIR = ASSETS_DIR / "css"
IMAGES_DIR = ASSETS_DIR / "images"
TEMP_DIR = PROJECT_ROOT / "temp"
MODELS_DIR = PROJECT_ROOT / "models"
BUFFALO_L_DIR = MODELS_DIR / "buffalo_l"

# File Format and Size Limits
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_VIDEO_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_IMAGE_SIZE_BYTES = 30 * 1024 * 1024   # 30 MB
MAX_PROCESSING_SECONDS = 600              # 10 minutes

# Target Resolutions
RESOLUTION_PRESETS = {
    "Original": None,
    "720p HD (1280x720)": (1280, 720),
    "1080p Full HD (1920x1080)": (1920, 1080),
    "4K Ultra HD (3840x2160)": (3840, 2160),
}

# Quality Presets (FFmpeg CRF values: lower = higher quality)
CRF_PRESETS = {
    "High": 18,
    "Medium": 23,
    "Economy": 28,
}

# Application UI Metadata
APP_TITLE = "AI Face Replacement & Animation Studio"
APP_SUBTITLE = "Transform authorized facial identities with AI-powered facial animation."
APP_DESCRIPTION = (
    "Production-grade facial replacement, expression transfer, head pose tracking, "
    "lip sync, and eye movement synchronization with strict consent verification and ethical AI disclosure."
)

VIDEO_PREVIEW_WIDTH = 640
IMAGE_PREVIEW_WIDTH = 480

# Default Processing Pipeline Configuration
DEFAULT_SETTINGS = {
    "face_replacement": True,
    "lip_sync": True,
    "eye_sync": True,
    "blinking_sync": True,
    "expression_sync": True,
    "head_movement_sync": True,
    "lighting_correction": True,
    "seamless_blending": True,
    "temporal_smoothing": True,
    "preserve_audio": True,
    "output_resolution": "Original",
    "video_quality": "High",
    "ai_modified_label": True,
}

# Hardware / Performance defaults
DEFAULT_BATCH_SIZE = 4
DEFAULT_DEVICE = "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "auto"

# Ensure all primary runtime directories exist
for directory in (
    UPLOADS_DIR,
    UPLOADS_VIDEOS_DIR,
    UPLOADS_IMAGES_DIR,
    OUTPUTS_DIR,
    ASSETS_DIR,
    CSS_DIR,
    IMAGES_DIR,
    TEMP_DIR,
    MODELS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
