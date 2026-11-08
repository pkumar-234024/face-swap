from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMP_DIR = PROJECT_ROOT / "temp"

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_VIDEO_SIZE_BYTES = 250 * 1024 * 1024
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
MAX_PROCESSING_SECONDS = 300

APP_TITLE = "AI Face Replacement Studio"
APP_DESCRIPTION = "Replace a person's face in a video using an authorized reference image."

for directory in (UPLOADS_DIR, OUTPUTS_DIR, ASSETS_DIR, TEMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

VIDEO_PREVIEW_WIDTH = 640
IMAGE_PREVIEW_WIDTH = 480

os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
