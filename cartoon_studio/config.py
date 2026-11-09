from __future__ import annotations

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure runtime directories exist
for directory in (MODELS_DIR, ASSETS_DIR, UPLOADS_DIR, TEMP_DIR, OUTPUTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Application UI Metadata
APP_TITLE = "AI 2D Cartoon Video Studio"
APP_SUBTITLE = "Transform real-life videos into consistent 2D cartoon animations with AI."
APP_DESCRIPTION = (
    "Production-grade video-to-cartoon synthesis preserving body motion, gestures, "
    "facial expressions, scene composition, and audio with multi-style 2D rendering and temporal anti-flicker stabilization."
)

# Supported Video File Formats
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_VIDEO_SIZE_MB = 250
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024

# Supported 2D Cartoon Styles
CARTOON_STYLES = {
    "Classic 2D Cartoon": {
        "id": "classic_2d",
        "description": "Bold clean ink lines, warm vibrant cel-shading, and playful color palette.",
        "prompt": "classic 2d animation style, crisp ink lines, flat cel shaded, vibrant colors, vintage cartoon, high quality, masterpiece",
        "line_strength": 1.2,
        "color_saturation": 1.35,
    },
    "Anime (Japanese Animation)": {
        "id": "anime",
        "description": "Makoto Shinkai & Kyoto Animation inspired, luminous atmospheric skies, delicate features, and soft aesthetic bloom.",
        "prompt": "anime visual novel style, makoto shinkai aesthetic, luminous lighting, vibrant anime sky, clean delicate line art, trending on pixiv, masterpiece",
        "line_strength": 0.85,
        "color_saturation": 1.25,
    },
    "Comic-Book Animation": {
        "id": "comic_book",
        "description": "Dramatic graphic novel inking, heavy shadows, high-contrast primary colors, and dynamic halftone contours.",
        "prompt": "modern comic book art style, graphic novel illustration, bold heavy ink contours, high contrast shadows, pop art colors, dynamic",
        "line_strength": 1.6,
        "color_saturation": 1.45,
    },
    "Hand-Drawn Animation": {
        "id": "hand_drawn",
        "description": "Artistic graphite/charcoal pencil strokes, gentle watercolor wash, and organic paper grain texture.",
        "prompt": "hand-drawn sketch animation, pencil outlines, soft watercolor wash, textured paper aesthetic, traditional 2d artist sketch, organic",
        "line_strength": 1.1,
        "color_saturation": 0.95,
    },
    "Colorful Animated Movie": {
        "id": "animated_movie",
        "description": "Disney/Pixar-inspired vibrant 2D storybook style, soft ambient highlights, rich saturation, and clean render.",
        "prompt": "disney pixar 2d storybook animation style, rich vibrant colors, soft specular highlights, polished clean cartoon movie render, magical",
        "line_strength": 0.90,
        "color_saturation": 1.40,
    },
}

# Export Resolution Presets
RESOLUTION_PRESETS = {
    "Original": None,
    "720p HD (1280x720)": (1280, 720),
    "1080p Full HD (1920x1080)": (1920, 1080),
    "4K Ultra HD (3840x2160)": (3840, 2160),
}

# FFmpeg Video Encoding Quality Presets (CRF: lower is higher quality)
CRF_PRESETS = {
    "High Quality (CRF 18)": 18,
    "Balanced (CRF 22)": 22,
    "Economy (CRF 26)": 26,
}

# Default Pipeline Settings
DEFAULT_SETTINGS = {
    "style": "Classic 2D Cartoon",
    "engine": "High-Performance Neural Engine",
    "animation_intensity": 0.80,
    "line_weight": 1.0,
    "temporal_smoothing": True,
    "preserve_pose": True,
    "preserve_depth": True,
    "preserve_audio": True,
    "output_resolution": "Original",
    "quality_preset": "High Quality (CRF 18)",
    "watermark": True,
}

