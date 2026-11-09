from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

@dataclass
class StyleProfile:
    name: str
    id: str
    description: str
    prompt: str
    bilateral_d: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    line_thickness: int
    edge_blur_ksize: int
    edge_threshold_low: int
    edge_threshold_high: int
    quantization_levels: int
    saturation_boost: float
    contrast_multiplier: float
    bloom_strength: float
    tint_bgr: Tuple[float, float, float]  # BGR channel multipliers

STYLE_PROFILES: Dict[str, StyleProfile] = {
    "Classic 2D Cartoon": StyleProfile(
        name="Classic 2D Cartoon",
        id="classic_2d",
        description="Bold clean ink lines, warm vibrant cel-shading, and flat color palette.",
        prompt="classic 2d animation, crisp bold ink lines, flat cel shaded, vibrant primary colors, vintage cartoon, high quality",
        bilateral_d=9,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
        line_thickness=2,
        edge_blur_ksize=5,
        edge_threshold_low=50,
        edge_threshold_high=150,
        quantization_levels=8,
        saturation_boost=1.35,
        contrast_multiplier=1.15,
        bloom_strength=0.0,
        tint_bgr=(0.98, 1.02, 1.05), # Slight warm golden tint
    ),
    "Anime (Japanese Animation)": StyleProfile(
        name="Anime (Japanese Animation)",
        id="anime",
        description="Makoto Shinkai & Kyoto Animation inspired, luminous atmospheric skies, delicate features, and soft aesthetic bloom.",
        prompt="anime visual novel aesthetic, makoto shinkai style, luminous vibrant anime sky, clean delicate line art, masterpiece",
        bilateral_d=7,
        bilateral_sigma_color=60.0,
        bilateral_sigma_space=60.0,
        line_thickness=1,
        edge_blur_ksize=3,
        edge_threshold_low=70,
        edge_threshold_high=180,
        quantization_levels=12,
        saturation_boost=1.28,
        contrast_multiplier=1.12,
        bloom_strength=0.25,
        tint_bgr=(1.08, 1.03, 0.96), # Luminous cyan-blue atmospheric anime tint
    ),
    "Comic-Book Animation": StyleProfile(
        name="Comic-Book Animation",
        id="comic_book",
        description="Dramatic graphic novel inking, heavy shadows, high-contrast primary colors, and dynamic halftone contours.",
        prompt="comic book art, graphic novel illustration, bold heavy ink contours, high contrast deep shadows, pop art colors",
        bilateral_d=11,
        bilateral_sigma_color=90.0,
        bilateral_sigma_space=90.0,
        line_thickness=3,
        edge_blur_ksize=5,
        edge_threshold_low=35,
        edge_threshold_high=120,
        quantization_levels=6,
        saturation_boost=1.45,
        contrast_multiplier=1.35,
        bloom_strength=0.0,
        tint_bgr=(0.95, 0.95, 1.10), # Punchy saturated comic red/yellow tone
    ),
    "Hand-Drawn Animation": StyleProfile(
        name="Hand-Drawn Animation",
        id="hand_drawn",
        description="Artistic graphite/charcoal pencil strokes, gentle watercolor wash, and organic paper grain texture.",
        prompt="hand-drawn sketch animation, charcoal pencil outlines, soft watercolor wash, textured paper aesthetic, traditional 2d",
        bilateral_d=5,
        bilateral_sigma_color=45.0,
        bilateral_sigma_space=45.0,
        line_thickness=2,
        edge_blur_ksize=3,
        edge_threshold_low=40,
        edge_threshold_high=130,
        quantization_levels=16,
        saturation_boost=0.92,
        contrast_multiplier=1.05,
        bloom_strength=0.08,
        tint_bgr=(0.96, 0.98, 1.02), # Soft parchment paper warm tone
    ),
    "Colorful Animated Movie": StyleProfile(
        name="Colorful Animated Movie",
        id="animated_movie",
        description="Disney/Pixar-inspired vibrant 2D storybook style, soft ambient highlights, rich saturation, and clean render.",
        prompt="disney pixar 2d storybook animation, rich vibrant colors, soft specular highlights, polished clean cartoon movie render",
        bilateral_d=9,
        bilateral_sigma_color=80.0,
        bilateral_sigma_space=80.0,
        line_thickness=1,
        edge_blur_ksize=5,
        edge_threshold_low=60,
        edge_threshold_high=160,
        quantization_levels=10,
        saturation_boost=1.40,
        contrast_multiplier=1.20,
        bloom_strength=0.18,
        tint_bgr=(1.02, 1.04, 1.06), # Vibrant storybook illumination
    ),
}

class StyleManager:
    """Manages cartoon style profiles and retrieves rendering parameters."""

    @staticmethod
    def get_style_names() -> list[str]:
        return list(STYLE_PROFILES.keys())

    @staticmethod
    def get_profile(style_name: str) -> StyleProfile:
        if style_name in STYLE_PROFILES:
            return STYLE_PROFILES[style_name]
        return STYLE_PROFILES["Classic 2D Cartoon"]

