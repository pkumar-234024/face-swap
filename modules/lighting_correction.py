from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from utils.logging_config import get_logger

logger = get_logger("lighting_correction")


class LightingCorrector:
    """Corrects ambient illumination, skin color balance, and preserves micro-textures."""

    @staticmethod
    def reinhard_color_transfer(
        source_face: np.ndarray,
        target_lighting: np.ndarray,
        strength: float = 1.0,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transfers color distribution in CIE L*a*b* space from target region to source face.

        If mask is provided, computes statistics strictly on the masked facial skin pixels.
        """
        if source_face is None or target_lighting is None:
            return source_face
        if source_face.size == 0 or target_lighting.size == 0:
            return source_face

        # Convert to float32 LAB
        src_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target_lighting, cv2.COLOR_BGR2LAB).astype(np.float32)

        if mask is not None and mask.shape[:2] == source_face.shape[:2] and (mask > 0).sum() > 50:
            mask_bool = mask > 0
            src_pixels = src_lab[mask_bool]
            tgt_pixels = tgt_lab[mask_bool]

            src_mean = np.mean(src_pixels, axis=0, keepdims=True)
            src_std = np.std(src_pixels, axis=0, keepdims=True) + 1e-5

            tgt_mean = np.mean(tgt_pixels, axis=0, keepdims=True)
            tgt_std = np.std(tgt_pixels, axis=0, keepdims=True) + 1e-5
        else:
            src_mean = np.mean(src_lab, axis=(0, 1), keepdims=True)
            src_std = np.std(src_lab, axis=(0, 1), keepdims=True) + 1e-5

            tgt_mean = np.mean(tgt_lab, axis=(0, 1), keepdims=True)
            tgt_std = np.std(tgt_lab, axis=(0, 1), keepdims=True) + 1e-5

        # Standardize source and scale by target distribution
        adjusted_lab = ((src_lab - src_mean) / src_std) * tgt_std + tgt_mean
        adjusted_lab = np.clip(adjusted_lab, 0, 255)

        if strength < 1.0:
            adjusted_lab = (1.0 - strength) * src_lab + strength * adjusted_lab

        adjusted_bgr = cv2.cvtColor(adjusted_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return adjusted_bgr

    @staticmethod
    def match_histograms(source: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Matches cumulative histogram of source image to match template illumination."""
        if source.size == 0 or template.size == 0:
            return source

        matched = np.zeros_like(source)
        for channel in range(3):
            src_c = source[:, :, channel]
            tmpl_c = template[:, :, channel]

            src_hist, _ = np.histogram(src_c.flatten(), 256, [0, 256])
            tmpl_hist, _ = np.histogram(tmpl_c.flatten(), 256, [0, 256])

            src_cdf = src_hist.cumsum().astype(np.float32)
            src_cdf /= src_cdf[-1] + 1e-7

            tmpl_cdf = tmpl_hist.cumsum().astype(np.float32)
            tmpl_cdf /= tmpl_cdf[-1] + 1e-7

            # Look-up table
            lut = np.interp(src_cdf, tmpl_cdf, np.arange(256)).astype(np.uint8)
            matched[:, :, channel] = cv2.LUT(src_c, lut)

        return matched

    @staticmethod
    def preserve_skin_texture(
        base_blended: np.ndarray,
        original_face: np.ndarray,
        texture_ratio: float = 0.20,
    ) -> np.ndarray:
        """Injects high-frequency pore and skin details to prevent waxy or plastic appearance."""
        if base_blended.shape != original_face.shape:
            return base_blended

        # Extract high-frequency components via unsharp mask on original
        blurred_orig = cv2.GaussianBlur(original_face, (5, 5), 1.5)
        high_freq = cv2.subtract(original_face, blurred_orig)

        # Blend high-frequency skin texture onto replacement
        enhanced = cv2.addWeighted(
            base_blended,
            1.0,
            high_freq,
            float(np.clip(texture_ratio, 0.0, 1.0)),
            0,
        )
        return enhanced
