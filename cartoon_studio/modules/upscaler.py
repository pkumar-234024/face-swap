from __future__ import annotations

from typing import Tuple, Optional
import cv2
import numpy as np

class Upscaler:
    """High-quality resolution scaling and 4K upscaling for 2D cartoon animation."""

    @staticmethod
    def calculate_scaled_dimensions(
        orig_w: int,
        orig_h: int,
        target_preset: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """Calculate aspect-ratio preserved dimensions for target resolution."""
        if target_preset is None:
            # Keep original resolution (ensure even dimensions for H.264 encoder)
            even_w = orig_w if orig_w % 2 == 0 else orig_w - 1
            even_h = orig_h if orig_h % 2 == 0 else orig_h - 1
            return even_w, even_h

        target_w, target_h = target_preset
        aspect_orig = orig_w / orig_h
        aspect_target = target_w / target_h

        if aspect_orig > aspect_target:
            # Width limited
            new_w = target_w
            new_h = int(round(target_w / aspect_orig))
        else:
            # Height limited
            new_h = target_h
            new_w = int(round(target_h * aspect_orig))

        # Ensure divisible by 2 for H.264
        new_w = new_w if new_w % 2 == 0 else new_w - 1
        new_h = new_h if new_h % 2 == 0 else new_h - 1

        return max(2, new_w), max(2, new_h)

    @staticmethod
    def upscale_frame(
        frame_bgr: np.ndarray,
        target_dimensions: Tuple[int, int],
        sharpen_edges: bool = True
    ) -> np.ndarray:
        """Upscale frame to target dimensions using Lanczos-4 and selective edge sharpening."""
        target_w, target_h = target_dimensions
        cur_h, cur_w = frame_bgr.shape[:2]

        if cur_w == target_w and cur_h == target_h:
            return frame_bgr

        # Upscale using high-order Lanczos interpolation (ideal for crisp 2D line art)
        scaled = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Apply subtle unsharp masking for 4K / 1080p outputs to maintain ink line punch
        if sharpen_edges and (target_w >= 1920 or target_h >= 1080):
            gaussian = cv2.GaussianBlur(scaled, (0, 0), 2.0)
            unsharp = cv2.addWeighted(scaled, 1.25, gaussian, -0.25, 0)
            return np.clip(unsharp, 0, 255).astype(np.uint8)

        return scaled

