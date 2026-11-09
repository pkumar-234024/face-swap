from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

class TemporalStabilizer:
    """Eliminates frame-to-frame flickering using dense optical flow motion compensation."""

    def __init__(self, smoothing_factor: float = 0.70):
        # alpha is weight for current frame (0.65 to 0.85 typical)
        self.alpha = float(np.clip(smoothing_factor, 0.40, 0.95))
        self.prev_gray_orig: Optional[np.ndarray] = None
        self.prev_cartoon: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal frame memory (used at scene cuts or stream start)."""
        self.prev_gray_orig = None
        self.prev_cartoon = None

    def stabilize_frame(
        self,
        current_cartoon_bgr: np.ndarray,
        current_orig_bgr: np.ndarray,
        is_scene_cut: bool = False
    ) -> np.ndarray:
        """Stabilize current cartoon frame against previous frame using motion compensation."""
        if is_scene_cut or self.prev_cartoon is None or self.prev_gray_orig is None:
            self.prev_cartoon = current_cartoon_bgr.copy()
            self.prev_gray_orig = cv2.cvtColor(current_orig_bgr, cv2.COLOR_BGR2GRAY)
            return current_cartoon_bgr

        curr_gray = cv2.cvtColor(current_orig_bgr, cv2.COLOR_BGR2GRAY)
        h, w = curr_gray.shape[:2]

        # Downsample for fast, robust optical flow calculation
        scale = 0.5
        small_w = max(64, int(w * scale))
        small_h = max(64, int(h * scale))

        small_prev = cv2.resize(self.prev_gray_orig, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_curr = cv2.resize(curr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Calculate dense optical flow (Farneback)
        flow_small = cv2.calcOpticalFlowFarneback(
            small_prev,
            small_curr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Upscale flow vectors to full resolution
        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR) / scale

        # Generate remap coordinates
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y - flow[:, :, 1]).astype(np.float32)

        # Motion-warp previous cartoon frame to align with current frame
        warped_prev = cv2.remap(
            self.prev_cartoon,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # Compute photometric motion residual between warped and current original
        warped_orig = cv2.remap(
            self.prev_gray_orig,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        residual = cv2.absdiff(curr_gray, warped_orig)
        # Weight map: lower residual -> higher smoothing; high residual (fast motion/occlusion) -> keep current
        norm_res = (residual.astype(np.float32) / 255.0)[:, :, np.newaxis]
        confidence = np.clip(1.0 - norm_res * 2.0, 0.0, 1.0)

        # Blend warped previous cartoon and current cartoon
        effective_alpha = self.alpha + (1.0 - self.alpha) * (1.0 - confidence)
        stabilized = (
            effective_alpha * current_cartoon_bgr.astype(np.float32) +
            (1.0 - effective_alpha) * warped_prev.astype(np.float32)
        )
        stabilized_bgr = np.clip(stabilized, 0, 255).astype(np.uint8)

        # Update historical state
        self.prev_cartoon = stabilized_bgr.copy()
        self.prev_gray_orig = curr_gray.copy()

        return stabilized_bgr

