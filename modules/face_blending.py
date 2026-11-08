from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from modules.lighting_correction import LightingCorrector
from utils.logging_config import get_logger

logger = get_logger("face_blending")


class FaceBlender:
    """Seamless face blending using Poisson cloning, Laplacian pyramids, and feathered alpha masking."""

    def __init__(self, temporal_smoothing: bool = True, smoothing_alpha: float = 0.85):
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_alpha = smoothing_alpha
        self.prev_blended: Optional[np.ndarray] = None

    def blend_face(
        self,
        target_frame: np.ndarray,
        warped_roi: np.ndarray,
        roi_mask: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        method: str = "poisson",
    ) -> np.ndarray:
        """Seamlessly composites the warped face ROI into the target frame.

        Args:
            target_frame: The full video frame (BGR).
            warped_roi: Warped reference face cropped to ROI.
            roi_mask: Binary facial mask within ROI coordinates.
            roi_bbox: (x1, y1, x2, y2) bounding box on target_frame.
            method: "poisson" (seamlessClone) or "feathered" (soft Gaussian alpha).
        """
        x1, y1, x2, y2 = [int(v) for v in roi_bbox]
        h, w = target_frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        roi_w = x2 - x1
        roi_h = y2 - y1

        if roi_w <= 10 or roi_h <= 10 or warped_roi.size == 0 or roi_mask.size == 0:
            return target_frame

        # Ensure warped_roi and roi_mask match ROI size
        if warped_roi.shape[:2] != (roi_h, roi_w):
            warped_roi = cv2.resize(warped_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        if roi_mask.shape[:2] != (roi_h, roi_w):
            roi_mask = cv2.resize(roi_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        target_roi = target_frame[y1:y2, x1:x2].copy()

        # 1. Masked Lighting and color correction on face skin
        color_corrected = LightingCorrector.reinhard_color_transfer(
            source_face=warped_roi,
            target_lighting=target_roi,
            strength=0.88,
            mask=roi_mask,
        )
        color_corrected = LightingCorrector.preserve_skin_texture(
            base_blended=color_corrected,
            original_face=target_roi,
            texture_ratio=0.15,
        )

        # 2. Blend inside the ROI
        if method == "poisson":
            blended_roi = self._poisson_blend(target_roi, color_corrected, roi_mask)
        else:
            blended_roi = self._feathered_blend(target_roi, color_corrected, roi_mask)

        # 3. Temporal smoothing to prevent flicker
        if self.temporal_smoothing and self.prev_blended is not None:
            if self.prev_blended.shape == blended_roi.shape:
                blended_roi = cv2.addWeighted(
                    blended_roi,
                    self.smoothing_alpha,
                    self.prev_blended,
                    1.0 - self.smoothing_alpha,
                    0,
                )
        self.prev_blended = blended_roi.copy()

        # 4. Composite back into the target video frame
        result_frame = target_frame.copy()
        result_frame[y1:y2, x1:x2] = blended_roi
        return result_frame

    def _poisson_blend(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Applies OpenCV Poisson Seamless Cloning with boundary erosion."""
        try:
            h, w = target.shape[:2]

            # Erode mask so boundary stays away from image edge
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            eroded = cv2.erode(mask, kernel, iterations=1)

            # Ensure 1-pixel zero border so seamlessClone never crashes on image boundaries
            eroded[0, :] = 0
            eroded[-1, :] = 0
            eroded[:, 0] = 0
            eroded[:, -1] = 0

            _, bin_mask = cv2.threshold(eroded, 127, 255, cv2.THRESH_BINARY)
            if bin_mask.sum() == 0:
                return target

            ys, xs = np.where(bin_mask > 0)
            center = (int(np.mean(xs)), int(np.mean(ys)))

            cloned = cv2.seamlessClone(source, target, bin_mask, center, cv2.NORMAL_CLONE)
            return cloned
        except Exception as exc:
            logger.debug("Poisson clone fallback: %s", exc)
            return self._feathered_blend(target, source, mask)

    def _feathered_blend(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Feathered alpha blending using Gaussian blurred alpha mask."""
        roi_h, roi_w = target.shape[:2]
        k = max(7, (min(roi_w, roi_h) // 12) | 1)

        # Erode mask slightly before blurring so blend transition occurs at the jaw/temples
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(mask, kernel, iterations=1)

        feathered = cv2.GaussianBlur(eroded, (k, k), 0).astype(np.float32) / 255.0
        feathered = np.expand_dims(feathered, axis=-1)

        blended = (source.astype(np.float32) * feathered + target.astype(np.float32) * (1.0 - feathered))
        return np.clip(blended, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        self.prev_blended = None
