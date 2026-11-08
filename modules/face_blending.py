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
        replacement_face: np.ndarray,
        target_bbox: np.ndarray,
        target_landmarks: Optional[np.ndarray] = None,
        method: str = "poisson",
    ) -> np.ndarray:
        """Seamlessly composites replacement face into the target frame.

        Args:
            target_frame: The full video frame (BGR).
            replacement_face: Cropped/warped replacement face image matching the target face scale.
            target_bbox: [x1, y1, x2, y2] bounding box on target_frame.
            target_landmarks: Optional 68 landmarks on target_frame.
            method: "poisson" (seamlessClone) or "pyramid" (Laplacian pyramid).
        """
        x1, y1, x2, y2 = [int(v) for v in target_bbox]
        h, w = target_frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 10 or box_h <= 10:
            return target_frame

        # Resize replacement to match bounding box exactly
        face_resized = cv2.resize(replacement_face, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
        target_roi = target_frame[y1:y2, x1:x2].copy()

        # Lighting and color transfer
        color_corrected = LightingCorrector.reinhard_color_transfer(face_resized, target_roi, strength=0.9)
        color_corrected = LightingCorrector.preserve_skin_texture(color_corrected, target_roi, texture_ratio=0.15)

        # Build mask for the ROI
        roi_mask = np.zeros((box_h, box_w), dtype=np.uint8)
        if target_landmarks is not None and len(target_landmarks) >= 17:
            # Shift landmarks to ROI coordinate frame
            roi_landmarks = target_landmarks[:, :2] - np.array([x1, y1], dtype=np.float32)
            # Clip within ROI
            roi_landmarks[:, 0] = np.clip(roi_landmarks[:, 0], 0, box_w - 1)
            roi_landmarks[:, 1] = np.clip(roi_landmarks[:, 1], 0, box_h - 1)
            hull = cv2.convexHull(roi_landmarks.astype(np.int32))
            cv2.fillPoly(roi_mask, [hull], 255)
            # Erode slightly so boundary stays within cheek/jawline
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            roi_mask = cv2.erode(roi_mask, kernel, iterations=1)
        else:
            center = (box_w // 2, box_h // 2)
            axes = (int(box_w * 0.40), int(box_h * 0.48))
            cv2.ellipse(roi_mask, center, axes, 0, 0, 360, 255, -1)

        # Try Poisson cloning if supported and valid
        if method == "poisson":
            blended_roi = self._poisson_blend(target_roi, color_corrected, roi_mask)
        else:
            blended_roi = self._pyramid_blend(target_roi, color_corrected, roi_mask)

        # Temporal smoothing to prevent flicker
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

        result_frame = target_frame.copy()
        result_frame[y1:y2, x1:x2] = blended_roi
        return result_frame

    def _poisson_blend(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Applies OpenCV Poisson Seamless Cloning."""
        try:
            h, w = target.shape[:2]
            center = (w // 2, h // 2)

            # Mask must be binary 0 or 255
            _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            if bin_mask.sum() == 0:
                return target

            cloned = cv2.seamlessClone(source, target, bin_mask, center, cv2.NORMAL_CLONE)
            return cloned
        except Exception as exc:
            logger.debug("Poisson clone fallback: %s", exc)
            return self._feathered_blend(target, source, mask)

    def _feathered_blend(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Feathered alpha blending using Gaussian blurred alpha mask."""
        k = max(7, (mask.shape[1] // 8) | 1)
        feathered = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
        feathered = np.expand_dims(feathered, axis=-1)

        blended = (source.astype(np.float32) * feathered + target.astype(np.float32) * (1.0 - feathered))
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _pyramid_blend(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray, levels: int = 4) -> np.ndarray:
        """Laplacian multi-band pyramid blending."""
        try:
            # Build Gaussian pyramid for mask
            mask_float = mask.astype(np.float32) / 255.0
            g_mask = [mask_float]
            for _ in range(levels):
                g_mask.append(cv2.pyrDown(g_mask[-1]))

            # Build Gaussian pyramids for source and target
            g_src = [source.astype(np.float32)]
            g_tgt = [target.astype(np.float32)]
            for _ in range(levels):
                g_src.append(cv2.pyrDown(g_src[-1]))
                g_tgt.append(cv2.pyrDown(g_tgt[-1]))

            # Build Laplacian pyramids
            lp_src = [g_src[levels]]
            lp_tgt = [g_tgt[levels]]
            for i in range(levels, 0, -1):
                sz = (g_src[i - 1].shape[1], g_src[i - 1].shape[0])
                lp_src.append(g_src[i - 1] - cv2.pyrUp(g_src[i], dstsize=sz))
                lp_tgt.append(g_tgt[i - 1] - cv2.pyrUp(g_tgt[i], dstsize=sz))

            # Blend pyramids
            blended_pyr = []
            for i in range(levels + 1):
                m = g_mask[levels - i] if i > 0 else g_mask[levels]
                if len(m.shape) == 2:
                    m = np.expand_dims(m, axis=-1)
                sz = (lp_src[i].shape[1], lp_src[i].shape[0])
                m_resized = cv2.resize(m, sz) if m.shape[:2] != sz[::-1] else m
                if len(m_resized.shape) == 2:
                    m_resized = np.expand_dims(m_resized, axis=-1)
                blended_pyr.append(lp_src[i] * m_resized + lp_tgt[i] * (1.0 - m_resized))

            # Reconstruct
            reconstructed = blended_pyr[0]
            for i in range(1, levels + 1):
                sz = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
                reconstructed = cv2.pyrUp(reconstructed, dstsize=sz) + blended_pyr[i]

            return np.clip(reconstructed, 0, 255).astype(np.uint8)
        except Exception as exc:
            logger.debug("Pyramid blend fallback: %s", exc)
            return self._feathered_blend(target, source, mask)

    def reset(self) -> None:
        self.prev_blended = None

