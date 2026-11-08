from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from modules.lighting_correction import LightingCorrector
from utils.logging_config import get_logger

logger = get_logger("face_blending")


class FaceBlender:
    """Seamless face blending using Multi-Band Laplacian Pyramids, Poisson cloning, and feathered alpha masking."""

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
        target_landmarks_roi: Optional[np.ndarray] = None,
        preserve_mouth_interior: bool = True,
        enhance_teeth: bool = True,
        method: str = "multiband",
    ) -> np.ndarray:
        """Seamlessly composites the warped face ROI into the target frame with natural teeth & tongue preservation.

        Args:
            target_frame: The full video frame (BGR).
            warped_roi: Warped reference face cropped to ROI.
            roi_mask: Binary facial mask within ROI coordinates.
            roi_bbox: (x1, y1, x2, y2) bounding box on target_frame.
            target_landmarks_roi: Optional facial landmarks in ROI space for teeth preservation.
            preserve_mouth_interior: If True, preserves real teeth and tongue during speech/singing.
            enhance_teeth: If True, applies subtle clarity sharpening to oral aperture.
            method: "multiband" (Laplacian pyramid), "poisson" (seamlessClone), or "feathered".
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

        # 2. Oral cavity mask extraction (teeth, tongue, mouth opening)
        mouth_open = False
        feathered_mouth = None
        if preserve_mouth_interior and target_landmarks_roi is not None and len(target_landmarks_roi) >= 68:
            inner_mouth_pts = target_landmarks_roi[60:68].astype(np.int32)
            inner_h = float(np.linalg.norm(target_landmarks_roi[62] - target_landmarks_roi[66]))
            mouth_w = float(np.linalg.norm(target_landmarks_roi[60] - target_landmarks_roi[64]))
            opening_ratio = inner_h / max(1.0, mouth_w)

            # When mouth is open for speech or singing, isolate teeth and tongue
            if opening_ratio > 0.030:
                mouth_open = True
                inner_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                cv2.fillPoly(inner_mask, [inner_mouth_pts], 255)
                # Adaptive feathering kernel scaled to opening height
                k_m = max(3, (int(inner_h * 0.18) | 1))
                feathered_mouth = cv2.GaussianBlur(inner_mask, (k_m, k_m), 0).astype(np.float32) / 255.0
                feathered_mouth = feathered_mouth[..., None]

        # 3. Blend face skin onto target_roi using chosen blending strategy
        if method == "multiband":
            blended_roi = self._laplacian_pyramid_blend(target_roi, color_corrected, roi_mask)
        elif method == "poisson":
            blended_roi = self._poisson_blend(target_roi, color_corrected, roi_mask)
        else:
            blended_roi = self._feathered_blend(target_roi, color_corrected, roi_mask)

        # 4. Temporal smoothing on the blended face to eliminate micro-flicker
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

        # 5. Composite authentic oral cavity (teeth and tongue) into the final blended face
        if mouth_open and feathered_mouth is not None:
            oral_source = target_roi.copy()
            if enhance_teeth:
                oral_source = self._enhance_oral_clarity(oral_source)

            blended_roi = (
                oral_source.astype(np.float32) * feathered_mouth
                + blended_roi.astype(np.float32) * (1.0 - feathered_mouth)
            ).astype(np.uint8)

        # 6. Composite back into the target video frame
        result_frame = target_frame.copy()
        result_frame[y1:y2, x1:x2] = blended_roi
        return result_frame

    def _enhance_oral_clarity(self, img: np.ndarray) -> np.ndarray:
        """Subtly sharpens dental enamel highlights and tongue texture."""
        gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(img, 1.25, gaussian, -0.25, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _laplacian_pyramid_blend(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray, num_levels: int = 4) -> np.ndarray:
        """Multi-Band Laplacian Pyramid Blending for seamless frequency-decomposed facial fusion."""
        try:
            h, w = target.shape[:2]
            if h < 16 or w < 16:
                return self._feathered_blend(target, source, mask)

            t = target.astype(np.float32) / 255.0
            s = source.astype(np.float32) / 255.0

            # Erode and feather mask for smooth seamless boundary transition
            k = max(9, (min(h, w) // 10) | 1)
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            m_gray = cv2.GaussianBlur(eroded, (k, k), 0).astype(np.float32) / 255.0
            m = np.repeat(m_gray[..., None], 3, axis=2)

            # Pad dimensions to be power-of-2 divisible
            div = 2 ** num_levels
            pad_h = (div - (h % div)) % div
            pad_w = (div - (w % div)) % div
            if pad_h > 0 or pad_w > 0:
                t = cv2.copyMakeBorder(t, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                s = cv2.copyMakeBorder(s, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                m = cv2.copyMakeBorder(m, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            # Generate Gaussian Pyramids
            t_pyr, s_pyr, m_pyr = [t], [s], [m]
            for _ in range(num_levels):
                t_pyr.append(cv2.pyrDown(t_pyr[-1]))
                s_pyr.append(cv2.pyrDown(s_pyr[-1]))
                m_pyr.append(cv2.pyrDown(m_pyr[-1]))

            # Generate Laplacian Pyramids
            t_lap, s_lap = [t_pyr[-1]], [s_pyr[-1]]
            for i in range(num_levels - 1, -1, -1):
                sz = (t_pyr[i].shape[1], t_pyr[i].shape[0])
                t_lap.append(t_pyr[i] - cv2.pyrUp(t_pyr[i + 1], dstsize=sz))
                s_lap.append(s_pyr[i] - cv2.pyrUp(s_pyr[i + 1], dstsize=sz))

            # Blend frequency bands
            blended_lap = [s_lap[0] * m_pyr[-1] + t_lap[0] * (1.0 - m_pyr[-1])]
            for i in range(1, len(t_lap)):
                mask_idx = num_levels - i
                b = s_lap[i] * m_pyr[mask_idx] + t_lap[i] * (1.0 - m_pyr[mask_idx])
                blended_lap.append(b)

            # Reconstruct from Laplacian pyramid
            blended = blended_lap[0]
            for i in range(1, len(blended_lap)):
                sz = (blended_lap[i].shape[1], blended_lap[i].shape[0])
                blended = cv2.pyrUp(blended, dstsize=sz) + blended_lap[i]

            if pad_h > 0 or pad_w > 0:
                blended = blended[:h, :w]

            return np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        except Exception as exc:
            logger.debug("Laplacian pyramid blend fallback to feathered: %s", exc)
            return self._feathered_blend(target, source, mask)

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
        k = max(9, (min(roi_w, roi_h) // 10) | 1)

        # Erode mask slightly before blurring so blend transition occurs inside the jaw/temples
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(mask, kernel, iterations=1)

        feathered = cv2.GaussianBlur(eroded, (k, k), 0).astype(np.float32) / 255.0
        feathered = np.expand_dims(feathered, axis=-1)

        blended = (source.astype(np.float32) * feathered + target.astype(np.float32) * (1.0 - feathered))
        return np.clip(blended, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        self.prev_blended = None
