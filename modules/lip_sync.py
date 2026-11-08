from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from utils.logging_config import get_logger

logger = get_logger("lip_sync")


@dataclass
class LipState:
    """Lip and mouth motion parameters extracted from video frames."""

    mouth_aspect_ratio: float  # Outer vertical to horizontal opening ratio
    inner_opening_ratio: float  # Inner aperture between upper & lower lip
    mouth_width_ratio: float    # Width normalized by face width
    smile_intensity: float      # Corner upturn & lateral stretch
    jaw_drop_ratio: float       # Vertical distance to chin
    is_speaking: bool           # True when mouth opening exceeds speech threshold


class LipSynchronizer:
    """Tracks, analyzes, and transfers mouth, lip, and jaw dynamics for speech synchronization."""

    SPEECH_MAR_THRESHOLD = 0.12

    def __init__(self, smoothing_alpha: float = 0.7):
        self.smoothing_alpha = smoothing_alpha
        self.prev_state: Optional[LipState] = None

    def analyze_mouth(self, landmarks_68: np.ndarray, face_size: float) -> LipState:
        """Extracts comprehensive mouth aperture, smile, and jaw metrics."""
        if landmarks_68 is None or len(landmarks_68) < 68:
            return LipState(0.0, 0.0, 0.4, 0.0, 0.5, False)

        face_h = max(1.0, face_size)

        # Outer mouth points: 48 to 59
        p48, p54 = landmarks_68[48], landmarks_68[54]
        mouth_w = np.linalg.norm(p54 - p48)

        p50, p58 = landmarks_68[50], landmarks_68[58]
        p51, p57 = landmarks_68[51], landmarks_68[57]
        p52, p56 = landmarks_68[52], landmarks_68[56]

        v1 = np.linalg.norm(p50 - p58)
        v2 = np.linalg.norm(p51 - p57)
        v3 = np.linalg.norm(p52 - p56)
        vertical_avg = (v1 + v2 + v3) / 3.0

        mar = float(vertical_avg / max(1.0, mouth_w))

        # Inner mouth opening (points 62 and 66)
        p62, p66 = landmarks_68[62], landmarks_68[66]
        inner_gap = np.linalg.norm(p66 - p62)
        inner_ratio = float(inner_gap / max(1.0, mouth_w))

        mouth_width_ratio = float(mouth_w / face_h)

        # Smile metric: corners (48, 54) Y position relative to lip center (51, 57)
        lip_center_y = (p51[1] + p57[1]) / 2.0
        corner_avg_y = (p48[1] + p54[1]) / 2.0
        # When smiling, corners are higher than center in image coordinates (lower Y)
        smile_lift = float((lip_center_y - corner_avg_y) / max(1.0, face_h * 0.1))
        smile_intensity = float(np.clip(smile_lift * 0.7 + (mouth_width_ratio - 0.4) * 1.5, 0.0, 1.0))

        # Jaw drop: distance from nose tip (30) or nose base (33) to chin (8)
        nose_base = landmarks_68[33]
        chin = landmarks_68[8]
        jaw_drop_ratio = float(np.linalg.norm(chin - nose_base) / face_h)

        is_speaking = mar > self.SPEECH_MAR_THRESHOLD or inner_ratio > 0.06

        # Temporal smoothing
        if self.prev_state is not None:
            mar = self.smoothing_alpha * mar + (1 - self.smoothing_alpha) * self.prev_state.mouth_aspect_ratio
            inner_ratio = (
                self.smoothing_alpha * inner_ratio
                + (1 - self.smoothing_alpha) * self.prev_state.inner_opening_ratio
            )
            mouth_width_ratio = (
                self.smoothing_alpha * mouth_width_ratio
                + (1 - self.smoothing_alpha) * self.prev_state.mouth_width_ratio
            )
            smile_intensity = (
                self.smoothing_alpha * smile_intensity
                + (1 - self.smoothing_alpha) * self.prev_state.smile_intensity
            )
            jaw_drop_ratio = (
                self.smoothing_alpha * jaw_drop_ratio
                + (1 - self.smoothing_alpha) * self.prev_state.jaw_drop_ratio
            )

        state = LipState(
            mouth_aspect_ratio=mar,
            inner_opening_ratio=inner_ratio,
            mouth_width_ratio=mouth_width_ratio,
            smile_intensity=smile_intensity,
            jaw_drop_ratio=jaw_drop_ratio,
            is_speaking=is_speaking,
        )
        self.prev_state = state
        return state

    def transfer_lip_motion(
        self,
        ref_landmarks: np.ndarray,
        source_state: LipState,
        ref_base_state: Optional[LipState] = None,
    ) -> np.ndarray:
        """Adapts reference face mouth landmarks to articulate source speech and lip shape."""
        adapted = ref_landmarks.copy()
        if len(adapted) < 68:
            return adapted

        # Reference mouth dimensions
        ref_p48, ref_p54 = adapted[48], adapted[54]
        ref_mouth_w = max(1.0, np.linalg.norm(ref_p54 - ref_p48))
        mouth_center = (ref_p48 + ref_p54) / 2.0

        # Target vertical opening
        ref_mar = ref_base_state.mouth_aspect_ratio if ref_base_state else 0.10
        opening_delta = (source_state.mouth_aspect_ratio - ref_mar) * ref_mouth_w

        # Lower lip displacement (points 55, 56, 57, 58, 59 and inner 65, 66, 67)
        lower_lip_indices = [55, 56, 57, 58, 59, 65, 66, 67]
        upper_lip_indices = [49, 50, 51, 52, 53, 61, 62, 63]

        if opening_delta > 0:
            # Jaw moves downward with lower lip (70% lower lip, 30% upper lip elevation)
            for idx in lower_lip_indices:
                adapted[idx, 1] += opening_delta * 0.70
            for idx in upper_lip_indices:
                adapted[idx, 1] -= opening_delta * 0.30

            # Chin follows jaw drop
            adapted[8, 1] += opening_delta * 0.45
            adapted[7, 1] += opening_delta * 0.35
            adapted[9, 1] += opening_delta * 0.35
        elif opening_delta < 0:
            # Subtle mouth closure
            for idx in lower_lip_indices:
                adapted[idx, 1] += opening_delta * 0.5
            for idx in upper_lip_indices:
                adapted[idx, 1] -= opening_delta * 0.2

        # Smile corner adjustment
        if source_state.smile_intensity > 0.05:
            smile_lift = source_state.smile_intensity * (ref_mouth_w * 0.12)
            smile_spread = source_state.smile_intensity * (ref_mouth_w * 0.06)
            # Left corner (48): move left and up
            adapted[48, 0] -= smile_spread
            adapted[48, 1] -= smile_lift
            adapted[60, 0] -= smile_spread * 0.8
            adapted[60, 1] -= smile_lift * 0.8
            # Right corner (54): move right and up
            adapted[54, 0] += smile_spread
            adapted[54, 1] -= smile_lift
            adapted[64, 0] += smile_spread * 0.8
            adapted[64, 1] -= smile_lift * 0.8

        return adapted

    def reset(self) -> None:
        self.prev_state = None

