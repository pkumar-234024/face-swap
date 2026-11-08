from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from utils.logging_config import get_logger

logger = get_logger("eye_sync")


@dataclass
class EyeState:
    """State metrics for eye opening, blinking, and gaze direction."""

    left_ear: float
    right_ear: float
    avg_ear: float
    is_blinking: bool
    blink_progress: float  # 0.0 (wide open) to 1.0 (fully closed)
    gaze_horizontal: float  # -1.0 (far left) to +1.0 (far right)
    gaze_vertical: float    # -1.0 (looking up) to +1.0 (looking down)


class EyeSynchronizer:
    """Tracks eye aspect ratio (EAR), blinking events, and gaze orientation."""

    DEFAULT_BLINK_THRESHOLD = 0.21

    def __init__(self, blink_threshold: float = DEFAULT_BLINK_THRESHOLD, smoothing_alpha: float = 0.75):
        self.blink_threshold = blink_threshold
        self.smoothing_alpha = smoothing_alpha
        self.prev_state: Optional[EyeState] = None
        self.baseline_ear: float = 0.30

    @staticmethod
    def compute_ear(eye_points: np.ndarray) -> float:
        """Computes the Eye Aspect Ratio (EAR) for a 6-point eye contour.

        Points ordering: 0: outer corner, 1: top-left, 2: top-right,
                         3: inner corner, 4: bottom-right, 5: bottom-left.
        """
        if eye_points is None or len(eye_points) < 6:
            return 0.30

        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])

        if h < 1e-4:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def analyze_eyes(self, landmarks_68: np.ndarray) -> EyeState:
        """Extracts EAR, blinking progress, and gaze vectors from 68 landmarks."""
        if landmarks_68 is None or len(landmarks_68) < 68:
            return EyeState(0.3, 0.3, 0.3, False, 0.0, 0.0, 0.0)

        right_eye_pts = landmarks_68[36:42]
        left_eye_pts = landmarks_68[42:48]

        r_ear = self.compute_ear(right_eye_pts)
        l_ear = self.compute_ear(left_eye_pts)
        avg_ear = (r_ear + l_ear) / 2.0

        # Update baseline EAR dynamically when eyes are open
        if avg_ear > self.blink_threshold:
            self.baseline_ear = 0.95 * self.baseline_ear + 0.05 * avg_ear

        # Blink detection & progress
        is_blinking = avg_ear < self.blink_threshold
        blink_progress = np.clip(1.0 - (avg_ear / max(0.05, self.baseline_ear)), 0.0, 1.0)

        # Gaze estimation relative to eye bounding box
        # Right eye center vs corner midpoints
        r_center = right_eye_pts.mean(axis=0)
        r_h_center = (right_eye_pts[0][0] + right_eye_pts[3][0]) / 2.0
        r_width = max(1.0, abs(right_eye_pts[3][0] - right_eye_pts[0][0]))
        gaze_h = float(np.clip((r_center[0] - r_h_center) / (r_width * 0.25), -1.0, 1.0))

        r_v_center = (right_eye_pts[1][1] + right_eye_pts[4][1]) / 2.0
        r_height = max(1.0, abs(right_eye_pts[4][1] - right_eye_pts[1][1]))
        gaze_v = float(np.clip((r_center[1] - r_v_center) / (r_height * 0.3), -1.0, 1.0))

        # Temporal smoothing
        if self.prev_state is not None:
            avg_ear = self.smoothing_alpha * avg_ear + (1 - self.smoothing_alpha) * self.prev_state.avg_ear
            blink_progress = (
                self.smoothing_alpha * blink_progress
                + (1 - self.smoothing_alpha) * self.prev_state.blink_progress
            )
            gaze_h = self.smoothing_alpha * gaze_h + (1 - self.smoothing_alpha) * self.prev_state.gaze_horizontal
            gaze_v = self.smoothing_alpha * gaze_v + (1 - self.smoothing_alpha) * self.prev_state.gaze_vertical

        state = EyeState(
            left_ear=float(l_ear),
            right_ear=float(r_ear),
            avg_ear=float(avg_ear),
            is_blinking=is_blinking,
            blink_progress=float(blink_progress),
            gaze_horizontal=gaze_h,
            gaze_vertical=gaze_v,
        )
        self.prev_state = state
        return state

    def adapt_target_landmarks(
        self,
        ref_landmarks: np.ndarray,
        source_eye_state: EyeState,
    ) -> np.ndarray:
        """Adjusts reference face eye landmark coordinates to mirror source blinking and eye openness."""
        adapted = ref_landmarks.copy()
        if len(adapted) < 68:
            return adapted

        # Ratio to scale vertical aperture of reference eyes
        ref_r_ear = self.compute_ear(adapted[36:42])
        ref_l_ear = self.compute_ear(adapted[42:48])
        ref_avg = (ref_r_ear + ref_l_ear) / 2.0

        if ref_avg > 0.01:
            scale_factor = float(np.clip(source_eye_state.avg_ear / ref_avg, 0.05, 1.6))
        else:
            scale_factor = 1.0

        # Adjust Right Eye (36-41)
        r_center_y = (adapted[37, 1] + adapted[38, 1] + adapted[40, 1] + adapted[41, 1]) / 4.0
        adapted[37, 1] = r_center_y + (adapted[37, 1] - r_center_y) * scale_factor
        adapted[38, 1] = r_center_y + (adapted[38, 1] - r_center_y) * scale_factor
        adapted[40, 1] = r_center_y + (adapted[40, 1] - r_center_y) * scale_factor
        adapted[41, 1] = r_center_y + (adapted[41, 1] - r_center_y) * scale_factor

        # Adjust Left Eye (42-47)
        l_center_y = (adapted[43, 1] + adapted[44, 1] + adapted[46, 1] + adapted[47, 1]) / 4.0
        adapted[43, 1] = l_center_y + (adapted[43, 1] - l_center_y) * scale_factor
        adapted[44, 1] = l_center_y + (adapted[44, 1] - l_center_y) * scale_factor
        adapted[46, 1] = l_center_y + (adapted[46, 1] - l_center_y) * scale_factor
        adapted[47, 1] = l_center_y + (adapted[47, 1] - l_center_y) * scale_factor

        return adapted

    def reset(self) -> None:
        self.prev_state = None
        self.baseline_ear = 0.30

