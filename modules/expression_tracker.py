from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from utils.logging_config import get_logger

logger = get_logger("expression_tracker")


@dataclass
class FacialExpression:
    """Quantitative facial expression coefficients and recognized emotion category."""

    smile: float            # 0.0 to 1.0 (happiness / smiling)
    laugh: float            # 0.0 to 1.0 (laughing: wide smile + open mouth)
    surprise: float         # 0.0 to 1.0 (raised brows + open mouth)
    sadness: float          # 0.0 to 1.0 (downward mouth corners + inner brow raise)
    anger: float            # 0.0 to 1.0 (lowered furrowed brows + tightened lips)
    brow_raise_left: float  # -1.0 (furrowed) to +1.0 (raised)
    brow_raise_right: float # -1.0 (furrowed) to +1.0 (raised)
    dominant_emotion: str   # Categorical label


class ExpressionTracker:
    """Tracks facial Action Units and emotional expressions across video frames."""

    def __init__(self, smoothing_alpha: float = 0.75):
        self.smoothing_alpha = smoothing_alpha
        self.prev_expression: Optional[FacialExpression] = None
        self.baseline_brow_dist: Optional[float] = None

    def analyze_expression(self, landmarks_68: np.ndarray, face_size: float) -> FacialExpression:
        """Computes expression coefficients and emotion classification from 68 landmarks."""
        if landmarks_68 is None or len(landmarks_68) < 68:
            return FacialExpression(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Neutral")

        face_h = max(1.0, face_size)

        # Eyebrows (Right 17-21, Left 22-26) vs Eyes (Right 36-41, Left 42-47)
        r_brow_y = landmarks_68[17:22, 1].mean()
        r_eye_y = landmarks_68[36:42, 1].mean()
        r_brow_dist = (r_eye_y - r_brow_y) / face_h

        l_brow_y = landmarks_68[22:27, 1].mean()
        l_eye_y = landmarks_68[42:48, 1].mean()
        l_brow_dist = (l_eye_y - l_brow_y) / face_h

        avg_brow_dist = (r_brow_dist + l_brow_dist) / 2.0
        if self.baseline_brow_dist is None:
            self.baseline_brow_dist = avg_brow_dist
        else:
            self.baseline_brow_dist = 0.98 * self.baseline_brow_dist + 0.02 * avg_brow_dist

        brow_raise_r = float(np.clip((r_brow_dist - self.baseline_brow_dist) / 0.08, -1.0, 1.0))
        brow_raise_l = float(np.clip((l_brow_dist - self.baseline_brow_dist) / 0.08, -1.0, 1.0))
        avg_brow_raise = (brow_raise_r + brow_raise_l) / 2.0

        # Mouth width and aperture
        p48, p54 = landmarks_68[48], landmarks_68[54]
        mouth_w = np.linalg.norm(p54 - p48) / face_h
        p51, p57 = landmarks_68[51], landmarks_68[57]
        mouth_h = np.linalg.norm(p57 - p51) / face_h

        # Smile: corner elevation relative to lip center
        lip_center_y = (p51[1] + p57[1]) / 2.0
        corner_y = (p48[1] + p54[1]) / 2.0
        corner_lift = (lip_center_y - corner_y) / (face_h * 0.08)
        smile = float(np.clip(corner_lift * 0.8 + (mouth_w - 0.38) * 2.0, 0.0, 1.0))

        # Laugh: high smile + high mouth opening
        laugh = float(np.clip(smile * 0.8 + (mouth_h - 0.15) * 2.5, 0.0, 1.0)) if smile > 0.4 else 0.0

        # Surprise: high brow raise + open mouth + round lips
        surprise = float(np.clip(avg_brow_raise * 0.7 + (mouth_h - 0.12) * 2.0, 0.0, 1.0))

        # Sadness: depressed mouth corners + slightly raised inner eyebrows
        corner_drop = (corner_y - lip_center_y) / (face_h * 0.08)
        sadness = float(np.clip(corner_drop * 1.2, 0.0, 1.0)) if corner_drop > 0.1 else 0.0

        # Anger: furrowed eyebrows (negative brow raise) + compressed lips
        anger = float(np.clip(-avg_brow_raise * 1.2 + (0.35 - mouth_h) * 1.5, 0.0, 1.0)) if avg_brow_raise < -0.2 else 0.0

        # Temporal smoothing
        if self.prev_expression is not None:
            smile = self.smoothing_alpha * smile + (1 - self.smoothing_alpha) * self.prev_expression.smile
            laugh = self.smoothing_alpha * laugh + (1 - self.smoothing_alpha) * self.prev_expression.laugh
            surprise = self.smoothing_alpha * surprise + (1 - self.smoothing_alpha) * self.prev_expression.surprise
            sadness = self.smoothing_alpha * sadness + (1 - self.smoothing_alpha) * self.prev_expression.sadness
            anger = self.smoothing_alpha * anger + (1 - self.smoothing_alpha) * self.prev_expression.anger
            brow_raise_r = (
                self.smoothing_alpha * brow_raise_r
                + (1 - self.smoothing_alpha) * self.prev_expression.brow_raise_right
            )
            brow_raise_l = (
                self.smoothing_alpha * brow_raise_l
                + (1 - self.smoothing_alpha) * self.prev_expression.brow_raise_left
            )

        # Categorize dominant emotion
        scores = {
            "Neutral": 0.25,
            "Smiling": smile,
            "Laughing": laugh,
            "Surprise": surprise,
            "Sadness": sadness,
            "Anger": anger,
        }
        dominant = max(scores, key=scores.get)

        expr = FacialExpression(
            smile=smile,
            laugh=laugh,
            surprise=surprise,
            sadness=sadness,
            anger=anger,
            brow_raise_left=brow_raise_l,
            brow_raise_right=brow_raise_r,
            dominant_emotion=dominant,
        )
        self.prev_expression = expr
        return expr

    def transfer_expression_to_landmarks(
        self,
        ref_landmarks: np.ndarray,
        expression: FacialExpression,
    ) -> np.ndarray:
        """Applies emotional expression shifts to reference eyebrow and cheek landmarks."""
        adapted = ref_landmarks.copy()
        if len(adapted) < 68:
            return adapted

        # Eyebrow shifts based on brow raise coefficients
        ref_h = max(1.0, np.linalg.norm(adapted[8] - adapted[27]))
        brow_scale = ref_h * 0.07

        # Right eyebrow (17-21)
        adapted[17:22, 1] -= expression.brow_raise_right * brow_scale
        # Left eyebrow (22-27)
        adapted[22:27, 1] -= expression.brow_raise_left * brow_scale

        # Furrow eyebrows closer horizontally when angry
        if expression.anger > 0.3:
            furrow_shift = expression.anger * (ref_h * 0.02)
            adapted[21, 0] += furrow_shift  # Inner right brow
            adapted[22, 0] -= furrow_shift  # Inner left brow

        # Sadness: lift inner eyebrows (21, 22) up while corners droop
        if expression.sadness > 0.3:
            sad_lift = expression.sadness * (ref_h * 0.035)
            adapted[21, 1] -= sad_lift
            adapted[22, 1] -= sad_lift
            # Droop mouth corners
            droop = expression.sadness * (ref_h * 0.03)
            adapted[48, 1] += droop
            adapted[54, 1] += droop

        return adapted

    def reset(self) -> None:
        self.prev_expression = None
        self.baseline_brow_dist = None

