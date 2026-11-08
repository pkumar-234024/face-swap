from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logging_config import get_logger

logger = get_logger("face_tracker")


class FaceTracker:
    """Temporal face tracker with motion smoothing, Kalman-like inertia, and lost-tracking recovery."""

    def __init__(self, smoothing_alpha: float = 0.75, max_missed_frames: int = 8):
        self.smoothing_alpha = smoothing_alpha
        self.max_missed_frames = max_missed_frames
        self.missed_frames = 0
        self.tracked_bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2]
        self.velocity: np.ndarray = np.zeros(4, dtype=np.float32)

    def update(
        self,
        frame: np.ndarray,
        detected_bbox: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Updates tracker state with frame and optional fresh detection box.

        Args:
            frame: Current video frame (BGR).
            detected_bbox: [x1, y1, x2, y2] float coordinates or None if detection missed.

        Returns:
            Smoothed [x1, y1, x2, y2] bounding box or None if track lost.
        """
        h, w = frame.shape[:2]

        if detected_bbox is not None:
            raw_box = np.asarray(detected_bbox, dtype=np.float32)
            self.missed_frames = 0

            if self.tracked_bbox is None:
                self.tracked_bbox = raw_box.copy()
                self.velocity = np.zeros(4, dtype=np.float32)
            else:
                # Calculate movement jump to prevent tracking identity-swapping jumps
                cur_center = np.array([
                    (raw_box[0] + raw_box[2]) / 2.0,
                    (raw_box[1] + raw_box[3]) / 2.0,
                ])
                prev_center = np.array([
                    (self.tracked_bbox[0] + self.tracked_bbox[2]) / 2.0,
                    (self.tracked_bbox[1] + self.tracked_bbox[3]) / 2.0,
                ])

                dist = np.linalg.norm(cur_center - prev_center)
                max_jump = max(w, h) * 0.45

                if dist > max_jump:
                    logger.debug("Large tracking jump detected (dist=%.1fpx). Resetting track.", dist)
                    self.tracked_bbox = raw_box.copy()
                    self.velocity = np.zeros(4, dtype=np.float32)
                else:
                    new_velocity = raw_box - self.tracked_bbox
                    self.velocity = 0.6 * new_velocity + 0.4 * self.velocity
                    self.tracked_bbox = (
                        self.smoothing_alpha * raw_box + (1 - self.smoothing_alpha) * self.tracked_bbox
                    )
        else:
            # Detection missed this frame: apply inertial dead-reckoning
            self.missed_frames += 1
            if self.tracked_bbox is not None and self.missed_frames <= self.max_missed_frames:
                # Coast using damped velocity
                self.velocity *= 0.7
                self.tracked_bbox += self.velocity
            else:
                self.tracked_bbox = None
                return None

        if self.tracked_bbox is None:
            return None

        # Clamp within frame boundaries
        box = self.tracked_bbox.copy()
        box[0] = np.clip(box[0], 0, w - 1)
        box[1] = np.clip(box[1], 0, h - 1)
        box[2] = np.clip(box[2], box[0] + 10, w)
        box[3] = np.clip(box[3], box[1] + 10, h)

        return box

    def reset(self) -> None:
        self.tracked_bbox = None
        self.velocity = np.zeros(4, dtype=np.float32)
        self.missed_frames = 0
