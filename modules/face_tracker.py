from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class FaceTracker:
    def __init__(self) -> None:
        self.last_bbox: Optional[tuple[int, int, int, int]] = None
        self.last_landmarks: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray, bbox: Optional[tuple[int, int, int, int]], landmarks: Optional[np.ndarray]) -> tuple[Optional[tuple[int, int, int, int]], Optional[np.ndarray]]:
        if bbox is None:
            return None, landmarks

        x, y, w, h = bbox
        margin = 0.25
        x1 = max(0, int(x - w * margin))
        y1 = max(0, int(y - h * margin))
        x2 = min(frame.shape[1], int(x + w * (1 + margin)))
        y2 = min(frame.shape[0], int(y + h * (1 + margin)))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return bbox, landmarks

        if self.last_bbox is not None and self.last_landmarks is not None and landmarks is not None:
            dx = abs((x + w / 2) - (self.last_bbox[0] + self.last_bbox[2] / 2))
            dy = abs((y + h / 2) - (self.last_bbox[1] + self.last_bbox[3] / 2))
            if dx > frame.shape[1] * 0.4 or dy > frame.shape[0] * 0.4:
                return self.last_bbox, self.last_landmarks

        self.last_bbox = bbox
        self.last_landmarks = landmarks
        return bbox, landmarks

    def reset(self) -> None:
        self.last_bbox = None
        self.last_landmarks = None
