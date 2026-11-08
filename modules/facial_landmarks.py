from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logging_config import get_logger

logger = get_logger("facial_landmarks")


# Landmark Index Indices for Standard 68-point model
JAW_INDICES = list(range(0, 17))
RIGHT_EYEBROW_INDICES = list(range(17, 22))
LEFT_EYEBROW_INDICES = list(range(22, 27))
NOSE_BRIDGE_INDICES = list(range(27, 31))
NOSE_TIP_INDICES = list(range(31, 36))
RIGHT_EYE_INDICES = list(range(36, 42))
LEFT_EYE_INDICES = list(range(42, 48))
OUTER_LIPS_INDICES = list(range(48, 60))
INNER_LIPS_INDICES = list(range(60, 68))


@dataclass
class FacialLandmarks:
    """Container for 2D/3D facial landmarks with regional subsets and temporal smoothing."""

    points_68: np.ndarray  # Shape (68, 2) or (68, 3)
    points_106: Optional[np.ndarray] = None  # Shape (106, 2)
    confidence: float = 1.0
    is_occluded: bool = False

    @property
    def points_2d(self) -> np.ndarray:
        """Returns (68, 2) 2D array."""
        return self.points_68[:, :2]

    @property
    def jaw(self) -> np.ndarray:
        return self.points_2d[JAW_INDICES]

    @property
    def right_eyebrow(self) -> np.ndarray:
        return self.points_2d[RIGHT_EYEBROW_INDICES]

    @property
    def left_eyebrow(self) -> np.ndarray:
        return self.points_2d[LEFT_EYEBROW_INDICES]

    @property
    def nose(self) -> np.ndarray:
        return self.points_2d[NOSE_BRIDGE_INDICES + NOSE_TIP_INDICES]

    @property
    def right_eye(self) -> np.ndarray:
        return self.points_2d[RIGHT_EYE_INDICES]

    @property
    def left_eye(self) -> np.ndarray:
        return self.points_2d[LEFT_EYE_INDICES]

    @property
    def outer_lips(self) -> np.ndarray:
        return self.points_2d[OUTER_LIPS_INDICES]

    @property
    def inner_lips(self) -> np.ndarray:
        return self.points_2d[INNER_LIPS_INDICES]

    @property
    def mouth(self) -> np.ndarray:
        return self.points_2d[OUTER_LIPS_INDICES + INNER_LIPS_INDICES]


class LandmarkSmoother:
    """Exponential Moving Average (EMA) and velocity-sensitive temporal smoother for landmarks."""

    def __init__(self, alpha: float = 0.65, max_jump_ratio: float = 0.35):
        self.alpha = alpha
        self.max_jump_ratio = max_jump_ratio
        self.prev_landmarks: Optional[np.ndarray] = None

    def smooth(self, current: np.ndarray, face_size: float) -> np.ndarray:
        """Smooths current landmark coordinates using previous frame history."""
        if current is None or len(current) == 0:
            return current

        if self.prev_landmarks is None or self.prev_landmarks.shape != current.shape:
            self.prev_landmarks = current.copy()
            return current

        diff = np.linalg.norm(current - self.prev_landmarks, axis=-1)
        max_allowed_jump = face_size * self.max_jump_ratio

        # Dynamic alpha: if motion is rapid (large diff), give more weight to current frame
        # to avoid sluggish lag during quick speaking or blinking.
        norm_diff = np.clip(np.mean(diff) / max(1.0, max_allowed_jump), 0.0, 1.0)
        dynamic_alpha = self.alpha + (1.0 - self.alpha) * (norm_diff ** 2)

        smoothed = dynamic_alpha * current + (1.0 - dynamic_alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed.copy()
        return smoothed

    def reset(self) -> None:
        self.prev_landmarks = None


class FacialLandmarkTracker:
    """Tracks, refines, and validates 68-point and dense 106-point facial landmarks."""

    def __init__(self, smoothing: bool = True):
        self.smoothing = smoothing
        self.smoother = LandmarkSmoother()
        self.history: List[FacialLandmarks] = []

    def extract_landmarks(
        self,
        face_data: Dict,
        frame_shape: Tuple[int, int],
    ) -> Optional[FacialLandmarks]:
        """Converts raw detector face dict into a validated, smoothed FacialLandmarks instance."""
        if not face_data:
            return None

        h, w = frame_shape[:2]
        pts_3d_68 = face_data.get("landmark_3d_68")
        pts_2d_106 = face_data.get("landmark_2d_106")
        kps_5 = face_data.get("kps")
        bbox = face_data.get("bbox")

        if bbox is not None:
            face_size = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
        else:
            face_size = max(w, h) * 0.3

        if pts_3d_68 is not None and len(pts_3d_68) == 68:
            points_68 = np.asarray(pts_3d_68, dtype=np.float32)
        elif pts_2d_106 is not None and len(pts_2d_106) == 106:
            # Downsample / map 106 points to 68 standard landmarks
            points_68 = self._map_106_to_68(pts_2d_106)
        elif kps_5 is not None and len(kps_5) >= 5 and bbox is not None:
            points_68 = self._synthesize_68_from_5(kps_5, bbox)
        else:
            return None

        # Check occlusion and bounds
        xs = points_68[:, 0]
        ys = points_68[:, 1]
        out_of_bounds = (xs < 0).sum() + (xs >= w).sum() + (ys < 0).sum() + (ys >= h).sum()
        is_occluded = out_of_bounds > 8

        # Apply temporal smoothing on 2D coordinates
        if self.smoothing:
            coords_2d = points_68[:, :2]
            smoothed_2d = self.smoother.smooth(coords_2d, face_size)
            points_68[:, :2] = smoothed_2d

        confidence = float(face_data.get("det_score", 1.0)) * (0.5 if is_occluded else 1.0)

        landmarks_obj = FacialLandmarks(
            points_68=points_68,
            points_106=pts_2d_106,
            confidence=confidence,
            is_occluded=is_occluded,
        )

        self.history.append(landmarks_obj)
        if len(self.history) > 30:
            self.history.pop(0)

        return landmarks_obj

    def _map_106_to_68(self, pts_106: np.ndarray) -> np.ndarray:
        """Interpolates standard 68 landmark array from 106-point dense face mesh."""
        pts = np.zeros((68, 2), dtype=np.float32)
        # Approximate mapping from 106 contour indices
        # Jaw (0-16):
        pts[0:17] = pts_106[np.linspace(0, 32, 17, dtype=int)]
        # Right Eyebrow (17-21):
        pts[17:22] = pts_106[np.linspace(33, 42, 5, dtype=int)]
        # Left Eyebrow (22-26):
        pts[22:27] = pts_106[np.linspace(43, 52, 5, dtype=int)]
        # Nose Bridge & Tip (27-35):
        pts[27:36] = pts_106[np.linspace(53, 62, 9, dtype=int)]
        # Right Eye (36-41):
        pts[36:42] = pts_106[np.linspace(63, 70, 6, dtype=int)]
        # Left Eye (42-47):
        pts[42:48] = pts_106[np.linspace(71, 78, 6, dtype=int)]
        # Outer Lips (48-59):
        pts[48:60] = pts_106[np.linspace(79, 90, 12, dtype=int)]
        # Inner Lips (60-67):
        pts[60:68] = pts_106[np.linspace(91, 98, 8, dtype=int)]
        return pts

    def _synthesize_68_from_5(self, kps: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Estimates 68 canonical landmarks using standard anthropometric proportions from 5 keypoints."""
        x1, y1, x2, y2 = bbox[:4]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        pts = np.zeros((68, 2), dtype=np.float32)

        # Jawline (0-16)
        for i in range(17):
            t = (i - 8) / 8.0
            pts[i] = [x1 + w * (0.5 + 0.48 * t), y1 + h * (0.35 + 0.65 * (t ** 2))]

        # Eyes from kps[0] (left) and kps[1] (right)
        le = kps[0]
        re = kps[1]
        eye_w = w * 0.12
        eye_h = h * 0.05

        # Right eye (36-41)
        pts[36] = [re[0] - eye_w, re[1]]
        pts[37] = [re[0] - eye_w * 0.5, re[1] - eye_h]
        pts[38] = [re[0] + eye_w * 0.5, re[1] - eye_h]
        pts[39] = [re[0] + eye_w, re[1]]
        pts[40] = [re[0] + eye_w * 0.5, re[1] + eye_h]
        pts[41] = [re[0] - eye_w * 0.5, re[1] + eye_h]

        # Left eye (42-47)
        pts[42] = [le[0] - eye_w, le[1]]
        pts[43] = [le[0] - eye_w * 0.5, le[1] - eye_h]
        pts[44] = [le[0] + eye_w * 0.5, le[1] - eye_h]
        pts[45] = [le[0] + eye_w, le[1]]
        pts[46] = [le[0] + eye_w * 0.5, le[1] + eye_h]
        pts[47] = [le[0] - eye_w * 0.5, le[1] + eye_h]

        # Eyebrows (17-26)
        brow_offset = h * 0.08
        pts[17:22] = pts[36:41] + np.array([0, -brow_offset])
        pts[22:27] = pts[42:47] + np.array([0, -brow_offset])

        # Nose (27-35)
        nose_pt = kps[2]
        pts[27] = [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2]
        pts[30] = nose_pt
        for i in range(28, 30):
            alpha = (i - 27) / 3.0
            pts[i] = (1 - alpha) * pts[27] + alpha * pts[30]
        for i, angle in enumerate(np.linspace(-0.6, 0.6, 5)):
            pts[31 + i] = [nose_pt[0] + w * 0.1 * np.sin(angle), nose_pt[1] + h * 0.05 * np.cos(angle)]

        # Mouth (48-67)
        lm = kps[3]
        rm = kps[4]
        mouth_center = (lm + rm) / 2.0
        m_w = np.linalg.norm(rm - lm) / 2.0
        m_h = h * 0.08

        for i, theta in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False)):
            pts[48 + i] = [mouth_center[0] - m_w * np.cos(theta), mouth_center[1] + m_h * np.sin(theta)]
        for i, theta in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
            pts[60 + i] = [mouth_center[0] - m_w * 0.6 * np.cos(theta), mouth_center[1] + m_h * 0.4 * np.sin(theta)]

        return pts

    def reset(self) -> None:
        self.smoother.reset()
        self.history.clear()

