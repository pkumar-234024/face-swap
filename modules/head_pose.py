from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logging_config import get_logger

logger = get_logger("head_pose")


@dataclass
class HeadPose:
    """Estimated 3D head pose orientation angles and translation vector."""

    pitch: float  # Nodding up/down in degrees
    yaw: float    # Turning left/right in degrees
    roll: float   # Tilting left/right in degrees
    translation: np.ndarray  # Translation vector [tx, ty, tz]
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    confidence: float = 1.0
    is_extreme: bool = False


class HeadPoseEstimator:
    """Estimates 3D head orientation (Yaw, Pitch, Roll) using PnP and InsightFace pose."""

    # 3D Canonical Anthropometric Face Model coordinates (in millimeters)
    MODEL_POINTS_3D = np.array([
        [0.0, 0.0, 0.0],          # Nose tip (index 30)
        [0.0, -330.0, -65.0],     # Chin (index 8)
        [-225.0, 170.0, -135.0],  # Left eye outer corner (index 45)
        [225.0, 170.0, -135.0],   # Right eye outer corner (index 36)
        [-150.0, -150.0, -125.0], # Left mouth corner (index 54)
        [150.0, -150.0, -125.0],  # Right mouth corner (index 48)
    ], dtype=np.float64)

    def __init__(self, smoothing_alpha: float = 0.7):
        self.smoothing_alpha = smoothing_alpha
        self.prev_pose: Optional[HeadPose] = None

    def estimate_pose(
        self,
        landmarks_2d: np.ndarray,
        frame_shape: Tuple[int, int],
        detector_pose: Optional[np.ndarray] = None,
    ) -> HeadPose:
        """Estimates 3D head pose from 68 2D landmarks and image frame shape.

        Args:
            landmarks_2d: (68, 2) array of facial landmarks.
            frame_shape: (height, width) of the video frame.
            detector_pose: Optional [pitch, yaw, roll] direct from model.
        """
        h, w = frame_shape[:2]

        if landmarks_2d is None or len(landmarks_2d) < 68:
            # Fallback if landmarks unavailable
            return self._fallback_pose(detector_pose)

        # Extract 6 key correspondence points
        image_points = np.array([
            landmarks_2d[30],  # Nose tip
            landmarks_2d[8],   # Chin
            landmarks_2d[45],  # Left eye outer corner
            landmarks_2d[36],  # Right eye outer corner
            landmarks_2d[54],  # Left mouth corner
            landmarks_2d[48],  # Right mouth corner
        ], dtype=np.float64)

        # Camera intrinsic matrix approximation
        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return self._fallback_pose(detector_pose)

        rmat, _ = cv2.Rodrigues(rvec)

        # Compute Euler angles from rotation matrix
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch_rad = np.arctan2(rmat[2, 1], rmat[2, 2])
            yaw_rad = np.arctan2(-rmat[2, 0], sy)
            roll_rad = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch_rad = np.arctan2(-rmat[1, 2], rmat[1, 1])
            yaw_rad = np.arctan2(-rmat[2, 0], sy)
            roll_rad = 0.0

        pnp_pitch = float(np.degrees(pitch_rad))
        pnp_yaw = float(np.degrees(yaw_rad))
        pnp_roll = float(np.degrees(roll_rad))

        # Blend with detector_pose if available for extra stability
        if detector_pose is not None and len(detector_pose) == 3:
            det_pitch, det_yaw, det_roll = [float(v) for v in detector_pose]
            pitch = 0.6 * pnp_pitch + 0.4 * det_pitch
            yaw = 0.6 * pnp_yaw + 0.4 * det_yaw
            roll = 0.6 * pnp_roll + 0.4 * det_roll
        else:
            pitch, yaw, roll = pnp_pitch, pnp_yaw, pnp_roll

        # Temporal smoothing
        if self.prev_pose is not None:
            pitch = self.smoothing_alpha * pitch + (1 - self.smoothing_alpha) * self.prev_pose.pitch
            yaw = self.smoothing_alpha * yaw + (1 - self.smoothing_alpha) * self.prev_pose.yaw
            roll = self.smoothing_alpha * roll + (1 - self.smoothing_alpha) * self.prev_pose.roll

        is_extreme = abs(yaw) > 40.0 or abs(pitch) > 30.0
        confidence = 0.6 if is_extreme else 0.95

        pose = HeadPose(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            translation=tvec.flatten(),
            rotation_matrix=rmat,
            confidence=confidence,
            is_extreme=is_extreme,
        )

        self.prev_pose = pose
        return pose

    def _fallback_pose(self, detector_pose: Optional[np.ndarray]) -> HeadPose:
        if detector_pose is not None and len(detector_pose) == 3:
            pitch, yaw, roll = [float(v) for v in detector_pose]
        else:
            pitch, yaw, roll = 0.0, 0.0, 0.0

        return HeadPose(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            translation=np.zeros(3, dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
            confidence=0.5,
            is_extreme=False,
        )

    def reset(self) -> None:
        self.prev_pose = None

