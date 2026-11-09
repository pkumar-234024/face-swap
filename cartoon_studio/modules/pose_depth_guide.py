from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np

# OpenPose limb connections and color specification
POSE_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Right arm
    (1, 5), (5, 6), (6, 7),                # Left arm
    (1, 8), (8, 9), (9, 10),               # Right leg
    (1, 11), (11, 12), (12, 13),           # Left leg
    (0, 14), (14, 16),                     # Right eye & ear
    (0, 15), (15, 17)                      # Left eye & ear
]

LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170)
]

@dataclass
class GuidanceFrame:
    pose_canvas: np.ndarray        # OpenPose skeleton map (H, W, 3) BGR
    depth_canvas: np.ndarray       # Monocular depth map (H, W) or (H, W, 3)
    edge_canvas: np.ndarray        # Structural line composition map (H, W)
    has_person: bool = False

class PoseDepthGuide:
    """Extracts OpenPose skeleton motion and depth structure to guide animation generation."""

    def __init__(self, use_mediapipe: bool = True):
        self.mp_pose = None
        self.mp_drawing = None
        self.pose_detector = None
        
        if use_mediapipe:
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception:
                self.pose_detector = None

    def process_frame(self, frame_bgr: np.ndarray) -> GuidanceFrame:
        """Extract pose skeleton, depth composition, and edge structure from a frame."""
        h, w = frame_bgr.shape[:2]
        pose_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        has_person = False

        # 1. Pose landmark extraction
        if self.pose_detector is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.pose_detector.process(rgb)
                if results.pose_landmarks:
                    has_person = True
                    pose_canvas = self._render_openpose_skeleton(
                        results.pose_landmarks.landmark, w, h
                    )
            except Exception:
                has_person = False

        if not has_person:
            # Fallback human contour & silhouette detector for pose guidance
            pose_canvas, has_person = self._fallback_motion_silhouette(frame_bgr)

        # 2. Structural Depth estimation (multi-frequency gradient depth field)
        depth_canvas = self._estimate_structural_depth(frame_bgr)

        # 3. Composition edge map
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edge_canvas = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)

        return GuidanceFrame(
            pose_canvas=pose_canvas,
            depth_canvas=depth_canvas,
            edge_canvas=edge_canvas,
            has_person=has_person
        )

    def _render_openpose_skeleton(self, landmarks, w: int, h: int) -> np.ndarray:
        """Draw standard ControlNet-compatible OpenPose skeleton canvas."""
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        keypoints = []

        # Convert normalized coordinates to pixel space
        for lm in landmarks:
            if lm.visibility > 0.35:
                keypoints.append((int(lm.x * w), int(lm.y * h)))
            else:
                keypoints.append(None)

        # Draw limb connections
        # MediaPipe pose has 33 landmarks, map standard trunk and limbs
        mp_connections = [
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) # Face / eyes
        ]

        for i, (p1, p2) in enumerate(mp_connections):
            if p1 < len(keypoints) and p2 < len(keypoints):
                pt1 = keypoints[p1]
                pt2 = keypoints[p2]
                if pt1 is not None and pt2 is not None:
                    color = LIMB_COLORS[i % len(LIMB_COLORS)]
                    cv2.line(canvas, pt1, pt2, color, thickness=3, lineType=cv2.LINE_AA)

        # Draw joint nodes
        for pt in keypoints:
            if pt is not None:
                cv2.circle(canvas, pt, 4, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        return canvas

    def _fallback_motion_silhouette(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Produce a contour-based human motion guidance canvas if MediaPipe is inactive."""
        h, w = frame_bgr.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Skin color segmentation in YCrCb color space
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_human = False
        for c in contours:
            if cv2.contourArea(c) > (w * h * 0.015):
                has_human = True
                cv2.drawContours(canvas, [c], -1, (0, 255, 128), 2)
                
        return canvas, has_human

    def _estimate_structural_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Estimate structural scene depth field using multi-scale gradient and atmospheric attenuation."""
        # Converts frame to luminance
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape

        # Vertical atmospheric perspective prior (objects lower in frame are typically nearer)
        vertical_gradient = np.tile(np.linspace(0.15, 0.95, h)[:, np.newaxis], (1, w))

        # Local contrast & high frequency attenuation
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        detail = np.abs(gray - blur)
        detail_smooth = cv2.GaussianBlur(detail, (31, 31), 0)
        cv2.normalize(detail_smooth, detail_smooth, 0.0, 1.0, cv2.NORM_MINMAX)

        # Combine atmospheric gradient with local surface detail
        depth = 0.65 * vertical_gradient + 0.35 * detail_smooth
        depth_u8 = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        # Return 3-channel depth canvas
        return cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

    def release(self):
        """Release MediaPipe resources."""
        if self.pose_detector is not None:
            try:
                self.pose_detector.close()
            except Exception:
                pass
            self.pose_detector = None

