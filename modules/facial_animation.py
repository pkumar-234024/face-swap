from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.expression_tracker import ExpressionTracker, FacialExpression
from modules.eye_sync import EyeState, EyeSynchronizer
from modules.head_pose import HeadPose, HeadPoseEstimator
from modules.lip_sync import LipState, LipSynchronizer
from utils.logging_config import get_logger

logger = get_logger("facial_animation")


def extend_face_landmarks(pts_68: np.ndarray) -> np.ndarray:
    """Extends 68 facial landmarks to 90 anatomical landmarks with a full cranial dome.

    Computes orientation-aligned cranial projection vectors along the chin-to-glabella axis
    so forehead and temple anchors rotate accurately with head roll, pitch, and yaw.
    """
    pts = pts_68[:, :2].astype(np.float32)
    chin = pts[8]
    glabella = (pts[21] + pts[22]) / 2.0

    # Cranial upward unit vector
    u = glabella - chin
    u_len = float(np.linalg.norm(u))
    u_hat = u / max(1.0, u_len)

    # Lateral unit vector (pointing to subject's left / image right)
    v_hat = np.array([-u_hat[1], u_hat[0]], dtype=np.float32)

    forehead_h = u_len * 0.40

    # Eyebrow landmark indices to project from: 17 to 26
    brow_idx = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    # Elliptical curvature coefficients (higher in central cranial dome, tapering at temples)
    curvatures = [0.65, 0.82, 0.94, 0.99, 1.00, 1.00, 0.99, 0.94, 0.82, 0.65]

    # Layer 1: Mid-forehead arch (10 points: indices 68 to 77)
    mid_pts = [
        pts[idx] + u_hat * (forehead_h * 0.48 * c)
        for idx, c in zip(brow_idx, curvatures)
    ]

    # Layer 2: Natural hairline contour arch (10 points: indices 78 to 87)
    hairline_pts = [
        pts[idx] + u_hat * (forehead_h * 0.95 * c)
        for idx, c in zip(brow_idx, curvatures)
    ]

    # Layer 3: Temple lateral anchors bridging eyebrows to upper jaw
    # Right temple (subject's right / viewer left: index 88)
    t_right = (pts[17] + pts[0]) / 2.0 + u_hat * (forehead_h * 0.20) - v_hat * (forehead_h * 0.08)
    # Left temple (subject's left / viewer right: index 89)
    t_left = (pts[26] + pts[16]) / 2.0 + u_hat * (forehead_h * 0.20) + v_hat * (forehead_h * 0.08)

    return np.vstack([pts, mid_pts, hairline_pts, [t_right, t_left]])


class FacialAnimationPipeline:
    """Identity-conditioned facial animation engine using Delaunay Piecewise Affine Warping."""

    def __init__(
        self,
        settings: Optional[Dict[str, bool | str]] = None,
    ):
        self.settings = settings or {
            "face_replacement": True,
            "lip_sync": True,
            "eye_sync": True,
            "blinking_sync": True,
            "expression_sync": True,
            "head_movement_sync": True,
            "lighting_correction": True,
            "seamless_blending": True,
            "temporal_smoothing": True,
            "ai_modified_label": True,
        }

        self.eye_sync = EyeSynchronizer()
        self.lip_sync = LipSynchronizer()
        self.expression_tracker = ExpressionTracker()
        self.head_pose_estimator = HeadPoseEstimator()

        self.ref_image: Optional[np.ndarray] = None
        self.ref_landmarks: Optional[np.ndarray] = None  # (68, 2)
        self.ref_extended_pts: Optional[np.ndarray] = None  # (90, 2)
        self.ref_bbox: Optional[np.ndarray] = None
        self.ref_base_lip_state: Optional[LipState] = None

    def load_reference(
        self,
        image_bgr: np.ndarray,
        landmarks_68: np.ndarray,
        bbox: np.ndarray,
    ) -> None:
        """Initializes authorized reference face identity and computes reference mesh points."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Reference image is empty.")
        if landmarks_68 is None or len(landmarks_68) < 68:
            raise ValueError("Reference image requires at least 68 detected facial landmarks.")

        self.ref_image = image_bgr.copy()
        self.ref_landmarks = landmarks_68[:, :2].astype(np.float32)
        self.ref_extended_pts = extend_face_landmarks(self.ref_landmarks)
        self.ref_bbox = bbox.astype(np.float32)

        ref_h = max(1.0, float(bbox[3] - bbox[1]))
        self.ref_base_lip_state = self.lip_sync.analyze_mouth(self.ref_landmarks, ref_h)

        logger.info("Reference identity loaded successfully with 90 anatomical cranial landmarks.")

    def warp_face_to_target(
        self,
        source_landmarks: np.ndarray,
        target_frame_shape: Tuple[int, int],
        source_bbox: np.ndarray,
        detector_pose: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], np.ndarray]:
        """Warps reference face directly onto the target facial geometry and pose.

        Returns:
            (warped_roi, roi_mask, (x1, y1, x2, y2), dst_roi_pts)
        """
        if self.ref_image is None or self.ref_extended_pts is None:
            raise RuntimeError("Reference face has not been loaded into animation pipeline.")

        pts_src = source_landmarks[:, :2].astype(np.float32)
        face_size = max(1.0, float(source_bbox[3] - source_bbox[1]))
        frame_h, frame_w = target_frame_shape[:2]

        # 1. Analyze motion parameters from source video frame
        eye_state = self.eye_sync.analyze_eyes(pts_src)
        lip_state = self.lip_sync.analyze_mouth(pts_src, face_size)
        expression = self.expression_tracker.analyze_expression(pts_src, face_size)
        _ = self.head_pose_estimator.estimate_pose(pts_src, target_frame_shape, detector_pose)

        # 2. Condition destination facial landmarks
        dst_pts = pts_src.copy()

        if self.settings.get("eye_sync", True) or self.settings.get("blinking_sync", True):
            dst_pts = self.eye_sync.adapt_target_landmarks(dst_pts, eye_state)

        if self.settings.get("lip_sync", True):
            dst_pts = self.lip_sync.transfer_lip_motion(
                dst_pts,
                lip_state,
                self.ref_base_lip_state,
            )

        if self.settings.get("expression_sync", True):
            dst_pts = self.expression_tracker.transfer_expression_to_landmarks(dst_pts, expression)

        # 3. Extend destination points with anatomical cranial dome (90 points)
        dst_extended = extend_face_landmarks(dst_pts)

        # 4. Calculate tight ROI around target face with padding
        hull_pts = cv2.convexHull(dst_extended.astype(np.int32))
        bx, by, bw, bh = cv2.boundingRect(hull_pts)
        pad = int(max(bw, bh) * 0.16)

        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(frame_w, bx + bw + pad)
        y2 = min(frame_h, by + bh + pad)

        roi_w = x2 - x1
        roi_h = y2 - y1

        if roi_w <= 10 or roi_h <= 10:
            blank_roi = np.zeros((10, 10, 3), dtype=np.uint8)
            blank_mask = np.zeros((10, 10), dtype=np.uint8)
            return blank_roi, blank_mask, (0, 0, 10, 10), dst_extended[:68]

        # Shift destination points into ROI coordinates
        dst_roi_pts = dst_extended - np.array([x1, y1], dtype=np.float32)

        # 5. Compute Delaunay Triangulation on target ROI coordinates
        rect = (0, 0, roi_w, roi_h)
        subdiv = cv2.Subdiv2D(rect)
        for pt in dst_roi_pts:
            px, py = float(pt[0]), float(pt[1])
            if 0 <= px < roi_w and 0 <= py < roi_h:
                subdiv.insert((px, py))

        triangles: List[Tuple[int, int, int]] = []
        for t in subdiv.getTriangleList():
            pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            indices = []
            valid = True
            for p in pts:
                dists = np.linalg.norm(dst_roi_pts - np.array(p), axis=1)
                idx = int(np.argmin(dists))
                if dists[idx] < 1.5:
                    indices.append(idx)
                else:
                    valid = False
                    break
            if valid and len(indices) == 3 and len(set(indices)) == 3:
                triangles.append((indices[0], indices[1], indices[2]))

        # 6. Warp each triangle from ref_image to warped_roi
        warped_roi = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        ref_pts = self.ref_extended_pts

        for i1, i2, i3 in triangles:
            src_tri = np.float32([ref_pts[i1], ref_pts[i2], ref_pts[i3]])
            dst_tri = np.float32([dst_roi_pts[i1], dst_roi_pts[i2], dst_roi_pts[i3]])

            r_dst = cv2.boundingRect(dst_tri)
            r_src = cv2.boundingRect(src_tri)

            dx, dy, dw, dh = r_dst
            sx, sy, sw, sh = r_src

            if dw <= 0 or dh <= 0 or sw <= 0 or sh <= 0:
                continue

            src_crop = self.ref_image[max(0, sy) : min(self.ref_image.shape[0], sy + sh), max(0, sx) : min(self.ref_image.shape[1], sx + sw)]
            if src_crop.shape[0] != sh or src_crop.shape[1] != sw:
                continue

            src_local = src_tri - np.array([sx, sy], dtype=np.float32)
            dst_local = dst_tri - np.array([dx, dy], dtype=np.float32)

            warp_mat = cv2.getAffineTransform(src_local, dst_local)
            warped_tri = cv2.warpAffine(
                src_crop,
                warp_mat,
                (dw, dh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            tri_mask = np.zeros((dh, dw), dtype=np.uint8)
            cv2.fillConvexPoly(tri_mask, dst_local.astype(np.int32), 255)

            dst_slice = warped_roi[dy : dy + dh, dx : dx + dw]
            sub_mask = tri_mask[: dst_slice.shape[0], : dst_slice.shape[1]]
            sub_warped = warped_tri[: dst_slice.shape[0], : dst_slice.shape[1]]

            np.copyto(dst_slice, sub_warped, where=sub_mask[..., None] > 0)

        # 7. Generate anatomical blending mask for ROI
        # Follow jawline (0..16) inset by 3.5% towards nose base, temple left, hairline, temple right
        center = dst_roi_pts[33]
        perimeter = []
        for i in range(0, 17):
            p = dst_roi_pts[i]
            inward = (center - p) * 0.035
            perimeter.append(p + inward)

        # Temple left (index 89)
        perimeter.append(dst_roi_pts[89])
        # Hairline contour arch from left to right (indices 87 down to 78)
        for idx in range(87, 77, -1):
            perimeter.append(dst_roi_pts[idx])
        # Temple right (index 88)
        perimeter.append(dst_roi_pts[88])

        poly = np.array(perimeter, dtype=np.int32)
        roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [poly], 255)

        return warped_roi, roi_mask, (x1, y1, x2, y2), dst_roi_pts

    def reset(self) -> None:
        self.eye_sync.reset()
        self.lip_sync.reset()
        self.expression_tracker.reset()
        self.head_pose_estimator.reset()
