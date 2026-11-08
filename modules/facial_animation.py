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
        self.ref_bbox: Optional[np.ndarray] = None
        self.ref_base_lip_state: Optional[LipState] = None
        self.ref_triangles: Optional[List[Tuple[int, int, int]]] = None

    def load_reference(
        self,
        image_bgr: np.ndarray,
        landmarks_68: np.ndarray,
        bbox: np.ndarray,
    ) -> None:
        """Initializes the authorized reference face identity and computes its base Delaunay mesh."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Reference image is empty.")
        if landmarks_68 is None or len(landmarks_68) < 68:
            raise ValueError("Reference image requires at least 68 detected facial landmarks.")

        self.ref_image = image_bgr.copy()
        self.ref_landmarks = landmarks_68[:, :2].astype(np.float32)
        self.ref_bbox = bbox.astype(np.float32)

        ref_h = max(1.0, float(bbox[3] - bbox[1]))
        self.ref_base_lip_state = self.lip_sync.analyze_mouth(self.ref_landmarks, ref_h)

        # Precompute Delaunay Triangles on reference landmarks + boundary anchor points
        self.ref_triangles = self._compute_delaunay_triangles(self.ref_landmarks, image_bgr.shape[:2])
        logger.info(
            "Reference identity loaded successfully. Triangulated into %d mesh facets.",
            len(self.ref_triangles),
        )

    def _compute_delaunay_triangles(
        self,
        points_68: np.ndarray,
        img_shape: Tuple[int, int],
    ) -> List[Tuple[int, int, int]]:
        """Computes Delaunay triangulation facets for the 68 facial points."""
        h, w = img_shape[:2]
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)

        # Insert 68 facial points
        for pt in points_68:
            px, py = float(pt[0]), float(pt[1])
            if 0 <= px < w and 0 <= py < h:
                subdiv.insert((px, py))

        triangle_list = subdiv.getTriangleList()
        triangles: List[Tuple[int, int, int]] = []

        for t in triangle_list:
            pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            indices = []
            valid = True
            for pt in pts:
                # Find matching landmark index
                dists = np.linalg.norm(points_68 - np.array(pt), axis=1)
                idx = int(np.argmin(dists))
                if dists[idx] < 1.5:
                    indices.append(idx)
                else:
                    valid = False
                    break
            if valid and len(indices) == 3 and len(set(indices)) == 3:
                triangles.append((indices[0], indices[1], indices[2]))

        return triangles

    def animate_face(
        self,
        source_landmarks: np.ndarray,
        source_frame_shape: Tuple[int, int],
        source_bbox: np.ndarray,
        detector_pose: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Drives the authorized reference identity using source facial motion parameters.

        Returns:
            (animated_reference_crop, adapted_landmarks)
        """
        if self.ref_image is None or self.ref_landmarks is None:
            raise RuntimeError("Reference face has not been loaded into animation pipeline.")

        pts_src = source_landmarks[:, :2].astype(np.float32)
        face_size = max(1.0, float(source_bbox[3] - source_bbox[1]))

        # 1. Analyze Source Motion
        eye_state = self.eye_sync.analyze_eyes(pts_src)
        lip_state = self.lip_sync.analyze_mouth(pts_src, face_size)
        expression = self.expression_tracker.analyze_expression(pts_src, face_size)
        head_pose = self.head_pose_estimator.estimate_pose(pts_src, source_frame_shape, detector_pose)

        # 2. Condition & Adapt Reference Landmarks
        adapted_pts = self.ref_landmarks.copy()

        # Eye & Blinking Synchronization
        if self.settings.get("eye_sync", True) or self.settings.get("blinking_sync", True):
            adapted_pts = self.eye_sync.adapt_target_landmarks(adapted_pts, eye_state)

        # Lip & Speech Synchronization
        if self.settings.get("lip_sync", True):
            adapted_pts = self.lip_sync.transfer_lip_motion(
                adapted_pts,
                lip_state,
                self.ref_base_lip_state,
            )

        # Expression Synchronization (Eyebrows, smile, emotional shifts)
        if self.settings.get("expression_sync", True):
            adapted_pts = self.expression_tracker.transfer_expression_to_landmarks(adapted_pts, expression)

        # 3. Piecewise Affine Triangle Mesh Warping
        warped_face = self._warp_mesh(
            src_img=self.ref_image,
            src_pts=self.ref_landmarks,
            dst_pts=adapted_pts,
            triangles=self.ref_triangles or [],
            out_shape=self.ref_image.shape[:2],
        )

        # 4. Head Pose Orientation Adjustments (perspective / rotation if enabled)
        if self.settings.get("head_movement_sync", True) and abs(head_pose.roll) > 1.0:
            h, w = warped_face.shape[:2]
            center = (w / 2.0, h / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, head_pose.roll * 0.7, 1.0)
            warped_face = cv2.warpAffine(
                warped_face,
                rot_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            # Transform landmarks
            ones = np.ones((len(adapted_pts), 1), dtype=np.float32)
            pts_homo = np.hstack([adapted_pts, ones])
            adapted_pts = (rot_mat @ pts_homo.T).T

        return warped_face, adapted_pts

    def _warp_mesh(
        self,
        src_img: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        triangles: List[Tuple[int, int, int]],
        out_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Piecewise affine warping over triangular mesh facets."""
        h, w = out_shape[:2]
        output = np.zeros((h, w, 3), dtype=np.uint8)

        # If no triangulation precomputed, fallback to full image
        if not triangles:
            return src_img.copy()

        for tri in triangles:
            idx0, idx1, idx2 = tri
            t_src = np.float32([src_pts[idx0], src_pts[idx1], src_pts[idx2]])
            t_dst = np.float32([dst_pts[idx0], dst_pts[idx1], dst_pts[idx2]])

            self._warp_single_triangle(src_img, output, t_src, t_dst)

        # Fill any background regions outside the mesh with source pixels
        mask = (output > 0).any(axis=-1)
        output[~mask] = src_img[~mask]
        return output

    @staticmethod
    def _warp_single_triangle(
        img_src: np.ndarray,
        img_dst: np.ndarray,
        t_src: np.ndarray,
        t_dst: np.ndarray,
    ) -> None:
        """Warps one triangle from img_src into img_dst using affine bounding box."""
        r_dst = cv2.boundingRect(t_dst)
        r_src = cv2.boundingRect(t_src)

        dx, dy, dw, dh = r_dst
        sx, sy, sw, sh = r_src

        if dw <= 0 or dh <= 0 or sw <= 0 or sh <= 0:
            return

        # Coordinates relative to bounding boxes
        t_dst_local = np.array([
            [t_dst[0][0] - dx, t_dst[0][1] - dy],
            [t_dst[1][0] - dx, t_dst[1][1] - dy],
            [t_dst[2][0] - dx, t_dst[2][1] - dy],
        ], dtype=np.float32)

        t_src_local = np.array([
            [t_src[0][0] - sx, t_src[0][1] - sy],
            [t_src[1][0] - sx, t_src[1][1] - sy],
            [t_src[2][0] - sx, t_src[2][1] - sy],
        ], dtype=np.float32)

        # Extract source triangle crop
        src_crop = img_src[max(0, sy) : min(img_src.shape[0], sy + sh), max(0, sx) : min(img_src.shape[1], sx + sw)]
        if src_crop.size == 0:
            return

        warp_mat = cv2.getAffineTransform(t_src_local, t_dst_local)
        warped_crop = cv2.warpAffine(
            src_crop,
            warp_mat,
            (dw, dh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Triangle raster mask
        tri_mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, t_dst_local.astype(np.int32), 255)

        # Composite into destination
        dst_y1, dst_y2 = max(0, dy), min(img_dst.shape[0], dy + dh)
        dst_x1, dst_x2 = max(0, dx), min(img_dst.shape[1], dx + dw)

        crop_h = dst_y2 - dst_y1
        crop_w = dst_x2 - dst_x1
        if crop_h <= 0 or crop_w <= 0:
            return

        sub_mask = tri_mask[:crop_h, :crop_w]
        sub_warped = warped_crop[:crop_h, :crop_w]

        dst_slice = img_dst[dst_y1:dst_y2, dst_x1:dst_x2]
        np.copyto(dst_slice, sub_warped, where=sub_mask[..., None] > 0)

    def reset(self) -> None:
        self.eye_sync.reset()
        self.lip_sync.reset()
        self.expression_tracker.reset()
        self.head_pose_estimator.reset()
