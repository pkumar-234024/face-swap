from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logging_config import get_logger

logger = get_logger("face_aligner")


# Standard 5-point alignment reference coordinates for a 112x112 or 256x256 normalized crop
CANONICAL_5PTS_112 = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041],  # Right mouth corner
], dtype=np.float32)


def estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Computes optimal 2D similarity transform matrix [s*R | T] using Umeyama algorithm."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    num = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    var_src = np.mean(np.sum(src_centered ** 2, axis=1))
    if var_src < 1e-8:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    cov = (dst_centered.T @ src_centered) / num
    U, S, Vt = np.linalg.svd(cov)

    D = np.eye(2)
    if np.linalg.det(cov) < 0 or (np.linalg.det(U) * np.linalg.det(Vt) < 0):
        D[1, 1] = -1

    R = U @ D @ Vt
    scale = (np.trace(np.diag(S) @ D)) / var_src
    t = dst_mean - scale * (R @ src_mean)

    affine_matrix = np.zeros((2, 3), dtype=np.float32)
    affine_matrix[:2, :2] = scale * R
    affine_matrix[:2, 2] = t
    return affine_matrix


def align_face_to_canonical(
    image: np.ndarray,
    landmarks_5: np.ndarray,
    output_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns face to canonical upright pose using 5-point facial keypoints.

    Returns:
        (aligned_crop, affine_matrix, inverse_affine_matrix)
    """
    scale = output_size[0] / 112.0
    canonical_pts = CANONICAL_5PTS_112 * scale

    affine_mat = estimate_similarity_transform(landmarks_5[:5], canonical_pts)
    inv_mat = cv2.invertAffineTransform(affine_mat)

    aligned = cv2.warpAffine(
        image,
        affine_mat,
        output_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )

    return aligned, affine_mat, inv_mat


def create_face_mask(
    shape: Tuple[int, int],
    landmarks: np.ndarray,
    blur_kernel: int = 25,
    erosion_iter: int = 1,
) -> np.ndarray:
    """Generates a smooth, feathered facial blending mask from jaw and brow contour.

    Args:
        shape: (height, width) of the mask.
        landmarks: (N, 2) facial points.
        blur_kernel: Gaussian blur kernel size for smooth alpha transition.
        erosion_iter: Number of erosion iterations to keep mask within facial boundary.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if landmarks is None or len(landmarks) < 17:
        # Fallback to soft ellipse
        center = (w // 2, h // 2)
        axes = (int(w * 0.40), int(h * 0.50))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    else:
        # Build convex hull around landmarks (jaw + eyebrows)
        pts = landmarks[:, :2].astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 255)

    if erosion_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=erosion_iter)

    k = max(3, blur_kernel | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask
