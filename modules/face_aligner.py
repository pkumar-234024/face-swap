from __future__ import annotations

import cv2
import numpy as np


def get_eye_center(landmarks: np.ndarray) -> tuple[float, float]:
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[45:51].mean(axis=0)
    return ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)


def compute_rotation_angle(landmarks: np.ndarray) -> float:
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[45:51].mean(axis=0)
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    return np.degrees(np.arctan2(delta_y, delta_x))


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: tuple[int, int] = (256, 256)) -> tuple[np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError("Input image is missing.")

    if landmarks.shape[0] < 68:
        raise ValueError("Face landmarks are insufficient for alignment.")

    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[45:51].mean(axis=0)
    eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    angle = compute_rotation_angle(landmarks)

    rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    left_eye_rot = cv2.transform(np.array([left_eye], dtype=np.float32), rot_matrix)[0]
    right_eye_rot = cv2.transform(np.array([right_eye], dtype=np.float32), rot_matrix)[0]
    eye_distance = np.linalg.norm(right_eye_rot - left_eye_rot)

    if eye_distance < 1.0:
        raise ValueError("Eye distance is too small to align the face.")

    reference_left = np.array([output_size[0] * 0.35, output_size[1] * 0.35], dtype=np.float32)
    reference_right = np.array([output_size[0] * 0.65, output_size[1] * 0.35], dtype=np.float32)

    src_pts = np.float32([left_eye_rot, right_eye_rot])
    dst_pts = np.float32([reference_left, reference_right])
    affine = cv2.getAffineTransform(src_pts, dst_pts)

    aligned = cv2.warpAffine(rotated, affine, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    adjusted_landmarks = np.array(landmarks, dtype=np.float32)
    aligned_landmarks = cv2.transform(adjusted_landmarks.reshape(-1, 1, 2), affine).reshape(-1, 2)
    return aligned, aligned_landmarks


def create_face_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if landmarks.shape[0] < 3:
        return np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(landmarks.astype(np.int32))
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return mask
