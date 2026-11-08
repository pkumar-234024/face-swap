from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from modules.face_aligner import align_face, create_face_mask


class FaceReplacementPipeline:
    def __init__(self, detector, tracker=None):
        self.detector = detector
        self.tracker = tracker
        self.reference_image: Optional[np.ndarray] = None
        self.reference_landmarks: Optional[np.ndarray] = None
        self.reference_face: Optional[np.ndarray] = None

    def load_reference(self, reference_image: np.ndarray) -> None:
        if reference_image is None or reference_image.size == 0:
            raise ValueError("Reference image is empty.")

        detected = self.detector.detect_best_face(reference_image)
        if detected is None:
            raise ValueError("No clear face was detected in the reference image.")

        self.reference_image = reference_image.copy()
        self.reference_landmarks = np.asarray(detected["kps"], dtype=np.float32)

        x1, y1, x2, y2 = np.asarray(detected["bbox"], dtype=np.int32)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(reference_image.shape[1], x2)
        y2 = min(reference_image.shape[0], y2)
        self.reference_face = reference_image[y1:y2, x1:x2].copy()

        if self.reference_face.size == 0:
            raise ValueError("The reference face crop is empty. Please upload a clearer face image.")

    def _color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        source_mean = source_lab.mean(axis=(0, 1), keepdims=True)
        target_mean = target_lab.mean(axis=(0, 1), keepdims=True)
        source_std = source_lab.std(axis=(0, 1), keepdims=True) + 1e-5
        target_std = target_lab.std(axis=(0, 1), keepdims=True) + 1e-5

        adjusted = ((source_lab - source_mean) / source_std) * target_std + target_mean
        adjusted = np.clip(adjusted, 0, 255)
        return cv2.cvtColor(adjusted.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _blend_face(self, target_crop: np.ndarray, source_face: np.ndarray) -> np.ndarray:
        h, w = target_crop.shape[:2]
        source_resized = cv2.resize(source_face, (w, h), interpolation=cv2.INTER_CUBIC)
        source_adjusted = self._color_transfer(source_resized, target_crop)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (max(15, int(w * 0.42)), max(15, int(h * 0.55))), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (35, 35), 0)
        mask_inv = cv2.bitwise_not(mask)

        target_lab = target_crop.copy()
        result = target_lab.copy().astype(np.float32)
        source_lab = source_adjusted.astype(np.float32)

        result = cv2.bitwise_and(result.astype(np.uint8), result.astype(np.uint8), mask=mask_inv)
        source_region = cv2.bitwise_and(source_lab.astype(np.uint8), source_lab.astype(np.uint8), mask=mask)
        blended = cv2.add(result.astype(np.uint8), source_region)
        return blended

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        if self.reference_image is None or self.reference_face is None:
            return frame

        detection = self.detector.detect_best_face(frame)
        if detection is None:
            return frame

        bbox = np.asarray(detection["bbox"], dtype=np.float32)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        width = max(10, x2 - x1)
        height = max(10, y2 - y1)

        if self.tracker is not None:
            tracked = self.tracker.update(frame, (x1, y1, width, height), np.asarray(detection["kps"], dtype=np.float32))
            if tracked[0] is not None:
                x1, y1, width, height = tracked[0]
                x2 = x1 + width
                y2 = y1 + height

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return frame

        target_crop = frame[y1:y2, x1:x2]
        if target_crop.size == 0:
            return frame

        source_face = cv2.resize(self.reference_face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
        blended_crop = self._blend_face(target_crop, source_face)

        output = frame.copy()
        output[y1:y2, x1:x2] = blended_crop
        return output
