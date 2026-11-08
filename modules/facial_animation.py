from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


class FacialAnimationPipeline:
    def __init__(self, detector, settings: Optional[Dict[str, bool | str]] = None):
        self.detector = detector
        self.settings = settings or {
            "face_replacement": True,
            "lip_sync": True,
            "eye_sync": True,
            "expression_sync": True,
            "head_movement_sync": True,
            "lighting_correction": True,
            "output_resolution": "Original",
            "ai_modified_label": True,
        }
        self.reference_image: Optional[np.ndarray] = None
        self.reference_face: Optional[np.ndarray] = None
        self.reference_bbox: Optional[np.ndarray] = None
        self.reference_kps: Optional[np.ndarray] = None
        self.reference_motion: Optional[Dict[str, float]] = None

    def load_reference(self, reference_image: np.ndarray) -> None:
        if reference_image is None or reference_image.size == 0:
            raise ValueError("Reference image is empty.")

        detection = self.detector.detect_best_face(reference_image)
        if detection is None:
            raise ValueError("No clear face was detected in the reference image.")

        bbox = np.asarray(detection["bbox"], dtype=np.float32)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(reference_image.shape[1], x2)
        y2 = min(reference_image.shape[0], y2)

        self.reference_image = reference_image.copy()
        self.reference_bbox = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        self.reference_kps = np.asarray(detection["kps"], dtype=np.float32)
        self.reference_face = reference_image[y1:y2, x1:x2].copy()

        if self.reference_face.size == 0:
            raise ValueError("The reference face crop is empty. Please upload a clearer face image.")

        self.reference_motion = self.extract_motion(self.reference_bbox, self.reference_kps)

    def _safe_point(self, points: np.ndarray, index: int) -> np.ndarray:
        if points is None or len(points) <= index:
            return np.array([0.0, 0.0], dtype=np.float32)
        return np.asarray(points[index], dtype=np.float32)

    def extract_motion(self, bbox: np.ndarray, kps: Optional[np.ndarray]) -> Dict[str, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        motion = {
            "mouth_open": 0.0,
            "smile": 0.0,
            "eye_open": 1.0,
            "blink": 0.0,
            "brow_motion": 0.0,
            "jaw_motion": 0.0,
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
            "face_width": face_w,
            "face_height": face_h,
            "center_x": center_x,
            "center_y": center_y,
        }

        if kps is None or len(kps) == 0:
            return motion

        p0 = self._safe_point(kps, 0)
        p1 = self._safe_point(kps, 1)
        p2 = self._safe_point(kps, 2)
        p3 = self._safe_point(kps, 3)
        p4 = self._safe_point(kps, 4)

        if len(kps) >= 5:
            mouth_left = p3
            mouth_right = p4
            mouth_center = (mouth_left + mouth_right) / 2.0
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            mouth_height = abs(p2[1] - mouth_center[1]) if len(p2) else 0.0
            mouth_open = np.clip((mouth_height / max(1.0, face_h)) * 5.0, 0.0, 1.0)
            smile = np.clip((mouth_width / max(1.0, face_w)) * 2.0, 0.0, 1.0)
            motion["mouth_open"] = mouth_open
            motion["smile"] = smile

            eye_mid = (p0 + p1) / 2.0
            eye_gap = np.linalg.norm(p1 - p0)
            eye_open = np.clip((eye_gap / max(1.0, face_w)) * 2.5, 0.0, 1.0)
            motion["eye_open"] = eye_open

            jaw_motion = np.clip((abs(p2[1] - center_y) / max(1.0, face_h)) * 4.0, 0.0, 1.0)
            motion["jaw_motion"] = jaw_motion

            brow_motion = np.clip((abs(p0[1] - p1[1]) / max(1.0, face_h)) * 5.0, 0.0, 1.0)
            motion["brow_motion"] = brow_motion

            motion["head_yaw"] = float(np.clip((p2[0] - center_x) / max(1.0, face_w), -1.0, 1.0))
            motion["head_pitch"] = float(np.clip((center_y - p2[1]) / max(1.0, face_h), -1.0, 1.0))
            motion["head_roll"] = float(np.clip(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) / 90.0, -1.0, 1.0))

        if len(kps) >= 68:
            left_eye = kps[36:42].mean(axis=0)
            right_eye = kps[45:51].mean(axis=0)
            eye_dist = np.linalg.norm(right_eye - left_eye)
            motion["eye_open"] = np.clip((eye_dist / max(1.0, face_w)) * 2.6, 0.0, 1.0)
            mouth_top = kps[61:68].mean(axis=0)
            mouth_bottom = kps[57:61].mean(axis=0).mean(axis=0)
            mouth_height = abs(mouth_top[1] - mouth_bottom[1])
            motion["mouth_open"] = np.clip((mouth_height / max(1.0, face_h)) * 5.0, 0.0, 1.0)
            motion["smile"] = np.clip((np.linalg.norm(kps[48] - kps[54]) / max(1.0, face_w)) * 2.0, 0.0, 1.0)

        return motion

    def _color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        if source.size == 0 or target.size == 0:
            return source

        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        source_mean = source_lab.mean(axis=(0, 1), keepdims=True)
        target_mean = target_lab.mean(axis=(0, 1), keepdims=True)
        source_std = source_lab.std(axis=(0, 1), keepdims=True) + 1e-5
        target_std = target_lab.std(axis=(0, 1), keepdims=True) + 1e-5
        adjusted = ((source_lab - source_mean) / source_std) * target_std + target_mean
        adjusted = np.clip(adjusted, 0, 255)
        return cv2.cvtColor(adjusted.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _build_face_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (max(30, width // 2), max(30, height // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return cv2.GaussianBlur(mask, (35, 35), 0)

    def _apply_lip_sync(self, face: np.ndarray, motion: Dict[str, float]) -> np.ndarray:
        if face.size == 0:
            return face

        h, w = face.shape[:2]
        mouth_x = int(w * 0.32)
        mouth_y = int(h * 0.55)
        mouth_w = int(w * 0.36)
        mouth_h = int(h * 0.20)

        if mouth_w <= 0 or mouth_h <= 0:
            return face

        mouth_region = face[mouth_y : mouth_y + mouth_h, mouth_x : mouth_x + mouth_w]
        if mouth_region.size == 0:
            return face

        scale = 1.0 + motion.get("mouth_open", 0.0) * 0.7
        target_h = max(8, int(mouth_h * scale))
        target_w = max(12, int(mouth_w * (0.9 + motion.get("smile", 0.0) * 0.3)))
        scaled = cv2.resize(mouth_region, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        overlay = np.zeros_like(face)
        x1 = max(0, mouth_x)
        y1 = max(0, mouth_y)
        x2 = min(w, x1 + target_w)
        y2 = min(h, y1 + target_h)
        overlay[y1:y2, x1:x2] = scaled[: y2 - y1, : x2 - x1]
        alpha = 0.55 * motion.get("mouth_open", 0.0) + 0.15
        return cv2.addWeighted(face, 1.0 - alpha, overlay, alpha, 0)

    def _apply_eye_sync(self, face: np.ndarray, motion: Dict[str, float]) -> np.ndarray:
        if face.size == 0:
            return face

        h, w = face.shape[:2]
        eye_y = int(h * 0.34)
        eye_h = max(8, int(h * 0.12))
        eye_w = max(30, int(w * 0.28))
        eye_x = int(w * 0.22)

        eye_region = face[eye_y : eye_y + eye_h, eye_x : eye_x + eye_w]
        if eye_region.size == 0:
            return face

        eye_alpha = 0.5 * (1.0 - motion.get("eye_open", 1.0))
        blurred = cv2.GaussianBlur(eye_region, (15, 15), 0)
        overlay = cv2.addWeighted(eye_region, 1.0 - eye_alpha, blurred, eye_alpha, 0)
        face = face.copy()
        face[eye_y : eye_y + eye_h, eye_x : eye_x + eye_w] = overlay
        return face

    def _apply_expression_sync(self, face: np.ndarray, motion: Dict[str, float]) -> np.ndarray:
        if face.size == 0:
            return face

        smile_strength = motion.get("smile", 0.0)
        if smile_strength <= 0.05:
            return face

        h, w = face.shape[:2]
        mouth_x = int(w * 0.35)
        mouth_y = int(h * 0.58)
        mouth_w = int(w * 0.30)
        mouth_h = int(h * 0.18)
        if mouth_w <= 0 or mouth_h <= 0:
            return face

        roi = face[mouth_y : mouth_y + mouth_h, mouth_x : mouth_x + mouth_w]
        if roi.size == 0:
            return face

        blurred = cv2.GaussianBlur(roi, (9, 9), 0)
        expression_roi = cv2.addWeighted(roi, 1.0, blurred, smile_strength * 0.25, 0)
        output = face.copy()
        output[mouth_y : mouth_y + mouth_h, mouth_x : mouth_x + mouth_w] = expression_roi
        return output

    def _apply_head_motion(self, face: np.ndarray, motion: Dict[str, float]) -> np.ndarray:
        if face.size == 0:
            return face

        h, w = face.shape[:2]
        yaw = motion.get("head_yaw", 0.0)
        pitch = motion.get("head_pitch", 0.0)
        roll = motion.get("head_roll", 0.0)

        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        scale = 1.0 + abs(yaw) * 0.18 + abs(pitch) * 0.10

        matrix = cv2.getRotationMatrix2D(center, float(roll * 25.0), scale)
        warped = cv2.warpAffine(face, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return warped

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
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return frame

        target_crop = frame[y1:y2, x1:x2]
        if target_crop.size == 0:
            return frame

        motion = self.extract_motion(bbox, np.asarray(detection["kps"], dtype=np.float32))
        face_shape = (y2 - y1, x2 - x1)
        source_face = cv2.resize(self.reference_face, face_shape[::-1], interpolation=cv2.INTER_CUBIC)

        if self.settings.get("face_replacement", True):
            output_face = source_face.copy()
        else:
            output_face = target_crop.copy()

        if self.settings.get("head_movement_sync", True):
            output_face = self._apply_head_motion(output_face, motion)

        if self.settings.get("lighting_correction", True):
            output_face = self._color_transfer(output_face, target_crop)

        if self.settings.get("lip_sync", True):
            output_face = self._apply_lip_sync(output_face, motion)

        if self.settings.get("eye_sync", True):
            output_face = self._apply_eye_sync(output_face, motion)

        if self.settings.get("expression_sync", True):
            output_face = self._apply_expression_sync(output_face, motion)

        mask = self._build_face_mask(face_shape[0], face_shape[1])
        mask_float = mask.astype(np.float32) / 255.0
        target_float = target_crop.astype(np.float32)
        source_float = output_face.astype(np.float32)
        if target_float.shape != source_float.shape:
            source_float = cv2.resize(source_float, (target_crop.shape[1], target_crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        if target_float.shape != mask_float.shape[:2] + (1,):
            mask_float = cv2.resize(mask_float, (target_crop.shape[1], target_crop.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_float = mask_float[..., None]
        else:
            mask_float = mask_float[..., None]

        blended = (target_float * (1.0 - mask_float) + source_float * mask_float).astype(np.uint8)

        processed = frame.copy()
        processed[y1:y2, x1:x2] = blended

        if self.settings.get("ai_modified_label", True):
            cv2.putText(
                processed,
                "AI-MODIFIED",
                (20, processed.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return processed
