from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import BUFFALO_L_DIR, MODELS_DIR
from utils.helpers import get_device
from utils.logging_config import get_logger

logger = get_logger("face_detector")

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    insightface = None
    FaceAnalysis = None


class FaceDetector:
    """Robust face detection and landmark extraction with dynamic multi-scale pyramid and orientation fallbacks."""

    def __init__(self, device: Optional[str] = None, det_size: tuple[int, int] = (640, 640)):
        self.device = device or get_device()
        self.det_size = det_size
        self.analysis: Optional[FaceAnalysis] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initializes the InsightFace Buffalo_L analysis engine."""
        if FaceAnalysis is None:
            logger.warning("InsightFace is not installed. Falling back to OpenCV Cascade.")
            return

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            model_root = str(MODELS_DIR.parent)
            logger.info("Initializing InsightFace with providers=%s, root=%s", providers, model_root)

            app = FaceAnalysis(name="buffalo_l", root=model_root, providers=providers)
            ctx_id = 0 if self.device == "cuda" else -1
            app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            self.analysis = app
            logger.info("InsightFace Buffalo_L pipeline ready on %s.", self.device)
        except Exception as exc:
            logger.warning("InsightFace initialization failed: %s. Switching to fallback.", exc)
            self.analysis = None

    def detect_faces(self, image: np.ndarray, max_num: int = 5) -> List[Dict[str, Any]]:
        """Detects faces in BGR image using dynamic multi-scale pyramid and orientation fallbacks."""
        if image is None or image.size == 0:
            return []

        # Ensure image is 3-channel BGR
        proc_img = image
        if len(proc_img.shape) == 3 and proc_img.shape[2] == 4:
            proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGRA2BGR)
        elif len(proc_img.shape) == 2 or (len(proc_img.shape) == 3 and proc_img.shape[2] == 1):
            proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)

        if self.analysis is not None:
            try:
                # 1. Primary detection pass with configured det_size
                faces = self.analysis.get(proc_img, max_num=max_num)

                # 2. Multi-scale dynamic pyramid fallback (critical for tight close-up portraits and high-res photos)
                if not faces:
                    det_model = self.analysis.models.get("detection")
                    if det_model is not None:
                        ctx_id = 0 if self.device == "cuda" else -1
                        for alt_size in ((512, 512), (320, 320), (480, 480)):
                            det_model.prepare(ctx_id=ctx_id, input_size=alt_size)
                            faces = self.analysis.get(proc_img, max_num=max_num)
                            if faces:
                                logger.info("Detected %d face(s) using dynamic pyramid scale %s.", len(faces), alt_size)
                                break
                        # Restore default detection size
                        det_model.prepare(ctx_id=ctx_id, input_size=self.det_size)

                # 3. Multi-angle fallback for images with uncorrected rotation (90, 180, 270 deg)
                if not faces:
                    faces = self._detect_rotated_fallback(proc_img, max_num=max_num)

                if faces:
                    results: List[Dict[str, Any]] = []
                    for face in faces:
                        entry = {
                            "bbox": np.asarray(face.bbox, dtype=np.float32),
                            "kps": np.asarray(face.kps, dtype=np.float32),
                            "det_score": float(getattr(face, "det_score", 1.0)),
                            "landmark_3d_68": (
                                np.asarray(face.landmark_3d_68, dtype=np.float32)
                                if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None
                                else None
                            ),
                            "landmark_2d_106": (
                                np.asarray(face.landmark_2d_106, dtype=np.float32)
                                if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None
                                else None
                            ),
                            "embedding": (
                                np.asarray(face.embedding, dtype=np.float32)
                                if hasattr(face, "embedding") and face.embedding is not None
                                else None
                            ),
                            "pose": (
                                np.asarray(face.pose, dtype=np.float32)
                                if hasattr(face, "pose") and face.pose is not None
                                else None
                            ),
                            "gender": getattr(face, "gender", None),
                            "age": getattr(face, "age", None),
                        }
                        results.append(entry)

                    # Sort faces by bounding box area descending
                    results.sort(
                        key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
                        reverse=True,
                    )
                    return results
            except Exception as exc:
                logger.debug("InsightFace inference warning: %s", exc)

        return self._detect_fallback(proc_img, max_num=max_num)

    def _detect_rotated_fallback(self, image: np.ndarray, max_num: int = 5) -> list:
        """Tries detection on rotated copies and maps coordinates back to original image space."""
        det_model = self.analysis.models.get("detection") if self.analysis else None
        ctx_id = 0 if self.device == "cuda" else -1
        h_orig, w_orig = image.shape[:2]

        rot_configs = [
            (cv2.ROTATE_90_CLOCKWISE, 90),
            (cv2.ROTATE_180, 180),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, 270),
        ]

        found_faces = []
        for rot_code, rot_angle in rot_configs:
            rot_img = cv2.rotate(image, rot_code)
            for scale in ((512, 512), (320, 320), self.det_size):
                if det_model is not None:
                    det_model.prepare(ctx_id=ctx_id, input_size=scale)
                rot_faces = self.analysis.get(rot_img, max_num=max_num)
                if rot_faces:
                    logger.info("Detected %d face(s) via %d deg rotation fallback at %s.", len(rot_faces), rot_angle, scale)
                    for rf in rot_faces:
                        self._unrotate_single_face(rf, (h_orig, w_orig), rot_angle)
                        found_faces.append(rf)
                    break
            if found_faces:
                break

        if det_model is not None:
            det_model.prepare(ctx_id=ctx_id, input_size=self.det_size)

        return found_faces

    @staticmethod
    def _unrotate_single_face(face_obj: Any, orig_shape: Tuple[int, int], rot_angle: int) -> None:
        """Transforms face bounding box, keypoints, and landmarks back to original image coordinates."""
        h_orig, w_orig = orig_shape[:2]

        def unrotate_2d(pts: np.ndarray) -> np.ndarray:
            p = pts.copy()
            if rot_angle == 90:
                p[:, 0] = pts[:, 1]
                p[:, 1] = h_orig - 1 - pts[:, 0]
            elif rot_angle == 180:
                p[:, 0] = w_orig - 1 - pts[:, 0]
                p[:, 1] = h_orig - 1 - pts[:, 1]
            elif rot_angle == 270:
                p[:, 0] = w_orig - 1 - pts[:, 1]
                p[:, 1] = pts[:, 0]
            return p

        # 1. Unrotate BBox (4 corners to stay axis-aligned)
        bx = np.asarray(face_obj.bbox, dtype=np.float32)
        corners = np.array([
            [bx[0], bx[1]],
            [bx[2], bx[1]],
            [bx[0], bx[3]],
            [bx[2], bx[3]],
        ], dtype=np.float32)
        unrot_corners = unrotate_2d(corners)
        face_obj.bbox = np.array([
            np.min(unrot_corners[:, 0]),
            np.min(unrot_corners[:, 1]),
            np.max(unrot_corners[:, 0]),
            np.max(unrot_corners[:, 1]),
        ], dtype=np.float32)

        # 2. Unrotate Keypoints
        if hasattr(face_obj, "kps") and face_obj.kps is not None:
            face_obj.kps = unrotate_2d(np.asarray(face_obj.kps, dtype=np.float32))

        # 3. Unrotate 3D 68 Landmarks
        if hasattr(face_obj, "landmark_3d_68") and face_obj.landmark_3d_68 is not None:
            lmk3d = np.asarray(face_obj.landmark_3d_68, dtype=np.float32).copy()
            lmk2d_unrot = unrotate_2d(lmk3d[:, :2])
            lmk3d[:, :2] = lmk2d_unrot
            face_obj.landmark_3d_68 = lmk3d

        # 4. Unrotate 2D 106 Landmarks
        if hasattr(face_obj, "landmark_2d_106") and face_obj.landmark_2d_106 is not None:
            face_obj.landmark_2d_106 = unrotate_2d(np.asarray(face_obj.landmark_2d_106, dtype=np.float32))

    def detect_best_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Finds the most prominent centered face in the frame."""
        faces = self.detect_faces(image, max_num=5)
        if not faces:
            return None

        h, w = image.shape[:2]
        center_x, center_y = w / 2.0, h / 2.0

        def score_face(face: Dict[str, Any]) -> float:
            bbox = face["bbox"]
            box_w = max(1.0, bbox[2] - bbox[0])
            box_h = max(1.0, bbox[3] - bbox[1])
            area = box_w * box_h
            bx = (bbox[0] + bbox[2]) / 2.0
            by = (bbox[1] + bbox[3]) / 2.0
            dist_sq = (bx - center_x) ** 2 + (by - center_y) ** 2
            max_dist_sq = (w ** 2 + h ** 2) / 4.0
            dist_penalty = 1.0 - min(1.0, dist_sq / max(1.0, max_dist_sq))
            return float(area * (dist_penalty ** 0.5))

        faces.sort(key=score_face, reverse=True)
        return faces[0]

    def _detect_fallback(self, image: np.ndarray, max_num: int = 1) -> List[Dict[str, Any]]:
        """OpenCV Haar Cascade fallback when InsightFace models are offline."""
        if not hasattr(cv2, "CascadeClassifier"):
            return []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade_dir = getattr(cv2, "data", None)
            if cascade_dir is None or not hasattr(cascade_dir, "haarcascades"):
                return []

            classifier = cv2.CascadeClassifier(cascade_dir.haarcascades + "haarcascade_frontalface_default.xml")
            detections = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if detections is None or len(detections) == 0:
                return []

            results = []
            for x, y, w, h in detections[:max_num]:
                results.append({
                    "bbox": np.array([x, y, x + w, y + h], dtype=np.float32),
                    "kps": np.array([
                        [x + w * 0.3, y + h * 0.35],
                        [x + w * 0.7, y + h * 0.35],
                        [x + w * 0.5, y + h * 0.55],
                        [x + w * 0.35, y + h * 0.75],
                        [x + w * 0.65, y + h * 0.75],
                    ], dtype=np.float32),
                    "det_score": 0.80,
                    "landmark_3d_68": None,
                    "landmark_2d_106": None,
                    "embedding": None,
                    "pose": np.zeros(3, dtype=np.float32),
                    "gender": None,
                    "age": None,
                })
            return results
        except Exception:
            return []
