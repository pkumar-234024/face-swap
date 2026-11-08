from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Robust face detection and landmark extraction using InsightFace SCRFD and Haar fallbacks."""

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
            # Look in MODELS_DIR root or current working dir
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
        """Detects faces in BGR image and extracts landmarks and attributes."""
        if image is None or image.size == 0:
            return []

        if self.analysis is not None:
            try:
                faces = self.analysis.get(image, max_num=max_num)
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

        return self._detect_fallback(image, max_num=max_num)

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

            # Sort by area descending
            sorted_boxes = sorted(detections, key=lambda b: b[2] * b[3], reverse=True)[:max_num]
            results: List[Dict[str, Any]] = []

            for x, y, w, h in sorted_boxes:
                # Estimate 5 rough keypoints (left eye, right eye, nose, left mouth, right mouth)
                kps = np.array([
                    [x + w * 0.32, y + h * 0.38],
                    [x + w * 0.68, y + h * 0.38],
                    [x + w * 0.50, y + h * 0.55],
                    [x + w * 0.35, y + h * 0.75],
                    [x + w * 0.65, y + h * 0.75],
                ], dtype=np.float32)

                results.append({
                    "bbox": np.array([x, y, x + w, y + h], dtype=np.float32),
                    "kps": kps,
                    "det_score": 0.85,
                    "landmark_3d_68": None,
                    "landmark_2d_106": None,
                    "embedding": None,
                    "pose": None,
                    "gender": None,
                    "age": None,
                })

            return results
        except Exception as exc:
            logger.error("Haar cascade fallback detection failed: %s", exc)
            return []
