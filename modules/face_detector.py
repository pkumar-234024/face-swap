from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover
    FaceAnalysis = None


class FaceDetector:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.analysis = None
        self._load_model()

    def _load_model(self) -> None:
        if FaceAnalysis is None:
            self.analysis = None
            return

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            app = FaceAnalysis(name="buffalo_l", root=".", providers=providers)
            app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
            self.analysis = app
        except Exception:
            self.analysis = None

    def detect_face(self, image: np.ndarray, max_num: int = 1) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            return []

        if self.analysis is not None:
            try:
                faces = self.analysis.get(image, max_num=max_num)
                results = []
                for face in faces:
                    results.append({
                        "bbox": np.asarray(face.bbox, dtype=np.float32),
                        "kps": np.asarray(face.kps, dtype=np.float32),
                        "embedding": np.asarray(face.embedding, dtype=np.float32),
                        "landmark_3d_68": getattr(face, "landmark_3d_68", None),
                    })
                if results:
                    return results
            except Exception:
                pass

        return self._fallback_detect(image)

    def _fallback_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if not hasattr(cv2, "CascadeClassifier"):
            return []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade_path = getattr(cv2, "data", None)
            if cascade_path is None or not hasattr(cascade_path, "haarcascades"):
                return []

            classifier = cv2.CascadeClassifier(cascade_path.haarcascades + "haarcascade_frontalface_default.xml")
            if classifier is None:
                return []

            detections = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if detections is None:
                return []

            detections = sorted(detections, key=lambda box: box[2] * box[3], reverse=True)
            results = []
            for x, y, w, h in detections[:1]:
                landmarks = self._approximate_landmarks((x, y, w, h), image.shape[:2])
                results.append({
                    "bbox": np.array([x, y, x + w, y + h], dtype=np.float32),
                    "kps": landmarks,
                    "embedding": None,
                })
            return results
        except Exception:
            return []

    def _approximate_landmarks(self, box: tuple, image_shape: tuple) -> np.ndarray:
        x, y, w, h = box
        hmid = x + w / 2
        vmid = y + h / 2
        points = np.array([
            [hmid - w * 0.18, vmid - h * 0.25],
            [hmid + w * 0.18, vmid - h * 0.25],
            [hmid, vmid],
            [hmid, vmid + h * 0.25],
            [hmid - w * 0.18, vmid + h * 0.32],
            [hmid + w * 0.18, vmid + h * 0.32],
        ], dtype=np.float32)
        return points

    def detect_best_face(self, image: np.ndarray, max_num: int = 1) -> Optional[Dict[str, Any]]:
        detections = self.detect_face(image, max_num=max_num)
        if not detections:
            return None
        return detections[0]
