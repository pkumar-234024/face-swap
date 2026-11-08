from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from config import MODELS_DIR
from modules.face_aligner import align_face_to_canonical, estimate_similarity_transform
from modules.face_blending import FaceBlender
from modules.face_detector import FaceDetector
from modules.face_tracker import FaceTracker
from modules.facial_animation import FacialAnimationPipeline
from modules.facial_landmarks import FacialLandmarkTracker, FacialLandmarks
from modules.lighting_correction import LightingCorrector
from utils.helpers import get_device
from utils.logging_config import get_logger

logger = get_logger("face_replacement")

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class FaceReplacementPipeline:
    """End-to-end facial replacement engine combining identity conditioning and blending."""

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        settings: Optional[Dict[str, bool | str]] = None,
    ):
        self.settings = settings or {}
        self.device = get_device()
        self.detector = detector or FaceDetector(device=self.device)
        self.landmark_tracker = FacialLandmarkTracker(
            smoothing=self.settings.get("temporal_smoothing", True),
        )
        self.face_tracker = FaceTracker(
            smoothing_alpha=0.75,
        )
        self.blender = FaceBlender(
            temporal_smoothing=self.settings.get("temporal_smoothing", True),
        )
        self.animation_pipeline = FacialAnimationPipeline(settings=self.settings)

        self.ref_image: Optional[np.ndarray] = None
        self.ref_face_data: Optional[Dict[str, Any]] = None
        self.ref_landmarks: Optional[FacialLandmarks] = None
        self.ref_embedding: Optional[np.ndarray] = None

        # Optional InSwapper ONNX session
        self.inswapper_session = None
        self._init_inswapper_model()

    def _init_inswapper_model(self) -> None:
        """Loads optional inswapper_128.onnx model if available on disk."""
        if ort is None:
            return

        model_path = MODELS_DIR / "inswapper_128.onnx"
        if not model_path.exists():
            return

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self.inswapper_session = ort.InferenceSession(str(model_path), providers=providers)
            logger.info("Deep InSwapper model loaded successfully from %s.", model_path)
        except Exception as exc:
            logger.warning("Could not load inswapper_128.onnx: %s. Using facial animation engine.", exc)
            self.inswapper_session = None

    def load_reference(self, reference_image: np.ndarray) -> None:
        """Analyzes and stores authorized reference identity."""
        if reference_image is None or reference_image.size == 0:
            raise ValueError("Reference image is empty.")

        best_face = self.detector.detect_best_face(reference_image)
        if best_face is None:
            raise ValueError("No authorized face detected in the reference image.")

        landmarks = self.landmark_tracker.extract_landmarks(best_face, reference_image.shape[:2])
        if landmarks is None or len(landmarks.points_68) < 68:
            raise ValueError("Unable to extract 68 facial landmarks from reference image.")

        self.ref_image = reference_image.copy()
        self.ref_face_data = best_face
        self.ref_landmarks = landmarks
        self.ref_embedding = best_face.get("embedding")

        # Load into facial animation engine
        bbox = best_face["bbox"]
        self.animation_pipeline.load_reference(reference_image, landmarks.points_68, bbox)
        logger.info("Reference identity validated and stored.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a single video frame: tracks face, animates reference likeness, and blends seamlessly."""
        if frame is None or frame.size == 0:
            return frame

        if self.ref_image is None or self.ref_landmarks is None:
            return frame

        h, w = frame.shape[:2]

        # 1. Detect face in frame
        detected_face = self.detector.detect_best_face(frame)
        raw_bbox = detected_face["bbox"] if detected_face else None

        # 2. Update tracker for smooth bounding box
        tracked_bbox = self.face_tracker.update(frame, raw_bbox)
        if tracked_bbox is None:
            return self._apply_watermark(frame)

        # 3. Extract landmarks
        landmarks = None
        if detected_face is not None:
            landmarks = self.landmark_tracker.extract_landmarks(detected_face, (h, w))

        if landmarks is None:
            return self._apply_watermark(frame)

        # 4. Generate replacement face
        if self.inswapper_session is not None and self.ref_embedding is not None and len(detected_face.get("kps", [])) >= 5:
            # Deep InSwapper ONNX execution
            swapped_frame = self._run_inswapper(frame, detected_face)
            processed_frame = swapped_frame if swapped_frame is not None else frame
        else:
            # Identity-Conditioned Facial Animation & Direct Delaunay Mesh Warping
            warped_roi, roi_mask, roi_bbox = self.animation_pipeline.warp_face_to_target(
                source_landmarks=landmarks.points_68,
                target_frame_shape=(h, w),
                source_bbox=tracked_bbox,
                detector_pose=detected_face.get("pose") if detected_face else None,
            )

            # Blend into target frame
            blend_method = "poisson" if self.settings.get("seamless_blending", True) else "feathered"
            processed_frame = self.blender.blend_face(
                target_frame=frame,
                warped_roi=warped_roi,
                roi_mask=roi_mask,
                roi_bbox=roi_bbox,
                method=blend_method,
            )

        return self._apply_watermark(processed_frame)

    def _run_inswapper(self, frame: np.ndarray, target_face: Dict[str, Any]) -> Optional[np.ndarray]:
        """Runs the official InSwapper ONNX model if configured."""
        try:
            # Canonical 128x128 face alignment
            aimg, affine_mat, inv_mat = align_face_to_canonical(frame, target_face["kps"], (128, 128))
            blob = aimg.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)  # HWC to CHW
            blob = np.expand_dims(blob, axis=0)  # NCHW

            input_feed = {
                self.inswapper_session.get_inputs()[0].name: blob,
                self.inswapper_session.get_inputs()[1].name: self.ref_embedding.reshape(1, -1).astype(np.float32),
            }
            outputs = self.inswapper_session.run(None, input_feed)
            out_img = outputs[0][0].transpose(1, 2, 0) * 255.0
            out_img = np.clip(out_img, 0, 255).astype(np.uint8)

            # Warp back onto original frame using inverse affine matrix
            h, w = frame.shape[:2]
            inv_warped = cv2.warpAffine(out_img, inv_mat, (w, h), flags=cv2.INTER_LINEAR)
            mask = np.ones((128, 128), dtype=np.float32)
            mask = cv2.warpAffine(mask, inv_mat, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.GaussianBlur(mask, (19, 19), 0)[..., None]

            result = (frame.astype(np.float32) * (1.0 - mask) + inv_warped.astype(np.float32) * mask).astype(np.uint8)
            return result
        except Exception as exc:
            logger.warning("InSwapper inference error: %s", exc)
            return None

    def _apply_watermark(self, frame: np.ndarray) -> np.ndarray:
        """Applies visible ethical AI content disclosure badge."""
        if not self.settings.get("ai_modified_label", True):
            return frame

        out = frame.copy()
        h, w = out.shape[:2]

        label = "AI-MODIFIED | CONSENTED LIKENESS"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.45, min(0.75, w / 1920.0 * 0.75))
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        margin = 16
        box_x1 = margin
        box_y1 = h - margin - text_h - 12
        box_x2 = margin + text_w + 16
        box_y2 = h - margin

        # Translucent pill background
        sub_img = out[box_y1:box_y2, box_x1:box_x2]
        rect = np.zeros_like(sub_img, dtype=np.uint8)
        rect[:] = (15, 23, 42)  # Dark slate
        cv2.addWeighted(sub_img, 0.25, rect, 0.75, 0, sub_img)
        out[box_y1:box_y2, box_x1:box_x2] = sub_img

        # Text with subtle shadow
        cv2.putText(
            out,
            label,
            (box_x1 + 8, box_y2 - 6),
            font,
            font_scale,
            (34, 211, 238),  # Cyan
            thickness,
            cv2.LINE_AA,
        )
        return out

    def reset(self) -> None:
        self.face_tracker.reset()
        self.landmark_tracker.reset()
        self.blender.reset()
        self.animation_pipeline.reset()
