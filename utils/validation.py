from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from config import (
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from utils.helpers import format_file_size
from utils.logging_config import get_logger

logger = get_logger("validation")

# Blur detection threshold on Laplacian variance
BLUR_THRESHOLD = 50.0
# Minimum recommended image resolution
MIN_IMAGE_DIMENSION = 128
MIN_FACE_BOX_SIZE = 64


def allowed_extension(filename: str, allowed: Iterable[str]) -> bool:
    """Checks whether filename matches allowed extensions (case-insensitive)."""
    if not filename:
        return False
    return Path(filename).suffix.lower() in {ext.lower() for ext in allowed}


def validate_file_size(file_size_bytes: int, max_bytes: int) -> bool:
    """Checks if file size is non-zero and within permitted ceiling."""
    return 0 < file_size_bytes <= max_bytes


def validate_video_file(uploaded_file) -> Tuple[bool, str]:
    """Validates an uploaded video stream before saving to disk."""
    if uploaded_file is None:
        return False, "Please upload a source video file."

    filename = getattr(uploaded_file, "name", "") or ""
    if not allowed_extension(filename, SUPPORTED_VIDEO_EXTENSIONS):
        ext_list = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        return False, f"Unsupported video format '{Path(filename).suffix}'. Supported formats: {ext_list}"

    file_size = getattr(uploaded_file, "size", 0) or 0
    if not validate_file_size(file_size, MAX_VIDEO_SIZE_BYTES):
        return (
            False,
            f"Video file exceeds maximum allowed size of {format_file_size(MAX_VIDEO_SIZE_BYTES)} "
            f"(Uploaded: {format_file_size(file_size)}).",
        )

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        header = uploaded_file.read(64)
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        if not header:
            return False, "Uploaded video file is completely empty."
    except Exception as exc:
        return False, f"Unable to read uploaded video stream: {exc}"

    return True, "Valid video file."


def validate_video_file_path(video_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Inspects a video file on disk for playability, metadata, and codecs."""
    info: Dict[str, Any] = {
        "valid": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration": 0.0,
        "codec": "Unknown",
        "file_size": 0,
    }

    if not video_path or not os.path.exists(video_path):
        return False, "Source video file not found on disk.", info

    info["file_size"] = os.path.getsize(video_path)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return (
            False,
            "Video file could not be opened. It may be corrupt or encoded in an unsupported codec.",
            info,
        )

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

    info["width"] = width
    info["height"] = height
    info["fps"] = fps if fps > 0 else 24.0
    info["frame_count"] = frame_count
    info["duration"] = frame_count / info["fps"] if info["fps"] > 0 else 0.0
    info["codec"] = codec or "avc1/mp4v"

    # Decode first frame to guarantee decode capability
    success, frame = capture.read()
    capture.release()

    if not success or frame is None or frame.size == 0:
        return False, "Video decoding test failed. The file appears to be damaged or truncated.", info

    if width < 64 or height < 64:
        return False, f"Video resolution ({width}x{height}) is too small for facial analysis.", info

    info["valid"] = True
    return True, "Video successfully verified.", info


def compute_image_sharpness(image_bgr: np.ndarray) -> float:
    """Computes the Laplacian variance to estimate image sharpness."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def validate_reference_image(uploaded_file) -> Tuple[bool, str]:
    """Validates uploaded reference image stream before saving."""
    if uploaded_file is None:
        return False, "Please upload a reference face image."

    filename = getattr(uploaded_file, "name", "") or ""
    if not allowed_extension(filename, SUPPORTED_IMAGE_EXTENSIONS):
        ext_list = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        return False, f"Unsupported image format '{Path(filename).suffix}'. Supported formats: {ext_list}"

    file_size = getattr(uploaded_file, "size", 0) or 0
    if not validate_file_size(file_size, MAX_IMAGE_SIZE_BYTES):
        return (
            False,
            f"Image exceeds maximum allowed size of {format_file_size(MAX_IMAGE_SIZE_BYTES)} "
            f"(Uploaded: {format_file_size(file_size)}).",
        )

    return True, "Valid reference image format."


def validate_reference_image_path(image_path: str, face_detector=None) -> Tuple[bool, str, Dict[str, Any]]:
    """Performs deep validation on reference face image including sharpness, resolution, and face presence."""
    info: Dict[str, Any] = {
        "valid": False,
        "width": 0,
        "height": 0,
        "sharpness": 0.0,
        "faces_detected": 0,
        "warnings": [],
        "face_bbox": None,
    }

    if not image_path or not os.path.exists(image_path):
        return False, "Reference image file not found on disk.", info

    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        return False, "Image could not be decoded. The file may be corrupt.", info

    h, w = image.shape[:2]
    info["width"] = w
    info["height"] = h

    if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
        return (
            False,
            f"Image resolution ({w}x{h}) is too low. Minimum recommended is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}.",
            info,
        )

    sharpness = compute_image_sharpness(image)
    info["sharpness"] = sharpness
    if sharpness < BLUR_THRESHOLD:
        info["warnings"].append(
            f"Image appears blurry (sharpness score {sharpness:.1f} < {BLUR_THRESHOLD:.1f}). "
            "A sharper, well-lit portrait produces higher quality facial animation."
        )

    # Optional face detection check if detector is passed
    if face_detector is not None:
        try:
            detected_faces = face_detector.detect_faces(image, max_num=5)
            info["faces_detected"] = len(detected_faces)
            if len(detected_faces) == 0:
                return (
                    False,
                    "No human face was detected in the reference image. Please upload a clear, front-facing portrait.",
                    info,
                )
            if len(detected_faces) > 1:
                info["warnings"].append(
                    f"Multiple faces ({len(detected_faces)}) detected. "
                    "The primary centered face will be used as the authorized reference identity."
                )

            primary_face = detected_faces[0]
            bbox = primary_face.get("bbox")
            if bbox is not None:
                info["face_bbox"] = [float(v) for v in bbox]
                fw = bbox[2] - bbox[0]
                fh = bbox[3] - bbox[1]
                if fw < MIN_FACE_BOX_SIZE or fh < MIN_FACE_BOX_SIZE:
                    info["warnings"].append(
                        f"Detected face crop is small ({int(fw)}x{int(fh)}). "
                        "A closer portrait will significantly improve expression transfer fidelity."
                    )
        except Exception as exc:
            logger.warning("Face detection during validation threw error: %s", exc)

    info["valid"] = True
    message = "Reference image verified." if not info["warnings"] else "Reference image verified with warnings."
    return True, message, info
