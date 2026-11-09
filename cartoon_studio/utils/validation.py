from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2

from cartoon_studio.config import SUPPORTED_VIDEO_EXTENSIONS, MAX_VIDEO_SIZE_BYTES, MAX_VIDEO_SIZE_MB

@dataclass
class VideoValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_seconds: float = 0.0
    file_size_bytes: int = 0

def validate_video_file(file_path: str | Path) -> VideoValidationResult:
    """Validate a video file for integrity, supported codecs, size, and duration."""
    path = Path(file_path)
    if not path.exists():
        return VideoValidationResult(is_valid=False, error_message=f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
        return VideoValidationResult(
            is_valid=False, 
            error_message=f"Unsupported format '{ext}'. Supported formats: {supported}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        return VideoValidationResult(is_valid=False, error_message="Video file is empty (0 bytes).")

    if file_size > MAX_VIDEO_SIZE_BYTES:
        return VideoValidationResult(
            is_valid=False,
            error_message=f"Video exceeds maximum allowed size of {MAX_VIDEO_SIZE_MB} MB."
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoValidationResult(is_valid=False, error_message="Unable to decode video stream.")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or fps > 240:
            fps = 25.0

        if width <= 0 or height <= 0:
            return VideoValidationResult(is_valid=False, error_message="Invalid video dimensions.")

        if frame_count <= 0:
            # Attempt to count frames manually if metadata is missing
            frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frames += 1
                if frames > 100000:
                    break
            frame_count = frames

        if frame_count == 0:
            return VideoValidationResult(is_valid=False, error_message="Video contains no readable frames.")

        duration = frame_count / fps

        return VideoValidationResult(
            is_valid=True,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            file_size_bytes=file_size,
        )
    finally:
        cap.release()

