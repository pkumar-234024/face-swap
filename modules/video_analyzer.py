from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from utils.helpers import format_file_size, format_seconds
from utils.logging_config import get_logger

logger = get_logger("video_analyzer")


@dataclass
class VideoAnalysisResult:
    """Comprehensive analysis metrics for a video file."""

    path: str
    filename: str
    file_size_bytes: int
    file_size_formatted: str
    width: int
    height: int
    resolution_label: str
    fps: float
    frame_count: int
    duration_seconds: float
    duration_formatted: str
    codec: str
    has_audio: bool
    is_valid: bool
    error_message: Optional[str] = None


class VideoAnalyzer:
    """Extracts metadata and verifies technical specifications of video files."""

    @staticmethod
    def analyze(video_path: str) -> VideoAnalysisResult:
        """Inspects video headers, frame count, codecs, and resolution."""
        p = Path(video_path)
        if not p.exists() or not p.is_file():
            return VideoAnalysisResult(
                path=video_path,
                filename=p.name,
                file_size_bytes=0,
                file_size_formatted="0 B",
                width=0,
                height=0,
                resolution_label="Unknown",
                fps=0.0,
                frame_count=0,
                duration_seconds=0.0,
                duration_formatted="00:00",
                codec="Unknown",
                has_audio=False,
                is_valid=False,
                error_message="Video file not found.",
            )

        file_size = p.stat().st_size
        capture = cv2.VideoCapture(str(p))

        if not capture.isOpened():
            return VideoAnalysisResult(
                path=video_path,
                filename=p.name,
                file_size_bytes=file_size,
                file_size_formatted=format_file_size(file_size),
                width=0,
                height=0,
                resolution_label="Unknown",
                fps=0.0,
                frame_count=0,
                duration_seconds=0.0,
                duration_formatted="00:00",
                codec="Unknown",
                has_audio=False,
                is_valid=False,
                error_message="Unable to open video stream.",
            )

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

        duration = total_frames / fps if fps > 0 else 0.0

        # Determine friendly resolution label
        if width >= 3840 or height >= 2160:
            res_label = f"4K Ultra HD ({width}x{height})"
        elif width >= 1920 or height >= 1080:
            res_label = f"1080p Full HD ({width}x{height})"
        elif width >= 1280 or height >= 720:
            res_label = f"720p HD ({width}x{height})"
        else:
            res_label = f"SD ({width}x{height})"

        # Check for audio presence via probe
        from modules.audio_processor import has_audio_stream
        audio_present = has_audio_stream(str(p))

        capture.release()

        return VideoAnalysisResult(
            path=str(p.resolve()),
            filename=p.name,
            file_size_bytes=file_size,
            file_size_formatted=format_file_size(file_size),
            width=width,
            height=height,
            resolution_label=res_label,
            fps=round(fps, 2),
            frame_count=total_frames,
            duration_seconds=round(duration, 2),
            duration_formatted=format_seconds(duration),
            codec=codec or "h264/mp4v",
            has_audio=audio_present,
            is_valid=True,
        )

