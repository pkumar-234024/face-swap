from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import cv2
import numpy as np

from config import CRF_PRESETS, RESOLUTION_PRESETS
from modules.audio_processor import merge_audio, resolve_ffmpeg_path, run_ffmpeg
from utils.logging_config import get_logger

logger = get_logger("video_exporter")


class VideoExporter:
    """Exports processed video frames to H.264 MP4 with multi-resolution and audio muxing."""

    @staticmethod
    def resolve_target_dimensions(
        original_width: int,
        original_height: int,
        preset_name: str = "Original",
    ) -> Tuple[int, int]:
        """Calculates scaled width and height adhering to aspect ratio and even pixel constraints."""
        target_preset = RESOLUTION_PRESETS.get(preset_name)
        if target_preset is None:
            # Original resolution: ensure dimensions are even numbers (required by H.264)
            w = original_width if original_width % 2 == 0 else original_width - 1
            h = original_height if original_height % 2 == 0 else original_height - 1
            return max(2, w), max(2, h)

        target_w, target_h = target_preset

        # Preserve aspect ratio while fitting within target bounding box
        aspect_ratio = original_width / float(max(1, original_height))
        target_ratio = target_w / float(target_h)

        if aspect_ratio > target_ratio:
            out_w = target_w
            out_h = int(round(target_w / aspect_ratio))
        else:
            out_h = target_h
            out_w = int(round(target_h * aspect_ratio))

        # Make dimensions even
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1
        return max(2, out_w), max(2, out_h)

    @staticmethod
    def create_video_writer(
        output_path: str,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        """Initializes OpenCV VideoWriter with fallback codecs."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        for fourcc_code in ("mp4v", "MJPG", "XVID"):
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
                if writer.isOpened():
                    logger.debug("Initialized VideoWriter with codec %s", fourcc_code)
                    return writer
                writer.release()
            except Exception:
                pass

        raise RuntimeError(f"Failed to initialize VideoWriter for {output_path}.")

    @classmethod
    def export_video_with_audio(
        cls,
        raw_video_path: str,
        audio_path: Optional[str],
        final_output_path: str,
        quality_preset: str = "High",
    ) -> str:
        """Encodes raw video to H.264 and muxes audio track."""
        crf = CRF_PRESETS.get(quality_preset, 18)
        return merge_audio(
            video_path=raw_video_path,
            audio_path=audio_path,
            final_output_path=final_output_path,
            crf=crf,
        )
