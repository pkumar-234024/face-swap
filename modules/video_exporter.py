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

    @classmethod
    def create_side_by_side_comparison(
        cls,
        original_video_path: str,
        modified_video_path: str,
        output_comparison_path: str,
    ) -> Optional[str]:
        """Generates a synchronous dual-screen comparison video in ONE unified frame.

        Left side displays the Original Driving Actor.
        Right side displays the AI-Modified Likeness.
        Both streams play in exact synchronous lockstep with audio in a single player.
        """
        if not os.path.exists(original_video_path) or not os.path.exists(modified_video_path):
            return None

        # Inspect dimensions
        cap = cv2.VideoCapture(original_video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)
        cap.release()

        # Calculate scaled dimensions so combined video fits comfortably in one frame
        is_portrait = h > w
        if is_portrait:
            target_w, target_h = 540, 960
        else:
            target_w, target_h = 640, 360

        filter_cmd = (
            f"[0:v]scale={target_w}:{target_h},drawtext=text='ORIGINAL ACTOR':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6[v0];"
            f"[1:v]scale={target_w}:{target_h},drawtext=text='AI-MODIFIED LIKENESS':x=20:y=20:fontsize=24:fontcolor=cyan:box=1:boxcolor=black@0.6[v1];"
            "[v0][v1]hstack=inputs=2[v]"
        )

        Path(output_comparison_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "-y",
            "-i", original_video_path,
            "-i", modified_video_path,
            "-filter_complex", filter_cmd,
            "-map", "[v]",
            "-map", "1:a?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-shortest",
            output_comparison_path,
        ]

        res = run_ffmpeg(cmd)
        if res.returncode == 0 and os.path.exists(output_comparison_path) and os.path.getsize(output_comparison_path) > 100:
            logger.info("Generated synchronous side-by-side comparison video at %s", output_comparison_path)
            return output_comparison_path

        # Fallback if drawtext filter is not present
        simple_filter = f"[0:v]scale={target_w}:{target_h}[v0];[1:v]scale={target_w}:{target_h}[v1];[v0][v1]hstack=inputs=2[v]"
        fallback_cmd = [
            "-y",
            "-i", original_video_path,
            "-i", modified_video_path,
            "-filter_complex", simple_filter,
            "-map", "[v]",
            "-map", "1:a?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-shortest",
            output_comparison_path,
        ]
        fallback_res = run_ffmpeg(fallback_cmd)
        if fallback_res.returncode == 0 and os.path.exists(output_comparison_path):
            return output_comparison_path

        return None
