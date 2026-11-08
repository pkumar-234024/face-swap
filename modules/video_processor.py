from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from modules.video_exporter import VideoExporter
from utils.logging_config import get_logger

logger = get_logger("video_processor")


@dataclass
class VideoProcessingResult:
    """Summary of video processing execution."""

    output_path: str
    width: int
    height: int
    fps: float
    total_frames_processed: int
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None


def process_video_stream(
    input_video_path: str,
    output_video_path: str,
    frame_processor: Callable[[np.ndarray], np.ndarray],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    target_dimensions: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
) -> VideoProcessingResult:
    """Processes video frame-by-frame in a memory-efficient stream without loading whole video into RAM.

    Args:
        input_video_path: Source video path.
        output_video_path: Intermediate raw output video path.
        frame_processor: Function that accepts (BGR frame) and returns (processed BGR frame).
        progress_callback: Optional callback receiving (current_frame_index, total_frames).
        target_dimensions: Optional (out_width, out_height) to scale output frames.
        max_frames: Optional limit on frames to process (useful for testing).
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video capture for: {input_video_path}")

    in_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames)

    out_width, out_height = target_dimensions if target_dimensions else (in_width, in_height)
    # Ensure even dimensions
    out_width = out_width if out_width % 2 == 0 else out_width - 1
    out_height = out_height if out_height % 2 == 0 else out_height - 1

    writer = None
    frame_count = 0

    try:
        writer = VideoExporter.create_video_writer(
            output_path=output_video_path,
            fps=fps,
            width=out_width,
            height=out_height,
        )

        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            success, frame = capture.read()
            if not success or frame is None:
                break

            try:
                # Apply pipeline transformations
                processed = frame_processor(frame)
            except Exception as proc_exc:
                logger.warning("Error processing frame %d: %s. Using original frame.", frame_count, proc_exc)
                processed = frame

            # Resize if output dimensions differ
            if processed.shape[1] != out_width or processed.shape[0] != out_height:
                processed = cv2.resize(processed, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

            writer.write(processed)
            frame_count += 1

            if progress_callback is not None:
                progress_callback(frame_count, total_frames)

        logger.info("Successfully processed %d frames to %s", frame_count, output_video_path)

        duration = frame_count / fps if fps > 0 else 0.0
        return VideoProcessingResult(
            output_path=output_video_path,
            width=out_width,
            height=out_height,
            fps=fps,
            total_frames_processed=frame_count,
            duration_seconds=duration,
            success=True,
        )

    except Exception as exc:
        logger.error("Video processing failed: %s", exc)
        return VideoProcessingResult(
            output_path=output_video_path,
            width=out_width,
            height=out_height,
            fps=fps,
            total_frames_processed=frame_count,
            duration_seconds=0.0,
            success=False,
            error_message=str(exc),
        )

    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
