from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


def load_video_metadata(video_path: str) -> VideoMetadata:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    capture.release()
    return VideoMetadata(path=video_path, width=width, height=height, fps=fps, frame_count=frame_count, duration=duration)


def process_video(video_path: str, pipeline, output_path: str, progress_callback: Optional[Callable[[float], None]] = None, target_size: Optional[tuple[int, int]] = None) -> VideoMetadata:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to process video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_width, out_height = width, height
    if target_size is not None:
        out_width, out_height = target_size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Unable to initialize video writer for processed output.")

    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        processed_frame = pipeline.process_frame(frame)
        if target_size is not None and processed_frame.shape[:2] != (out_height, out_width):
            processed_frame = cv2.resize(processed_frame, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
        writer.write(processed_frame)
        frame_index += 1
        if progress_callback is not None and total_frames > 0:
            progress_callback(min(frame_index / total_frames, 1.0))

    writer.release()
    capture.release()
    return VideoMetadata(path=output_path, width=out_width, height=out_height, fps=fps, frame_count=frame_index, duration=frame_index / fps if fps > 0 else 0.0)
