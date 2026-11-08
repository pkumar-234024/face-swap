from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from modules.audio_processor import resolve_ffmpeg_path


def export_frames_to_video(frames: Iterable[np.ndarray], output_path: str, fps: float, width: int, height: int, audio_path: Optional[str] = None) -> str:
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video file: {output_path}")

    for frame in frames:
        if frame is None:
            continue
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)

    writer.release()

    if audio_path and os.path.exists(audio_path):
        final_output = str(Path(output_path).with_name(Path(output_path).stem + "_final.mp4"))
        command = [
            "-y",
            "-i",
            output_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            final_output,
        ]
        ffmpeg_path = resolve_ffmpeg_path()
        subprocess.run([ffmpeg_path, *command], check=False)
        if os.path.exists(final_output):
            os.replace(final_output, output_path)

    return output_path
