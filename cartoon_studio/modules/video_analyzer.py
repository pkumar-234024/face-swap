from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

@dataclass
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    file_size_bytes: int
    has_audio: bool = False
    scene_cuts: List[int] = field(default_factory=list)

class VideoAnalyzer:
    """Analyzes video metadata, verifies integrity, and detects scene cut boundaries."""

    @staticmethod
    def analyze(video_path: str | Path) -> VideoMetadata:
        """Extract video properties and inspect audio stream presence."""
        path = Path(video_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        file_size = path.stat().st_size
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0 or fps > 240:
                fps = 25.0

            if frame_count <= 0:
                # Count frames by reading if container lacks frame count
                count = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    count += 1
                frame_count = count

            duration = frame_count / fps if fps > 0 else 0.0

            # Check for audio presence using ffprobe
            has_audio = VideoAnalyzer._check_audio_presence(path)

            return VideoMetadata(
                path=path,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration_seconds=duration,
                file_size_bytes=file_size,
                has_audio=has_audio,
            )
        finally:
            cap.release()

    @staticmethod
    def detect_scene_cuts(
        video_path: str | Path,
        threshold: float = 0.55,
        max_frames_to_sample: int = 1000
    ) -> List[int]:
        """Detect scene cut frame indices using HSV histogram correlation difference."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        scene_cuts: List[int] = [0]
        prev_hist: Optional[np.ndarray] = None
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames_to_sample and frame_idx >= max_frames_to_sample):
                    break

                # Downsample frame for fast histogram calculation
                small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                if prev_hist is not None:
                    # Compare histogram correlation (1.0 = identical, < 0.45 = scene change)
                    corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if corr < threshold:
                        scene_cuts.append(frame_idx)

                prev_hist = hist
                frame_idx += 1

            return scene_cuts
        finally:
            cap.release()

    @staticmethod
    def _check_audio_presence(video_path: Path) -> bool:
        """Check if video file has an audio stream using ffmpeg inspection."""
        import subprocess
        from cartoon_studio.utils.helpers import resolve_ffmpeg_binary
        try:
            ffmpeg_bin = resolve_ffmpeg_binary()
            cmd = [ffmpeg_bin, "-i", str(video_path)]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5
            )
            # Inspect stderr for Audio stream signature
            return "Audio:" in result.stderr
        except Exception:
            return False
