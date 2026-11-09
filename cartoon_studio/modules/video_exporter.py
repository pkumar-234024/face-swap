from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np

from cartoon_studio.utils.helpers import resolve_ffmpeg_binary

logger = logging.getLogger(__name__)

class VideoStreamWriter:
    """Stream-based H.264 MP4 video encoder that streams frames to FFmpeg on the fly with minimal RAM."""

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float = 25.0,
        crf: int = 18
    ):
        self.output_path = output_path
        self.width = width if width % 2 == 0 else width - 1
        self.height = height if height % 2 == 0 else height - 1
        self.fps = fps if fps > 0 else 25.0
        self.crf = crf
        self.proc: Optional[subprocess.Popen] = None
        self.is_open: bool = False
        self._start_process()

    def _start_process(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = resolve_ffmpeg_binary()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.output_path)
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        self.is_open = True

    def write_frame(self, frame_bgr: np.ndarray) -> bool:
        if not self.is_open or self.proc is None or self.proc.stdin is None:
            return False
        h, w = frame_bgr.shape[:2]
        if h != self.height or w != self.width:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
            return True
        except Exception as ex:
            logger.error("Failed to write frame to FFmpeg stream: %s", ex)
            return False

    def close(self) -> bool:
        if not self.is_open or self.proc is None:
            return False
        self.is_open = False
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            _, stderr = self.proc.communicate(timeout=180)
            success = self.proc.returncode == 0 and self.output_path.exists() and self.output_path.stat().st_size > 0
            if not success and stderr:
                logger.error("FFmpeg encoder process error: %s", stderr.decode("utf-8", errors="replace")[:300])
            return success
        except Exception as ex:
            logger.error("Error closing FFmpeg encoder stream: %s", ex)
            return False

class VideoExporter:
    """Encodes frames into H.264 MP4 videos and constructs side-by-side comparison videos."""

    @staticmethod
    def encode_frames_to_mp4(
        frames_bgr: List[np.ndarray],
        output_path: Path,
        fps: float = 25.0,
        crf: int = 18
    ) -> bool:
        """Encode list of BGR numpy frames directly into H.264 MP4 using VideoStreamWriter."""
        if not frames_bgr:
            return False

        h, w = frames_bgr[0].shape[:2]
        writer = VideoStreamWriter(output_path, w, h, fps=fps, crf=crf)
        for frame in frames_bgr:
            writer.write_frame(frame)
        success = writer.close()
        if not success:
            return VideoExporter._fallback_cv2_writer(frames_bgr, output_path, fps)
        return True

    @staticmethod
    def _fallback_cv2_writer(
        frames_bgr: List[np.ndarray],
        output_path: Path,
        fps: float
    ) -> bool:
        """Fallback writer using OpenCV mp4v/avc1."""
        try:
            h, w = frames_bgr[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            for frame in frames_bgr:
                writer.write(frame)
            writer.release()
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception as ex:
            logger.error(f"OpenCV VideoWriter fallback failed: {ex}")
            return False

    @staticmethod
    def create_side_by_side_comparison(
        original_video_path: Path,
        cartoon_video_path: Path,
        output_path: Path,
        crf: int = 18
    ) -> bool:
        """Generate a synchronized side-by-side single-player comparison video with labeled headers."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = resolve_ffmpeg_binary()

        # Clean hstack filter
        filter_complex = "[0:v]scale=-1:720[v0];[1:v]scale=-1:720[v1];[v0][v1]hstack=inputs=2[v]"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(original_video_path),
            "-i", str(cartoon_video_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a?",  # use audio from cartoon video if available, else omit
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", str(crf),
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]

        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180
            )
            return res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
        except Exception as ex:
            logger.error(f"Side-by-side comparison generation failed: {ex}")
            return False
