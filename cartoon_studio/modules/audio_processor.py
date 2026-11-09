from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from cartoon_studio.utils.helpers import resolve_ffmpeg_binary

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio extraction, format conversion, and synchronized video remuxing via FFmpeg."""

    @staticmethod
    def extract_audio(video_path: Path, output_audio_path: Path) -> bool:
        """Extract lossless audio track from original video."""
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = resolve_ffmpeg_binary()
        # Attempt stream copy first
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-c:a", "copy",
            str(output_audio_path)
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60
            )
            if res.returncode == 0 and output_audio_path.exists() and output_audio_path.stat().st_size > 0:
                return True
        except Exception as ex:
            logger.warning(f"Direct stream copy failed: {ex}. Attempting AAC re-encode.")

        # Fallback to high-quality AAC re-encode
        cmd_fallback = [
            ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_audio_path)
        ]
        try:
            res = subprocess.run(
                cmd_fallback,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60
            )
            return res.returncode == 0 and output_audio_path.exists() and output_audio_path.stat().st_size > 0
        except Exception as ex:
            logger.error(f"Audio extraction failed completely: {ex}")
            return False

    @staticmethod
    def remux_audio(
        video_only_path: Path,
        audio_path: Path,
        output_path: Path,
        crf: int = 18
    ) -> bool:
        """Merge audio track into video file ensuring lip sync and temporal alignment."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = resolve_ffmpeg_binary()

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(video_only_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
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
            if res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                return True
            logger.error(f"FFmpeg audio remux failed with code {res.returncode}: {res.stderr[:400]}")
            return False
        except Exception as ex:
            logger.error(f"FFmpeg audio remux exception: {ex}")
            return False
