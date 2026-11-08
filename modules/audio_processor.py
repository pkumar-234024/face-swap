from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger

logger = get_logger("audio_processor")

try:
    from imageio_ffmpeg import get_ffmpeg_exe
except ImportError:
    get_ffmpeg_exe = None


def resolve_ffmpeg_path() -> str:
    """Finds available FFmpeg binary from PATH, env var, or bundled imageio-ffmpeg."""
    custom_path = os.environ.get("FFMPEG_PATH")
    if custom_path and os.path.exists(custom_path):
        return custom_path

    for binary_name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(binary_name)
        if found and os.path.exists(found):
            return found

    if get_ffmpeg_exe is not None:
        try:
            bundled = get_ffmpeg_exe()
            if bundled and os.path.exists(bundled):
                return bundled
        except Exception:
            pass

    raise FileNotFoundError(
        "FFmpeg executable not found. Please install FFmpeg and add it to PATH, "
        "or ensure imageio-ffmpeg is installed."
    )


def run_ffmpeg(command_args: list[str]) -> subprocess.CompletedProcess[str]:
    """Executes FFmpeg command with stderr capture."""
    ffmpeg_exe = resolve_ffmpeg_path()
    full_cmd = [ffmpeg_exe, *command_args]
    logger.debug("Executing FFmpeg: %s", " ".join(full_cmd))
    return subprocess.run(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def has_audio_stream(video_path: str) -> bool:
    """Checks whether video file contains an audio stream."""
    if not os.path.exists(video_path):
        return False

    try:
        cmd = ["-i", video_path]
        res = run_ffmpeg(cmd)
        # Check stderr for "Audio:"
        return "Audio:" in res.stderr
    except Exception as exc:
        logger.debug("Error probing audio stream: %s", exc)
        return False


def extract_audio(video_path: str, output_audio_path: str) -> Optional[str]:
    """Extracts the audio track from video preserving original encoding."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not has_audio_stream(video_path):
        logger.info("Video %s contains no audio stream. Skipping extraction.", video_path)
        return None

    os.makedirs(os.path.dirname(output_audio_path) or ".", exist_ok=True)

    # First attempt: stream copy (lossless, zero re-encoding)
    command = [
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "copy",
        output_audio_path,
    ]
    result = run_ffmpeg(command)
    if result.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 100:
        logger.info("Extracted audio via stream-copy to %s", output_audio_path)
        return output_audio_path

    # Fallback attempt: re-encode to clean AAC stream
    fallback_command = [
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "aac",
        "-b:a",
        "192k",
        output_audio_path,
    ]
    fallback_res = run_ffmpeg(fallback_command)
    if fallback_res.returncode != 0:
        logger.warning("Audio extraction failed: %s", fallback_res.stderr.strip()[:200])
        return None

    logger.info("Extracted audio via AAC fallback to %s", output_audio_path)
    return output_audio_path


def merge_audio(
    video_path: str,
    audio_path: Optional[str],
    final_output_path: str,
    crf: int = 18,
) -> str:
    """Muxes audio and video streams together with precise sync and H.264 encoding."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not audio_path or not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
        # No audio to merge, simply re-encode video to ensure browser compatibility
        logger.info("No audio track present. Finalizing video-only output.")
        cmd = [
            "-y",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            "fast",
            final_output_path,
        ]
        res = run_ffmpeg(cmd)
        if res.returncode == 0:
            return final_output_path
        shutil.copy2(video_path, final_output_path)
        return final_output_path

    command = [
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        final_output_path,
    ]

    result = run_ffmpeg(command)
    if result.returncode != 0:
        logger.warning("FFmpeg muxing warning: %s. Attempting copy fallback.", result.stderr.strip()[:200])
        # Fallback copy
        copy_cmd = [
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            final_output_path,
        ]
        copy_res = run_ffmpeg(copy_cmd)
        if copy_res.returncode != 0:
            shutil.copy2(video_path, final_output_path)

    logger.info("Successfully merged audio and exported final video to %s", final_output_path)
    return final_output_path
