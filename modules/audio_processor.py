from __future__ import annotations

import os
import shutil
import subprocess

try:
    from imageio_ffmpeg import get_ffmpeg_exe
except ImportError:  # pragma: no cover
    get_ffmpeg_exe = None


def resolve_ffmpeg_path() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        shutil.which("ffmpeg.exe"),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    if get_ffmpeg_exe is not None:
        fallback = get_ffmpeg_exe()
        if fallback and os.path.exists(fallback):
            return fallback

    raise FileNotFoundError(
        "FFmpeg was not found on PATH. Install FFmpeg for Windows and add its bin folder to PATH, "
        "for example C:\\ffmpeg\\bin, or install via winget: winget install Gyan.Dev.FFmpeg"
    )


def _run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[str]:
    ffmpeg_path = resolve_ffmpeg_path()
    full_command = [ffmpeg_path, *command]
    return subprocess.run(full_command, capture_output=True, text=True, check=False)


def extract_audio(video_path: str, output_audio_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(os.path.dirname(output_audio_path) or ".", exist_ok=True)
    command = [
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "copy",
        output_audio_path,
    ]
    result = _run_ffmpeg(command)
    if result.returncode != 0:
        raise RuntimeError(f"Unable to extract audio: {result.stderr.strip() or result.stdout.strip()}")
    return output_audio_path


def merge_audio(video_path: str, audio_path: str, final_output_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        return video_path

    command = [
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
    result = _run_ffmpeg(command)
    if result.returncode != 0:
        raise RuntimeError(f"Unable to merge audio: {result.stderr.strip() or result.stdout.strip()}")
    return final_output_path
