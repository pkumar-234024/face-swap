from __future__ import annotations

import os
from pathlib import Path

import pytest

from modules.audio_processor import has_audio_stream, resolve_ffmpeg_path


def test_resolve_ffmpeg_path():
    ffmpeg_exe = resolve_ffmpeg_path()
    assert ffmpeg_exe is not None
    assert os.path.exists(ffmpeg_exe)


def test_has_audio_stream_on_missing_file():
    assert has_audio_stream("non_existent_file.mp4") is False


def test_audio_probe_on_asset():
    video_asset = Path("assets/video1.mp4")
    if not video_asset.exists():
        pytest.skip("assets/video1.mp4 not found")

    result = has_audio_stream(str(video_asset))
    assert isinstance(result, bool)

