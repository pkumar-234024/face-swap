from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.video_analyzer import VideoAnalyzer
from modules.video_exporter import VideoExporter
from modules.video_processor import process_video_stream


def test_resolve_target_dimensions():
    # 1080p source into 720p HD (1280x720)
    w720, h720 = VideoExporter.resolve_target_dimensions(1920, 1080, "720p HD (1280x720)")
    assert (w720, h720) == (1280, 720)

    # 720p source into 1080p Full HD (1920x1080)
    w1080, h1080 = VideoExporter.resolve_target_dimensions(1280, 720, "1080p Full HD (1920x1080)")
    assert (w1080, h1080) == (1920, 1080)

    # 4K
    w4k, h4k = VideoExporter.resolve_target_dimensions(1920, 1080, "4K Ultra HD (3840x2160)")
    assert (w4k, h4k) == (3840, 2160)

    # Original even dimension verification
    w_orig, h_orig = VideoExporter.resolve_target_dimensions(1921, 1079, "Original")
    assert w_orig % 2 == 0
    assert h_orig % 2 == 0


def test_video_analyzer_on_asset():
    asset = Path("assets/video1.mp4")
    if not asset.exists():
        pytest.skip("assets/video1.mp4 not found")

    res = VideoAnalyzer.analyze(str(asset))
    assert res.is_valid is True
    assert res.width == 1080 or res.width == 1920
    assert res.fps > 0
    assert res.frame_count > 0


def test_process_video_stream_synthetic(tmp_path):
    # Create small 5-frame synthetic MP4 video
    test_video_path = str(tmp_path / "test_in.mp4")
    writer = cv2.VideoWriter(
        test_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (320, 240),
    )
    for i in range(5):
        frame = np.full((240, 320, 3), i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    out_video_path = str(tmp_path / "test_out.mp4")

    # Invert frame filter
    def inverter(f):
        return cv2.bitwise_not(f)

    proc_result = process_video_stream(
        input_video_path=test_video_path,
        output_video_path=out_video_path,
        frame_processor=inverter,
    )

    assert proc_result.success is True
    assert proc_result.total_frames_processed == 5
    assert Path(out_video_path).exists()

