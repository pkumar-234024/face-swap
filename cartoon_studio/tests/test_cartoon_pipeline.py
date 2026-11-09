from __future__ import annotations

import os
import shutil
from pathlib import Path
import pytest
import numpy as np
import cv2

from cartoon_studio.config import PROJECT_ROOT, CARTOON_STYLES
from cartoon_studio.utils.helpers import get_device_info, format_duration, format_file_size
from cartoon_studio.utils.validation import validate_video_file
from cartoon_studio.modules.video_analyzer import VideoAnalyzer
from cartoon_studio.modules.pose_depth_guide import PoseDepthGuide
from cartoon_studio.modules.style_manager import StyleManager, STYLE_PROFILES
from cartoon_studio.modules.neural_cartoonizer import NeuralCartoonizer
from cartoon_studio.modules.animatediff_engine import AnimateDiffEngine
from cartoon_studio.modules.temporal_stabilizer import TemporalStabilizer
from cartoon_studio.modules.upscaler import Upscaler
from cartoon_studio.modules.audio_processor import AudioProcessor
from cartoon_studio.modules.video_exporter import VideoExporter
from cartoon_studio.modules.processing_pipeline import CartoonStudioPipeline

TEST_VIDEO = PROJECT_ROOT / "cartoon_studio" / "assets" / "sample_video.mp4"
if not TEST_VIDEO.exists():
    # Fallback to any uploaded video
    uploaded_videos = list((PROJECT_ROOT / "cartoon_studio" / "uploads").glob("*.mp4"))
    if uploaded_videos:
        TEST_VIDEO = uploaded_videos[0]
TEMP_TEST_DIR = PROJECT_ROOT / "cartoon_studio" / "temp" / "test_run"

@pytest.fixture(scope="session", autouse=True)
def setup_teardown():
    TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEMP_TEST_DIR.exists():
        shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)

def test_device_info():
    info = get_device_info()
    assert "device" in info
    assert "torch_version" in info
    assert info["device"] in ("cuda", "cpu")

def test_helpers_formatting():
    assert format_duration(75) == "01:15"
    assert format_duration(3665) == "01:01:05"
    assert "MB" in format_file_size(1024 * 1024 * 5)

def test_video_validation():
    assert TEST_VIDEO.exists(), f"Sample video missing: {TEST_VIDEO}"
    result = validate_video_file(TEST_VIDEO)
    assert result.is_valid is True
    assert result.width > 0
    assert result.height > 0
    assert result.fps > 0
    assert result.frame_count > 0

def test_video_analyzer():
    meta = VideoAnalyzer.analyze(TEST_VIDEO)
    assert meta.width > 0
    assert meta.height > 0
    assert meta.frame_count > 0
    assert meta.duration_seconds > 0
    cuts = VideoAnalyzer.detect_scene_cuts(TEST_VIDEO, max_frames_to_sample=30)
    assert isinstance(cuts, list)
    assert len(cuts) >= 1

def test_pose_depth_guide():
    guide = PoseDepthGuide(use_mediapipe=False)
    dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    res = guide.process_frame(dummy_frame)
    assert res.pose_canvas.shape == (240, 320, 3)
    assert res.depth_canvas.shape == (240, 320, 3)
    assert res.edge_canvas.shape == (240, 320)
    guide.release()

def test_style_manager_profiles():
    names = StyleManager.get_style_names()
    assert len(names) == 5
    for style_name in names:
        profile = StyleManager.get_profile(style_name)
        assert profile.name == style_name
        assert profile.quantization_levels >= 4
        assert profile.saturation_boost > 0.5
        assert len(profile.tint_bgr) == 3

def test_neural_cartoonizer_all_styles():
    dummy_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    cv2.circle(dummy_frame, (80, 60), 30, (50, 100, 220), -1)
    
    for style_name in StyleManager.get_style_names():
        cartoonizer = NeuralCartoonizer(style_name=style_name)
        out = cartoonizer.cartoonize_frame(dummy_frame, intensity=0.9, line_weight=1.2)
        assert out.shape == dummy_frame.shape
        assert out.dtype == np.uint8
        assert not np.isnan(out).any()

def test_animatediff_engine():
    engine = AnimateDiffEngine(style_name="Anime (Japanese Animation)")
    status = engine.check_environment()
    assert "diffusers_installed" in status
    assert "cuda_available" in status
    
    dummy_frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    # Tests that engine processes frames gracefully without crashing (using fallback when diffusers/CUDA is unconfigured)
    out_frames = engine.process_frames([dummy_frame], intensity=0.8)
    assert len(out_frames) == 1
    assert out_frames[0].shape == (100, 100, 3)
    assert not np.isnan(out_frames[0]).any()

def test_temporal_stabilizer():
    stabilizer = TemporalStabilizer(smoothing_factor=0.7)
    f1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    f2 = np.full((100, 100, 3), 105, dtype=np.uint8)
    
    # First frame (init)
    out1 = stabilizer.stabilize_frame(f1, f1, is_scene_cut=False)
    assert out1.shape == f1.shape
    
    # Second frame (smooth)
    out2 = stabilizer.stabilize_frame(f2, f2, is_scene_cut=False)
    assert out2.shape == f2.shape
    
    # Reset
    stabilizer.reset()
    assert stabilizer.prev_cartoon is None

def test_upscaler():
    dim = Upscaler.calculate_scaled_dimensions(1920, 1080, (1280, 720))
    assert dim == (1280, 720)
    
    dummy_frame = np.zeros((100, 150, 3), dtype=np.uint8)
    upscaled = Upscaler.upscale_frame(dummy_frame, (300, 200))
    assert upscaled.shape == (200, 300, 3)

def test_video_exporter_and_comparison():
    frames = [np.full((120, 160, 3), i * 10, dtype=np.uint8) for i in range(15)]
    out_mp4 = TEMP_TEST_DIR / "test_export.mp4"
    success = VideoExporter.encode_frames_to_mp4(frames, out_mp4, fps=15.0, crf=22)
    assert success is True
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 0

    comp_mp4 = TEMP_TEST_DIR / "test_comp.mp4"
    success_comp = VideoExporter.create_side_by_side_comparison(out_mp4, out_mp4, comp_mp4, crf=22)
    assert success_comp is True
    assert comp_mp4.exists()
    assert comp_mp4.stat().st_size > 0

def test_audio_processor_and_short_clip_pipeline():
    # Create a 1-second test video clip from assets/video1.mp4
    short_input = TEMP_TEST_DIR / "short_sample.mp4"
    import subprocess
    from cartoon_studio.utils.helpers import resolve_ffmpeg_binary
    ffmpeg_bin = resolve_ffmpeg_binary()
    cmd = [
        ffmpeg_bin, "-y", "-ss", "00:00:00", "-i", str(TEST_VIDEO),
        "-t", "1.0", "-c", "copy", str(short_input)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0 and short_input.exists()

    # Run full pipeline on the 1-second sample
    pipeline = CartoonStudioPipeline(job_id="pytest_run")
    pipeline_result = pipeline.process(
        video_path=short_input,
        style_name="Classic 2D Cartoon",
        engine_name="High-Performance Neural Engine",
        animation_intensity=0.85,
        temporal_smoothing=True,
        preserve_pose=False, # Fast test mode
        preserve_depth=True,
        preserve_audio=True,
        output_resolution="Original",
    )

    assert pipeline_result.success is True
    assert pipeline_result.cartoon_video_path is not None
    assert pipeline_result.cartoon_video_path.exists()
    assert pipeline_result.comparison_video_path is not None
    assert pipeline_result.comparison_video_path.exists()
    assert pipeline_result.stats["frames_processed"] > 0
