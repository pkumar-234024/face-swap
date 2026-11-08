from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.face_detector import FaceDetector
from modules.face_replacement import FaceReplacementPipeline
from modules.processing_pipeline import StudioProcessingPipeline


def test_face_replacement_pipeline_initialization():
    detector = FaceDetector(device="cpu")
    pipeline = FaceReplacementPipeline(detector=detector)
    assert pipeline is not None


def test_load_reference_and_process_frame():
    asset_img = Path("assets/g1.png")
    if not asset_img.exists():
        pytest.skip("assets/g1.png not found")

    ref_bgr = cv2.imread(str(asset_img))
    assert ref_bgr is not None

    detector = FaceDetector(device="cpu")
    pipeline = FaceReplacementPipeline(detector=detector)
    pipeline.load_reference(ref_bgr)

    # Process frame with the same image
    processed = pipeline.process_frame(ref_bgr)
    assert processed is not None
    assert processed.shape == ref_bgr.shape


def test_studio_pipeline_execution_limited_frames(tmp_path):
    asset_img = Path("assets/g1.png")
    asset_vid = Path("assets/video1.mp4")
    if not asset_img.exists() or not asset_vid.exists():
        pytest.skip("Assets g1.png or video1.mp4 missing")

    pipeline = StudioProcessingPipeline(
        settings={
            "face_replacement": True,
            "lip_sync": True,
            "eye_sync": True,
            "blinking_sync": True,
            "expression_sync": True,
            "head_movement_sync": True,
            "lighting_correction": True,
            "seamless_blending": True,
            "preserve_audio": False,
            "output_resolution": "Original",
            "video_quality": "High",
            "ai_modified_label": True,
        }
    )

    # Run for 2 frames to test complete end-to-end flow quickly
    job_res = pipeline.execute(
        source_video_path=str(asset_vid),
        reference_image_path=str(asset_img),
        max_frames=2,
    )

    assert job_res.success is True
    assert job_res.total_frames == 2
    assert job_res.output_video_path is not None
    assert Path(job_res.output_video_path).exists()

