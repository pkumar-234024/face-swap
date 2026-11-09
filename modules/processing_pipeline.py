from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from config import OUTPUTS_DIR, TEMP_DIR
from modules.audio_processor import extract_audio, merge_audio
from modules.face_detector import FaceDetector
from modules.face_replacement import FaceReplacementPipeline
from modules.video_analyzer import VideoAnalysisResult, VideoAnalyzer
from modules.video_exporter import VideoExporter
from modules.video_processor import process_video_stream
from utils.file_handler import cleanup_file, create_job_temp_dir
from utils.logging_config import get_logger
from utils.progress_manager import ProgressManager, ProgressUpdate

logger = get_logger("processing_pipeline")


@dataclass
class PipelineJobResult:
    """Final result of an end-to-end face replacement and animation job."""

    success: bool
    output_video_path: Optional[str]
    total_frames: int
    duration_seconds: float
    fps: float
    width: int
    height: int
    has_audio: bool
    error_message: Optional[str] = None


class StudioProcessingPipeline:
    """Main orchestrator coordinating video analysis, facial animation, and export."""

    def __init__(
        self,
        settings: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ):
        self.settings = settings or {}
        self.progress_callback = progress_callback
        self.detector = FaceDetector()
        self.replacer = FaceReplacementPipeline(detector=self.detector, settings=self.settings)

    def execute(
        self,
        source_video_path: str,
        reference_image_path: str,
        job_id: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> PipelineJobResult:
        """Executes the complete 15-step studio pipeline."""
        uid = job_id or uuid.uuid4().hex[:8]
        job_dir = create_job_temp_dir(uid)

        raw_video_path = str(job_dir / "raw_processed.mp4")
        temp_audio_path = str(job_dir / "audio.aac")
        final_filename = f"studio_export_{uid}.mp4"
        final_output_path = str((OUTPUTS_DIR / final_filename).resolve())

        progress = ProgressManager(callback=self.progress_callback)

        try:
            # Stage 1: Analyze & Validate Input Video
            progress.set_stage("Analyzing Video", "Extracting video metadata, codecs, and stream properties...")
            video_meta = VideoAnalyzer.analyze(source_video_path)
            if not video_meta.is_valid:
                raise ValueError(f"Video validation failed: {video_meta.error_message}")

            progress.set_total_frames(max_frames if max_frames else video_meta.frame_count)

            # Stage 2: Load and Verify Reference Identity
            progress.set_stage("Analyzing Reference Face", "Detecting authorized identity, landmarks, and 3D facial mesh...")
            from utils.file_handler import read_image_bgr
            ref_bgr = read_image_bgr(reference_image_path)
            if ref_bgr is None or ref_bgr.size == 0:
                ref_bgr = cv2.imread(reference_image_path)
            if ref_bgr is None or ref_bgr.size == 0:
                raise ValueError(f"Unable to read reference face image at {reference_image_path}")

            self.replacer.load_reference(ref_bgr)

            # Stage 3: Audio Extraction
            progress.set_stage("Extracting Audio", "Preserving speech, timing, and soundtrack...")
            audio_path = None
            if self.settings.get("preserve_audio", True) and video_meta.has_audio:
                audio_path = extract_audio(source_video_path, temp_audio_path)

            # Stage 4: Frame Dimensions
            res_preset = self.settings.get("output_resolution", "Original")
            target_dims = VideoExporter.resolve_target_dimensions(
                original_width=video_meta.width,
                original_height=video_meta.height,
                preset_name=res_preset,
            )

            # Stage 5: Frame-by-frame AI Facial Animation & Blending
            progress.set_stage("AI Facial Animation & Blending", "Processing facial dynamics, lips, eyes, and pose...")

            def frame_step_callback(current: int, total: int) -> None:
                progress.advance_frame(1, message=f"Frame {current} / {total}")

            proc_res = process_video_stream(
                input_video_path=source_video_path,
                output_video_path=raw_video_path,
                frame_processor=self.replacer.process_frame,
                progress_callback=frame_step_callback,
                target_dimensions=target_dims,
                max_frames=max_frames,
            )

            if not proc_res.success:
                raise RuntimeError(f"Frame processing error: {proc_res.error_message}")

            # Stage 6: Audio-Video Synchronization & Video Encoding
            progress.set_stage("Final Video Export", "Encoding H.264 MP4 and synchronizing audio track...")
            quality_preset = self.settings.get("video_quality", "High")
            final_exported = VideoExporter.export_video_with_audio(
                raw_video_path=raw_video_path,
                audio_path=audio_path,
                final_output_path=final_output_path,
                quality_preset=quality_preset,
            )

            progress.set_stage("Complete", "AI face replacement and animation completed successfully!")

            return PipelineJobResult(
                success=True,
                output_video_path=final_exported,
                total_frames=proc_res.total_frames_processed,
                duration_seconds=proc_res.duration_seconds,
                fps=proc_res.fps,
                width=proc_res.width,
                height=proc_res.height,
                has_audio=audio_path is not None,
            )

        except Exception as exc:
            logger.error("Pipeline job %s failed: %s", uid, exc, exc_info=True)
            progress.add_error(str(exc))
            progress.set_stage("Failed", f"Error: {exc}")
            return PipelineJobResult(
                success=False,
                output_video_path=None,
                total_frames=0,
                duration_seconds=0.0,
                fps=0.0,
                width=0,
                height=0,
                has_audio=False,
                error_message=str(exc),
            )

        finally:
            # Clean up intermediate raw files, keep final output
            cleanup_file(raw_video_path)
            cleanup_file(temp_audio_path)
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            self.replacer.reset()

