from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

from cartoon_studio.config import (
    RESOLUTION_PRESETS,
    CRF_PRESETS,
    OUTPUTS_DIR,
    TEMP_DIR,
)
from cartoon_studio.utils.file_handler import FileHandler
from cartoon_studio.utils.progress_manager import ProgressManager, ProgressCallback
from cartoon_studio.modules.video_analyzer import VideoAnalyzer, VideoMetadata
from cartoon_studio.modules.pose_depth_guide import PoseDepthGuide
from cartoon_studio.modules.neural_cartoonizer import NeuralCartoonizer
from cartoon_studio.modules.animatediff_engine import AnimateDiffEngine
from cartoon_studio.modules.temporal_stabilizer import TemporalStabilizer
from cartoon_studio.modules.upscaler import Upscaler
from cartoon_studio.modules.audio_processor import AudioProcessor
from cartoon_studio.modules.video_exporter import VideoExporter, VideoStreamWriter

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    success: bool
    error_message: Optional[str] = None
    cartoon_video_path: Optional[Path] = None
    comparison_video_path: Optional[Path] = None
    metadata: Optional[VideoMetadata] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

class CartoonStudioPipeline:
    """Production 10-step orchestrator for video-to-2D cartoon animation synthesis."""

    def __init__(self, job_id: Optional[str] = None):
        self.job_dir = FileHandler.create_job_dir(job_id)
        self.progress_mgr = ProgressManager(total_stages=10)

    def process(
        self,
        video_path: str | Path,
        style_name: str = "Classic 2D Cartoon",
        engine_name: str = "High-Performance Neural Engine",
        animation_intensity: float = 0.85,
        line_weight: float = 1.0,
        temporal_smoothing: bool = True,
        preserve_pose: bool = True,
        preserve_depth: bool = True,
        preserve_audio: bool = True,
        output_resolution: str = "Original",
        quality_preset: str = "High Quality (CRF 18)",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        """Run the end-to-end 2D cartoon video transformation pipeline."""
        start_time = time.time()
        video_path = Path(video_path).resolve()

        def report(update):
            if progress_callback:
                progress_callback(update)

        try:
            # -------------------------------------------------------------
            # Stage 1: Video Analysis & Scene Cut Detection
            # -------------------------------------------------------------
            self.progress_mgr.start_stage(1, "Analyzing video and detecting scenes")
            report(self.progress_mgr.update_frame(0, "Analyzing video streams and container metadata..."))
            
            meta = VideoAnalyzer.analyze(video_path)
            scene_cuts = VideoAnalyzer.detect_scene_cuts(video_path, max_frames_to_sample=500)
            meta.scene_cuts = scene_cuts
            report(self.progress_mgr.complete_stage(f"Analysis complete: {meta.width}x{meta.height} @ {meta.fps:.1f} fps, {meta.frame_count} frames"))

            # -------------------------------------------------------------
            # Stage 2: Audio Stream Extraction
            # -------------------------------------------------------------
            self.progress_mgr.start_stage(2, "Extracting audio track")
            extracted_audio_path = self.job_dir / "original_audio.aac"
            has_valid_audio = False
            if preserve_audio and meta.has_audio:
                report(self.progress_mgr.update_frame(0, "Extracting original audio track losslessly..."))
                has_valid_audio = AudioProcessor.extract_audio(video_path, extracted_audio_path)
            report(self.progress_mgr.complete_stage("Audio extraction complete" if has_valid_audio else "No audio stream or audio preservation disabled"))

            # -------------------------------------------------------------
            # Stage 3: Guidance & Motion Tracking Setup
            # -------------------------------------------------------------
            self.progress_mgr.start_stage(3, "Initializing pose and motion guidance")
            report(self.progress_mgr.update_frame(0, "Setting up OpenPose & Depth structural guidance..."))
            pose_guide = PoseDepthGuide(use_mediapipe=preserve_pose)
            report(self.progress_mgr.complete_stage("Pose and motion guidance initialized"))

            # -------------------------------------------------------------
            # Stage 4: Engine & Style Profile Configuration
            # -------------------------------------------------------------
            self.progress_mgr.start_stage(4, "Configuring 2D cartoon rendering engine")
            report(self.progress_mgr.update_frame(0, f"Configuring {style_name} shaders..."))
            
            # Select engine
            use_animatediff = ("AnimateDiff" in engine_name)
            animatediff_engine: Optional[AnimateDiffEngine] = None
            neural_engine = NeuralCartoonizer(style_name=style_name)

            if use_animatediff:
                report(self.progress_mgr.update_frame(0, "Checking AnimateDiff models..."))
                animatediff_engine = AnimateDiffEngine(style_name=style_name)
                # Attempt to initialize pipeline if supported
                if animatediff_engine.is_diffusers_supported():
                    animatediff_engine.initialize_pipeline()

            report(self.progress_mgr.complete_stage(f"Engine ready: {engine_name}"))

            # -------------------------------------------------------------
            # Stage 5 & 6: Frame-by-Frame Cartoonization & Motion Guidance
            # -------------------------------------------------------------
            self.progress_mgr.start_stage(5, "Transforming frames into 2D cartoon", total_frames=meta.frame_count)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open video for reading frames.")

            stabilizer = TemporalStabilizer(smoothing_factor=0.75) if temporal_smoothing else None
            
            # Determine output scaled dimensions
            target_preset = RESOLUTION_PRESETS.get(output_resolution)
            target_w, target_h = Upscaler.calculate_scaled_dimensions(meta.width, meta.height, target_preset)
            crf_val = CRF_PRESETS.get(quality_preset, 18)

            # Stream directly to H.264 video encoder on the fly (near-zero RAM usage)
            temp_video_only = self.job_dir / "cartoon_video_only.mp4"
            writer = VideoStreamWriter(
                output_path=temp_video_only,
                width=target_w,
                height=target_h,
                fps=meta.fps,
                crf=crf_val
            )

            frame_idx = 0
            scene_cut_set = set(meta.scene_cuts)

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                is_cut = (frame_idx in scene_cut_set)

                # Step A: Motion & Depth extraction
                if preserve_pose or preserve_depth:
                    guidance = pose_guide.process_frame(frame_bgr)
                else:
                    guidance = None

                # Step B: Cartoon stylization
                if use_animatediff and animatediff_engine is not None:
                    pose_list = [guidance.pose_canvas] if (guidance and preserve_pose) else None
                    depth_list = [guidance.depth_canvas] if (guidance and preserve_depth) else None
                    cartoon_frame = animatediff_engine.process_frames(
                        frames_bgr=[frame_bgr],
                        pose_canvases=pose_list,
                        depth_canvases=depth_list,
                        intensity=animation_intensity,
                        line_weight=line_weight
                    )[0]
                else:
                    cartoon_frame = neural_engine.cartoonize_frame(
                        frame_bgr=frame_bgr,
                        intensity=animation_intensity,
                        line_weight=line_weight
                    )

                # Step C: Temporal Anti-Flicker Stabilization
                if stabilizer is not None:
                    cartoon_frame = stabilizer.stabilize_frame(
                        current_cartoon_bgr=cartoon_frame,
                        current_orig_bgr=frame_bgr,
                        is_scene_cut=is_cut
                    )

                # Step D: Resolution Scaling / Upscaling
                if target_w != meta.width or target_h != meta.height:
                    cartoon_frame = Upscaler.upscale_frame(
                        cartoon_frame, (target_w, target_h), sharpen_edges=True
                    )

                # Step E: Stream frame directly to H.264 MP4 encoder
                writer.write_frame(cartoon_frame)
                frame_idx += 1

                if meta.frame_count <= 80 or frame_idx % 2 == 0 or frame_idx == meta.frame_count:
                    report(self.progress_mgr.update_frame(frame_idx))

            cap.release()
            pose_guide.release()
            
            # Finalize H.264 video stream
            writer_success = writer.close()
            if not writer_success or not temp_video_only.exists():
                raise RuntimeError("Failed to finalize H.264 video stream.")
                
            report(self.progress_mgr.complete_stage(f"Transformed and encoded {frame_idx} frames"))

            # -------------------------------------------------------------
            # Stage 7 & 8: Audio-Video Synchronization & Remuxing
            # -------------------------------------------------------------
            self.progress_mgr.start_stage(7, "Synchronizing audio and video")
            final_cartoon_name = f"cartoon_{style_name.lower().replace(' ', '_')}_{int(time.time())}.mp4"
            final_cartoon_path = OUTPUTS_DIR / final_cartoon_name

            if has_valid_audio and extracted_audio_path.exists():
                report(self.progress_mgr.update_frame(0, "Remuxing original audio with frame synchronization..."))
                AudioProcessor.remux_audio(
                    video_only_path=temp_video_only,
                    audio_path=extracted_audio_path,
                    output_path=final_cartoon_path,
                    crf=crf_val
                )
            else:
                # Direct copy video only
                import shutil
                shutil.copy2(temp_video_only, final_cartoon_path)

            report(self.progress_mgr.complete_stage(f"Final cartoon video saved: {final_cartoon_name}"))

            # -------------------------------------------------------------
            # Stage 8/9: Side-by-Side Comparison Video Generation
            # -------------------------------------------------------------
            self.progress_mgr.start_stage(8, "Generating side-by-side comparison video")
            report(self.progress_mgr.update_frame(0, "Creating dual synchronous comparison view..."))
            
            comparison_name = f"comparison_{style_name.lower().replace(' ', '_')}_{int(time.time())}.mp4"
            comparison_video_path = OUTPUTS_DIR / comparison_name

            VideoExporter.create_side_by_side_comparison(
                original_video_path=video_path,
                cartoon_video_path=final_cartoon_path,
                output_path=comparison_video_path,
                crf=crf_val
            )
            report(self.progress_mgr.complete_stage("Comparison video created successfully"))

            # -------------------------------------------------------------
            # Stage 10: Finalization & Metrics
            # -------------------------------------------------------------
            self.progress_mgr.start_stage(10, "Finalizing studio output")
            total_elapsed = time.time() - start_time
            avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0.0

            stats = {
                "elapsed_seconds": round(total_elapsed, 2),
                "frames_processed": frame_idx,
                "avg_processing_fps": round(avg_fps, 2),
                "style": style_name,
                "engine": engine_name,
                "output_resolution": f"{target_w}x{target_h}",
                "final_file_size": final_cartoon_path.stat().st_size if final_cartoon_path.exists() else 0,
            }

            report(self.progress_mgr.complete_stage("All processing complete!"))

            return PipelineResult(
                success=True,
                cartoon_video_path=final_cartoon_path,
                comparison_video_path=comparison_video_path,
                metadata=meta,
                stats=stats,
                logs=list(self.progress_mgr.logs),
            )

        except Exception as ex:
            logger.exception("Pipeline execution encountered an error")
            return PipelineResult(
                success=False,
                error_message=str(ex),
                logs=list(self.progress_mgr.logs),
            )
        finally:
            # Clean up job temp directory
            FileHandler.cleanup_directory(self.job_dir)

