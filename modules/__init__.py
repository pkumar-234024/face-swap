"""Core processing modules for AI Face Replacement & Animation Studio."""

from modules.audio_processor import extract_audio, has_audio_stream, merge_audio
from modules.expression_tracker import ExpressionTracker, FacialExpression
from modules.eye_sync import EyeState, EyeSynchronizer
from modules.face_aligner import align_face_to_canonical, create_face_mask, estimate_similarity_transform
from modules.face_blending import FaceBlender
from modules.face_detector import FaceDetector
from modules.face_replacement import FaceReplacementPipeline
from modules.face_tracker import FaceTracker
from modules.facial_animation import FacialAnimationPipeline
from modules.facial_landmarks import FacialLandmarks, FacialLandmarkTracker
from modules.head_pose import HeadPose, HeadPoseEstimator
from modules.lighting_correction import LightingCorrector
from modules.lip_sync import LipState, LipSynchronizer
from modules.processing_pipeline import PipelineJobResult, StudioProcessingPipeline
from modules.video_analyzer import VideoAnalysisResult, VideoAnalyzer
from modules.video_exporter import VideoExporter
from modules.video_processor import VideoProcessingResult, process_video_stream

__all__ = [
    "FaceDetector",
    "FacialLandmarks",
    "FacialLandmarkTracker",
    "FaceTracker",
    "HeadPose",
    "HeadPoseEstimator",
    "EyeState",
    "EyeSynchronizer",
    "LipState",
    "LipSynchronizer",
    "FacialExpression",
    "ExpressionTracker",
    "LightingCorrector",
    "FaceBlender",
    "FacialAnimationPipeline",
    "FaceReplacementPipeline",
    "StudioProcessingPipeline",
    "PipelineJobResult",
    "VideoAnalyzer",
    "VideoAnalysisResult",
    "VideoExporter",
    "VideoProcessingResult",
    "process_video_stream",
    "align_face_to_canonical",
    "create_face_mask",
    "estimate_similarity_transform",
    "extract_audio",
    "merge_audio",
    "has_audio_stream",
]

