from __future__ import annotations

import importlib
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import torch

from cartoon_studio.config import MODELS_DIR
from cartoon_studio.modules.style_manager import StyleProfile, StyleManager
from cartoon_studio.modules.neural_cartoonizer import NeuralCartoonizer

logger = logging.getLogger(__name__)

class AnimateDiffEngine:
    """AnimateDiff Video-to-Video Engine with ControlNet OpenPose and Depth guidance.
    
    Provides Diffusers-based motion synthesis with memory-efficient VAE slicing,
    ControlNet structural guidance, and seamless integration with the Neural Cartoonizer fallback.
    """

    def __init__(
        self,
        style_name: str = "Classic 2D Cartoon",
        device: Optional[str] = None,
        use_low_vram: bool = True
    ):
        self.style_name = style_name
        self.profile: StyleProfile = StyleManager.get_profile(style_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_low_vram = use_low_vram
        
        self.pipeline = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.controlnet_loaded = False
        self.fallback_engine = NeuralCartoonizer(style_name=style_name)

    def set_style(self, style_name: str) -> None:
        """Update style profile and prompts."""
        self.style_name = style_name
        self.profile = StyleManager.get_profile(style_name)
        self.fallback_engine.set_style(style_name)

    def check_environment(self) -> Dict[str, Any]:
        """Inspect system hardware and diffusers package availability."""
        diffusers_installed = False
        try:
            importlib.import_module("diffusers")
            diffusers_installed = True
        except ImportError:
            pass

        cuda_available = torch.cuda.is_available()
        status = {
            "diffusers_installed": diffusers_installed,
            "cuda_available": cuda_available,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "load_error": self.load_error,
        }
        return status

    def is_diffusers_supported(self) -> bool:
        """Check if Diffusers and AnimateDiff dependencies are installed and CUDA is ready."""
        try:
            diffusers_mod = importlib.import_module("diffusers")
            return hasattr(diffusers_mod, "MotionAdapter") and torch.cuda.is_available()
        except ImportError:
            return False

    def initialize_pipeline(
        self,
        motion_model_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        use_controlnet: bool = True,
    ) -> bool:
        """Attempt to load AnimateDiff pipeline with optional ControlNet and VAE memory optimizations."""
        env = self.check_environment()
        if not env["diffusers_installed"]:
            self.load_error = "Diffusers package is not installed. Using High-Performance Neural Engine fallback."
            logger.info(self.load_error)
            return False

        if not env["cuda_available"]:
            self.load_error = "CUDA GPU acceleration is required for AnimateDiff diffusion video rendering. Using high-performance neural engine fallback."
            logger.info(self.load_error)
            return False

        try:
            diffusers = importlib.import_module("diffusers")
            AnimateDiffPipeline = getattr(diffusers, "AnimateDiffPipeline", None)
            AnimateDiffControlNetPipeline = getattr(diffusers, "AnimateDiffControlNetPipeline", None)
            MotionAdapter = getattr(diffusers, "MotionAdapter", None)
            DDIMScheduler = getattr(diffusers, "DDIMScheduler", None)
            ControlNetModel = getattr(diffusers, "ControlNetModel", None)

            if MotionAdapter is None or AnimateDiffPipeline is None:
                self.load_error = "Required AnimateDiff classes not found in diffusers module."
                return False

            dtype = torch.float16 if self.device == "cuda" else torch.float32

            logger.info("Initializing AnimateDiff Motion Adapter from %s...", motion_model_id)
            adapter = MotionAdapter.from_pretrained(
                motion_model_id,
                torch_dtype=dtype,
                cache_dir=str(MODELS_DIR)
            )

            # Attempt ControlNet loading for pose/depth guidance if requested
            controlnet = None
            if use_controlnet and ControlNetModel is not None and AnimateDiffControlNetPipeline is not None:
                try:
                    logger.info("Loading ControlNet OpenPose guidance model...")
                    controlnet_openpose = ControlNetModel.from_pretrained(
                        "lllyasviel/control_v11p_sd15_openpose",
                        torch_dtype=dtype,
                        cache_dir=str(MODELS_DIR)
                    )
                    controlnet = controlnet_openpose
                    self.controlnet_loaded = True
                except Exception as cn_ex:
                    logger.warning("ControlNet loading deferred (%s). Proceeding with standard AnimateDiff pipeline.", cn_ex)
                    controlnet = None
                    self.controlnet_loaded = False

            logger.info("Loading AnimateDiff Base Pipeline from %s...", base_model_id)
            if controlnet is not None and AnimateDiffControlNetPipeline is not None:
                pipe = AnimateDiffControlNetPipeline.from_pretrained(
                    base_model_id,
                    motion_adapter=adapter,
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    cache_dir=str(MODELS_DIR)
                )
            else:
                pipe = AnimateDiffPipeline.from_pretrained(
                    base_model_id,
                    motion_adapter=adapter,
                    torch_dtype=dtype,
                    cache_dir=str(MODELS_DIR)
                )

            scheduler = DDIMScheduler.from_config(
                pipe.scheduler.config,
                clip_sample=False,
                timestep_spacing="linspace",
                steps_offset=1
            )
            pipe.scheduler = scheduler

            # Memory optimization settings
            if self.use_low_vram:
                pipe.enable_vae_slicing()
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                elif hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                else:
                    pipe.to(self.device)
            else:
                pipe.to(self.device)

            self.pipeline = pipe
            self.is_loaded = True
            self.load_error = None
            return True

        except Exception as e:
            self.load_error = f"AnimateDiff pipeline initialization failed: {str(e)}"
            logger.warning(self.load_error)
            self.is_loaded = False
            return False

    def process_frames(
        self,
        frames_bgr: List[np.ndarray],
        pose_canvases: Optional[List[np.ndarray]] = None,
        depth_canvases: Optional[List[np.ndarray]] = None,
        intensity: float = 0.85,
        line_weight: float = 1.0,
    ) -> List[np.ndarray]:
        """Process a sequence of video frames into 2D cartoon animation."""
        if not frames_bgr:
            return []

        # If AnimateDiff is not loaded or cannot run on current hardware, use neural engine
        if not self.is_loaded or self.pipeline is None:
            output_frames: List[np.ndarray] = []
            for frame in frames_bgr:
                out_frame = self.fallback_engine.cartoonize_frame(
                    frame_bgr=frame,
                    intensity=intensity,
                    line_weight=line_weight
                )
                output_frames.append(out_frame)
            return output_frames

        # Run AnimateDiff Video-to-Video generation in batches of up to 16 frames
        try:
            return self._run_animatediff_batch(
                frames_bgr=frames_bgr,
                pose_canvases=pose_canvases,
                depth_canvases=depth_canvases,
                intensity=intensity
            )
        except Exception as ex:
            logger.error("AnimateDiff batch execution failed: %s. Falling back to Neural Cartoonizer.", ex)
            output_frames = []
            for frame in frames_bgr:
                out_frame = self.fallback_engine.cartoonize_frame(
                    frame_bgr=frame,
                    intensity=intensity,
                    line_weight=line_weight
                )
                output_frames.append(out_frame)
            return output_frames

    def _run_animatediff_batch(
        self,
        frames_bgr: List[np.ndarray],
        pose_canvases: Optional[List[np.ndarray]],
        depth_canvases: Optional[List[np.ndarray]],
        intensity: float
    ) -> List[np.ndarray]:
        """Execute diffusers inference on video batch with motion & structural guidance."""
        from PIL import Image

        prompt = self.profile.prompt
        negative_prompt = (
            "photorealistic, real-life photo, noisy, grainy, blurry, deformed limbs, "
            "flicker, bad anatomy, disfigured face, low resolution"
        )

        h, w = frames_bgr[0].shape[:2]
        # Diffusers dimensions divisible by 64
        target_w = (w // 64) * 64
        target_h = (h // 64) * 64
        target_w = max(384, min(768, target_w))
        target_h = max(384, min(768, target_h))

        # Convert input frames to RGB PIL
        pil_frames = [
            Image.fromarray(cv2.cvtColor(cv2.resize(f, (target_w, target_h)), cv2.COLOR_BGR2RGB))
            for f in frames_bgr
        ]

        generator = torch.Generator(device=self.device).manual_seed(42)
        num_inference_steps = max(10, int(25 * intensity))
        guidance_scale = 7.5

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": len(pil_frames),
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        # If ControlNet is active and pose canvases exist, feed conditioning images
        if self.controlnet_loaded and pose_canvases and len(pose_canvases) == len(frames_bgr):
            pil_poses = [
                Image.fromarray(cv2.cvtColor(cv2.resize(p, (target_w, target_h)), cv2.COLOR_BGR2RGB))
                for p in pose_canvases
            ]
            kwargs["image"] = pil_poses

        # Execute generation
        output = self.pipeline(**kwargs)

        # Convert back to BGR cv2 frames resized to original resolution
        generated_frames = []
        for pil_img in output.frames[0]:
            arr_rgb = np.array(pil_img)
            arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
            arr_resized = cv2.resize(arr_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
            generated_frames.append(arr_resized)

        return generated_frames
