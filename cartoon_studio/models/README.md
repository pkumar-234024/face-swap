# Model Hub & Checkpoint Documentation

The **AI 2D Cartoon Video Studio** utilizes a hybrid dual-engine architecture:

---

## Engine 1: High-Performance Neural & Cel-Shaded Cartoon Engine
- **Architecture**: Multi-stage bilateral surface smoothing, adaptive Difference-of-Gaussians (DoG) ink line extraction, CIELAB color quantization, and style-specific shaders (Anime bloom, Comic halftone/deep inking, Hand-drawn pencil grain, Pixar/Disney storybook radiance).
- **Execution Speed**: 15–30 FPS on local hardware (NVIDIA GTX 1650 or CPU fallback).
- **VRAM Requirements**: < 500 MB VRAM (Zero OOM risk).
- **License**: Apache 2.0 / MIT compatible.

---

## Engine 2: AnimateDiff Video-to-Video Engine (Diffusers + ControlNet)
For offline high-end GPU workstations:
- **Motion Adapter**: `guoyww/animatediff-motion-adapter-v1-5-2`
  - Checkpoint: Motion module trained on 16-frame video clips.
  - License: CreativeML Open RAIL-M.
- **Base Diffusion Models**:
  - `runwayml/stable-diffusion-v1-5`
  - Style checkpoints: `ToonYou`, `Lyriel`, `Flat2DAnimerge`.
- **ControlNet Structural Guidance**:
  - OpenPose: `lllyasviel/control_v11p_sd15_openpose` (human body movements, gestures, hand pose).
  - Depth: `lllyasviel/control_v11f1p_sd15_depth` (scene perspective, composition, camera angles).
- **Memory Optimizations**:
  - `enable_vae_slicing()`
  - `enable_model_cpu_offload()`
  - Sliding temporal window batching.

