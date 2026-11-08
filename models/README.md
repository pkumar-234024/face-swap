# AI Models & Licenses Directory

This directory contains deep learning and computer vision models utilized by the **AI Face Replacement & Animation Studio**.

---

## 1. InsightFace Buffalo_L Model Suite

- **Location**: `models/buffalo_l/`
- **Source**: InsightFace Project (DeepInsight)
- **Model Components**:
  - `det_10g.onnx`: SCRFD 10G high-accuracy face detection network.
  - `w600k_r50.onnx`: ArcFace ResNet50 identity feature extractor (outputs 512-dimensional facial embedding vectors).
  - `1k3d68.onnx`: 3D 68-point facial landmark estimation network.
  - `2d106det.onnx`: Dense 2D 106-point facial landmark detector.
  - `genderage.onnx`: Attribute analysis model for age and gender estimation.
- **License**: The InsightFace models in `buffalo_l` are provided for **academic research and non-commercial educational purposes** under the InsightFace Software License. Commercial deployment requires commercial licensing from DeepInsight.
- **Model Download**: Downloaded automatically or unpack `buffalo_l.zip` into `models/buffalo_l/`.

---

## 2. Optional: Deep Face Swapping Model (`inswapper_128.onnx`)

- **Target Path**: `models/inswapper_128.onnx`
- **Architecture**: Latent Identity-Conditioned InSwapper (128x128 latent generator with AdaIN feature modulation).
- **Download Instruction**:
  ```bash
  # Download from public model repository (e.g. HuggingFace / InsightFace community hub)
  # Place into the models/ folder:
  models/inswapper_128.onnx
  ```
- **Fallback Behavior**:
  When `inswapper_128.onnx` is not present, the studio utilizes its internal high-precision **Piecewise Affine / Delaunay Triangulation & Poisson Blending Facial Animation Engine**, which warps, animates, and blends the authorized reference identity using 68/106 facial landmarks without requiring proprietary unfree swap weights.

---

## 3. Hardware Requirements

- **GPU Acceleration**: NVIDIA GPU with CUDA 11.8+ / 12.1+ and cuDNN support for fast real-time inference via `CUDAExecutionProvider`.
- **CPU Fallback**: Highly optimized CPU inference supported via ONNX Runtime `CPUExecutionProvider` and OpenCV multi-threaded matrix operations.
- **System Memory**: Minimum 8 GB RAM (16 GB recommended for 4K video rendering).

