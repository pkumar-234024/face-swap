# AI Face Replacement & Facial Animation Studio

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Academic%20Research-green.svg)](models/README.md)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A modular, production-grade Python studio for identity-conditioned facial animation, lip synchronization, eye tracking, expression transfer, head pose tracking, and realistic facial replacement with strict consent verification and ethical AI disclosure labeling.

---

## 🌟 Key Capabilities

1. **Identity-Conditioned Face Replacement**:
   Transfers authorized likeness into video while preserving facial texture, skin grain, and bone structure.
2. **Realistic Lip Synchronization**:
   Tracks mouth landmarks, speech aperture (MAR), upper/lower lip displacement, and jaw motion.
3. **Eye Movement & Blinking Synchronization**:
   Computes Eye Aspect Ratio (EAR) and gaze vectors to naturally synchronize blinking patterns and pupil orientation without distortion.
4. **Facial Expression Synchronization**:
   Captures and transfers emotion action units: smiling, laughter, surprise, sadness, anger, and eyebrow movements.
5. **3D Head Pose Estimation**:
   Solves Perspective-n-Point (solvePnP) using 3D anthropometric models to compute yaw, pitch, roll, and perspective alignment.
6. **Facial Landmark Tracking**:
   Tracks 68 standard and 106 dense 2D/3D landmarks with velocity-sensitive exponential moving average (EMA) temporal smoothing.
7. **Lighting & Color Correction**:
   Transfers ambient lighting in perceptual CIE $L^*a^*b^*$ color space and injects high-frequency skin micro-texture.
8. **Realistic Seamless Blending**:
   Poisson seamless cloning (`cv2.seamlessClone`) and multi-band Laplacian pyramid blending with feathered boundary masks.
9. **Audio Stream Preservation**:
   Extracts and re-muxes original speech, ambient background sounds, and effects using FFmpeg with frame-accurate synchronization.
10. **Multi-Resolution Video Export**:
    Exports to 720p HD, 1080p Full HD, or 4K Ultra HD using H.264 (`libx264`) video encoding.
11. **Consent Verification & AI Disclosure**:
    Requires explicit user certification of authorized consent and embeds a visible "AI-MODIFIED | CONSENTED LIKENESS" watermark.

---

## 🏗️ Project Architecture

```
face-swap/
│
├── app.py                     # Streamlit interactive dashboard application
├── config.py                  # Global settings, resolutions, paths, and limits
├── requirements.txt           # Verified Python dependencies
├── README.md                  # System documentation & setup guide
├── .env.example               # Environment variables template
├── .gitignore                 # Version control exclusions
│
├── modules/                   # Core computer vision & AI modules
│   ├── __init__.py
│   ├── video_processor.py     # Memory-efficient streaming video processor
│   ├── video_analyzer.py      # Metadata, codec, resolution, and format analyzer
│   ├── face_detector.py       # InsightFace SCRFD detector with Haar fallback
│   ├── face_aligner.py        # 5-point Umeyama canonical alignment & masks
│   ├── face_tracker.py        # Temporal bounding box tracker with inertia
│   ├── facial_landmarks.py    # 68 & 106 landmark tracker & smoother
│   ├── head_pose.py           # 3D Head Pose (PnP yaw, pitch, roll estimator)
│   ├── expression_tracker.py  # Facial action units & emotion tracker
│   ├── lip_sync.py            # Lip aspect ratio (MAR) & speech articulation
│   ├── eye_sync.py            # Eye aspect ratio (EAR) & blinking synchronizer
│   ├── facial_animation.py    # Delaunay mesh piecewise affine warping engine
│   ├── face_replacement.py    # End-to-end face replacement pipeline
│   ├── face_blending.py       # Poisson cloning & Laplacian pyramid blending
│   ├── lighting_correction.py # CIE LAB Reinhard color transfer & texture
│   ├── audio_processor.py     # FFmpeg audio extraction & lossless muxing
│   ├── video_exporter.py      # Multi-resolution H.264 MP4 video encoder
│   └── processing_pipeline.py # 15-step studio orchestrator
│
├── utils/                     # Helper utilities & managers
│   ├── __init__.py
│   ├── file_handler.py        # Secure file saving, sanitization, and cleanup
│   ├── validation.py          # Video & image integrity and blur validators
│   ├── model_manager.py       # Model path discovery, extraction, & status
│   ├── progress_manager.py    # Real-time frame progress, FPS, and ETA tracker
│   ├── logging_config.py      # Unified logger configuration
│   └── helpers.py             # Formatters, directory tools, device selector
│
├── models/                    # AI model weights & documentation
│   ├── README.md              # Model sources, versions, and licenses
│   └── buffalo_l/             # Pre-configured InsightFace models
│       ├── det_10g.onnx       # Face detector (SCRFD 10G)
│       ├── w600k_r50.onnx     # Identity ArcFace ResNet50
│       ├── 1k3d68.onnx        # 3D 68 facial landmarks
│       ├── 2d106det.onnx      # Dense 2D 106 landmarks
│       └── genderage.onnx     # Gender & age attribute model
│
├── assets/                    # Static assets & stylesheets
│   ├── css/
│   │   └── style.css          # Modern dark glassmorphism theme
│   └── images/                # Reference test assets
│
└── tests/                     # Unit and integration test suite
    ├── test_uploads.py        # File validation tests
    ├── test_video_processing.py # Video analysis & streaming tests
    ├── test_face_detection.py # Detection, landmarks, pose, eyes, lips tests
    ├── test_audio_sync.py     # FFmpeg & audio stream tests
    └── test_pipeline.py       # End-to-end studio pipeline tests
```

---

## 💻 Hardware Requirements

- **Operating System**: Windows 10 / 11 (64-bit), Linux, or macOS.
- **CPU**: Intel Core i5 / AMD Ryzen 5 or higher.
- **RAM**: Minimum 8 GB (16 GB recommended for 1080p and 4K video rendering).
- **GPU (Optional)**: NVIDIA GPU with CUDA 11.8+ or 12.1+ and cuDNN for hardware acceleration via `CUDAExecutionProvider`.
- **CPU Fallback**: Full CPU execution is automatically supported via ONNX Runtime `CPUExecutionProvider` and OpenCV multi-threaded pipelines.

---

## 🪟 Windows Installation Guide

### Step 1: Install Python 3.11+
Download and install Python 3.11 or newer from [python.org](https://www.python.org/downloads/). Ensure you check **"Add Python to PATH"** during installation.

### Step 2: Open PowerShell in Project Directory
Navigate to the project root:
```powershell
cd D:\F\mine\face-swap
```

### Step 3: Create and Activate Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: FFmpeg Setup
FFmpeg is automatically handled via `imageio-ffmpeg` which bundles a standalone Windows x86_64 FFmpeg executable. If you prefer using system FFmpeg:
```powershell
winget install Gyan.Dev.FFmpeg
```

### Step 6: Verify AI Models
The Buffalo_L models are stored in `models/buffalo_l/`. If `buffalo_l.zip` is present, it will be automatically unpacked on first launch:
- `det_10g.onnx`
- `w600k_r50.onnx`
- `1k3d68.onnx`
- `2d106det.onnx`
- `genderage.onnx`

*(Optional)*: If you wish to use deep latent swapping instead of landmark mesh animation, place `inswapper_128.onnx` directly into the `models/` directory.

### Step 7: Launch the Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Independent Tests

Execute the automated test suite using pytest:
```powershell
python -m pytest tests/ -v
```

All 21 unit and integration tests will execute and verify:
- Upload verification and sanitization
- SCRFD face detection and 68/106 landmark tracking
- Eye Aspect Ratio (EAR) and blink analysis
- Mouth Aspect Ratio (MAR) and speech tracking
- 3D Head Pose estimation (solvePnP)
- Video resolution resizing (720p, 1080p, 4K)
- Audio stream extraction and H.264 muxing
- End-to-end studio pipeline processing

---

## 🛡️ Security, Privacy & Ethics

1. **Consent Certification**:
   The user interface enforces a mandatory consent verification checkbox certifying that the user has explicit permission to use the reference individual's likeness.
2. **AI-Generated Content Disclosure**:
   All rendered videos automatically embed a clean, visible disclosure badge: `AI-MODIFIED | CONSENTED LIKENESS`.
3. **No Impersonation or Deceptive Misuse**:
   This studio is engineered exclusively for authorized creative production, video localization, accessibility, and facial animation research.
4. **Temporary File Management**:
   All intermediate frame extractions and temporary audio slices are isolated per job ID in `temp/` and deleted upon job completion.
