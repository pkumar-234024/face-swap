from __future__ import annotations

import asyncio
import os
import time
import warnings
from pathlib import Path
from typing import Optional

if os.name == "nt":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if selector_policy is not None:
            asyncio.set_event_loop_policy(selector_policy())

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from config import (
    APP_DESCRIPTION,
    APP_SUBTITLE,
    APP_TITLE,
    CRF_PRESETS,
    DEFAULT_SETTINGS,
    RESOLUTION_PRESETS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from modules.face_detector import FaceDetector
from modules.processing_pipeline import StudioProcessingPipeline
from modules.video_analyzer import VideoAnalysisResult, VideoAnalyzer
from modules.video_exporter import VideoExporter
from utils.file_handler import cleanup_directory, cleanup_file, save_uploaded_file
from utils.helpers import format_file_size, format_seconds, get_device
from utils.logging_config import get_logger
from utils.model_manager import ModelManager
from utils.progress_manager import ProgressUpdate
from utils.validation import (
    compute_image_sharpness,
    validate_reference_image,
    validate_reference_image_path,
    validate_video_file,
    validate_video_file_path,
)

logger = get_logger("app")

# Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css() -> None:
    """Injects high-end glassmorphism dark CSS theme."""
    css_path = Path("assets/css/style.css")
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_custom_css()


@st.cache_resource
def get_cached_detector() -> FaceDetector:
    """Initializes and caches FaceDetector engine."""
    return FaceDetector(device=get_device())


@st.cache_resource
def get_model_status():
    """Retrieves AI model availability status."""
    manager = ModelManager()
    return manager.get_system_status()


# Session State Initialization
if "source_video_path" not in st.session_state:
    st.session_state.source_video_path = None
if "reference_image_path" not in st.session_state:
    st.session_state.reference_image_path = None
if "video_meta" not in st.session_state:
    st.session_state.video_meta = None
if "image_info" not in st.session_state:
    st.session_state.image_info = None
if "last_output_video" not in st.session_state:
    st.session_state.last_output_video = None
if "sbs_video_path" not in st.session_state:
    st.session_state.sbs_video_path = None


# -------------------------------------------------------------
# SIDEBAR: SYSTEM STATUS & CONSENT VERIFICATION
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Status")
    device_mode = get_device()
    st.markdown(
        f"<div class='feature-badge'>💻 Compute Device: <b>{device_mode.upper()}</b></div>",
        unsafe_allow_html=True,
    )

    models_info = get_model_status()
    buffalo = models_info.get("buffalo_l", {})
    if buffalo.get("all_present"):
        st.success("✅ Buffalo_L Face Models Ready (SCRFD, ArcFace, 68/106 Lmk)")
    else:
        st.warning("⚠️ Some Buffalo_L models missing; fallback detector active.")

    inswapper = models_info.get("inswapper", {})
    if inswapper.get("present"):
        st.success("✅ Deep InSwapper ONNX Active")
    else:
        st.info("ℹ️ Using Mesh Delaunay Animation Engine (InSwapper optional)")

    st.markdown("---")
    st.markdown("### 🛡️ Consent & Ethical AI")
    st.markdown(
        """
        <div style='font-size: 0.82rem; color: #94a3b8; line-height: 1.45;'>
        This application requires explicit authorization from any individual whose likeness is uploaded.
        All exported videos contain an embedded AI-disclosure badge.
        </div>
        """,
        unsafe_allow_html=True,
    )

    consent_agreed = st.checkbox(
        "I certify that I have obtained legal consent to use this reference person's likeness and agree to visible AI disclosure.",
        value=False,
        key="user_consent_checkbox",
    )

    st.markdown("---")
    st.markdown("### 🧹 Workspace Management")
    if st.button("Clear Uploads & Temp Cache", use_container_width=True):
        cleanup_directory("temp", keep_root=True)
        st.session_state.source_video_path = None
        st.session_state.reference_image_path = None
        st.session_state.video_meta = None
        st.session_state.image_info = None
        st.session_state.last_output_video = None
        st.session_state.sbs_video_path = None
        st.toast("Temporary files cleaned successfully.")
        st.rerun()


# -------------------------------------------------------------
# SECTION 1: HEADER
# -------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-header">
        <span class="hero-pill">Professional AI Studio</span>
        <h1 class="hero-title">{APP_TITLE}</h1>
        <p class="hero-subtitle">{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# SECTION 2: FILE UPLOAD
# -------------------------------------------------------------
st.markdown("### 📂 Upload Media Assets")
col_upload_vid, col_upload_img = st.columns(2)

with col_upload_vid:
    st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
    st.subheader("Panel A: Source Video")
    uploaded_video = st.file_uploader(
        "Upload driving video (MP4, MOV, AVI, MKV)",
        type=[ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS],
        key="uploader_source_video",
    )
    if uploaded_video is not None:
        is_valid_vid, vid_err = validate_video_file(uploaded_video)
        if not is_valid_vid:
            st.error(vid_err)
        else:
            saved_vid_path = save_uploaded_file(uploaded_video, "videos")
            st.session_state.source_video_path = saved_vid_path
            # Analyze metadata
            st.session_state.video_meta = VideoAnalyzer.analyze(saved_vid_path)
    st.markdown("</div>", unsafe_allow_html=True)

with col_upload_img:
    st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
    st.subheader("Panel B: Reference Face Image")
    uploaded_image = st.file_uploader(
        "Upload authorized face portrait (JPG, PNG, WEBP)",
        type=[ext.lstrip(".") for ext in SUPPORTED_IMAGE_EXTENSIONS],
        key="uploader_ref_image",
    )
    if uploaded_image is not None:
        is_valid_img, img_err = validate_reference_image(uploaded_image)
        if not is_valid_img:
            st.error(img_err)
        else:
            saved_img_path = save_uploaded_file(uploaded_image, "images")
            st.session_state.reference_image_path = saved_img_path
            detector = get_cached_detector()
            _, _, img_info = validate_reference_image_path(saved_img_path, face_detector=detector)
            st.session_state.image_info = img_info
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# SECTION 3: VIDEO AND IMAGE PREVIEW
# -------------------------------------------------------------
if st.session_state.source_video_path or st.session_state.reference_image_path:
    st.markdown("### 🔍 Media Preview & Technical Inspection")
    col_prev_vid, col_prev_img = st.columns(2)

    with col_prev_vid:
        st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
        st.markdown("#### Source Video Analysis")
        if st.session_state.source_video_path and os.path.exists(st.session_state.source_video_path):
            st.video(st.session_state.source_video_path)
            meta: Optional[VideoAnalysisResult] = st.session_state.video_meta
            if meta:
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='metric-box'><div class='metric-value'>{meta.resolution_label.split(' ')[0]}</div><div class='metric-label'>Resolution</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-box'><div class='metric-value'>{meta.fps}</div><div class='metric-label'>Frame Rate</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-box'><div class='metric-value'>{meta.duration_formatted}</div><div class='metric-label'>Duration</div></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-box'><div class='metric-value'>{meta.file_size_formatted}</div><div class='metric-label'>File Size</div></div>", unsafe_allow_html=True)
        else:
            st.info("Upload a video to view playback and technical properties.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_prev_img:
        st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
        st.markdown("#### Reference Identity Inspection")
        if st.session_state.reference_image_path and os.path.exists(st.session_state.reference_image_path):
            st.image(st.session_state.reference_image_path, use_container_width=True)
            info = st.session_state.image_info
            if info:
                i1, i2, i3 = st.columns(3)
                i1.markdown(f"<div class='metric-box'><div class='metric-value'>{info['width']}x{info['height']}</div><div class='metric-label'>Dimensions</div></div>", unsafe_allow_html=True)
                i2.markdown(f"<div class='metric-box'><div class='metric-value'>{info['faces_detected']}</div><div class='metric-label'>Faces Detected</div></div>", unsafe_allow_html=True)
                i3.markdown(f"<div class='metric-box'><div class='metric-value'>{info['sharpness']:.1f}</div><div class='metric-label'>Sharpness</div></div>", unsafe_allow_html=True)

                if info.get("warnings"):
                    for warn in info["warnings"]:
                        st.warning(warn)
        else:
            st.info("Upload a reference face image to inspect portrait quality and landmarks.")
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# SECTION 4 & 5: PROCESSING SETTINGS & EXPORT CONFIGURATION
# -------------------------------------------------------------
st.markdown("### 🎛️ Studio Configuration & Output Settings")

with st.expander("🛠️ Advanced Facial Animation & Synchronization Parameters", expanded=True):
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown("##### 👤 Identity & Motion")
        opt_face_replacement = st.toggle("Face Replacement", value=True, help="Replaces the primary face with authorized reference identity.")
        opt_head_pose = st.toggle("Head Movement Synchronization", value=True, help="Tracks 3D head yaw, pitch, roll, and orientation.")
        opt_expressions = st.toggle("Facial Expression Synchronization", value=True, help="Tracks smiling, eyebrow raising, sadness, and muscle shifts.")

    with col_s2:
        st.markdown("##### 👄 Eyes & Speech")
        opt_lip_sync = st.toggle("Lip Synchronization", value=True, help="Synchronizes speech articulation, jaw motion, and mouth opening.")
        opt_teeth = st.toggle("Teeth & Tongue Preservation", value=True, help="Preserves real teeth, tongue, and oral cavity during speaking/singing instead of stretching closed lips.")
        opt_eye_sync = st.toggle("Eye Movement Synchronization", value=True, help="Mirrors pupil gaze orientation and eye aperture.")
        opt_blinking = st.toggle("Blinking Synchronization", value=True, help="Tracks and synchronizes natural blink timing.")

    with col_s3:
        st.markdown("##### 🎨 Blending & Realism")
        opt_blend_mode = st.selectbox(
            "Blending Technique",
            options=["Multi-Band Pyramid (Recommended)", "Poisson Seamless Clone", "Feathered Alpha"],
            index=0,
            help="Multi-Band Pyramid decomposes spatial frequencies for natural skin tone fusion without discoloration. Poisson provides gradient cloning.",
        )
        opt_lighting = st.toggle("Lighting & Skin Tone Transfer", value=True, help="CIE LAB color transfer and skin illumination matching.")
        opt_blending = st.toggle("Seamless Skin Integration", value=True, help="Seamlessly blends face boundaries into the neck and forehead.")
        opt_enhance_teeth = st.toggle("Teeth & Oral Clarity Sharpening", value=True, help="Sharpens dental enamel highlights and tongue texture to avoid video compression softness.")
        opt_temporal = st.toggle("Temporal Smoothing", value=True, help="Exponential moving average filter to eliminate jitter and flickering.")
        opt_audio = st.toggle("Preserve Original Audio", value=True, help="Extracts and preserves sound effects, background music, and speech.")

with st.expander("📺 Video Export & Quality Settings", expanded=True):
    col_out1, col_out2, col_out3 = st.columns(3)

    with col_out1:
        opt_resolution = st.selectbox(
            "Output Resolution",
            options=list(RESOLUTION_PRESETS.keys()),
            index=0,
            help="Select export resolution preset. Original preserves source video dimensions.",
        )

    with col_out2:
        opt_quality = st.selectbox(
            "Video Compression Quality",
            options=list(CRF_PRESETS.keys()),
            index=0,
            help="High uses CRF 18 (visually lossless), Medium uses CRF 23, Economy uses CRF 28.",
        )

    with col_out3:
        opt_watermark = st.toggle(
            "Visible AI Disclosure Badge",
            value=True,
            disabled=True,
            help="Mandatory ethical disclosure badge ('AI-MODIFIED | CONSENTED LIKENESS') is embedded.",
        )


# -------------------------------------------------------------
# SECTION 6: PROCESSING BUTTON & VALIDATION GATES
# -------------------------------------------------------------
st.markdown("---")

can_process = (
    st.session_state.source_video_path is not None
    and st.session_state.reference_image_path is not None
    and consent_agreed
)

if not consent_agreed:
    st.warning("⚠️ Please check the Consent & Ethical AI box in the sidebar before starting processing.")

btn_col1, btn_col2 = st.columns([1, 3])
with btn_col1:
    start_processing = st.button(
        "🚀 START AI PROCESSING",
        type="primary",
        disabled=not can_process,
        use_container_width=True,
    )

with btn_col2:
    if not can_process and (st.session_state.source_video_path is None or st.session_state.reference_image_path is None):
        st.info("Upload both a Source Video and an Authorized Reference Face Image to proceed.")


# -------------------------------------------------------------
# SECTION 7: PROCESSING STATUS & EXECUTION
# -------------------------------------------------------------
if start_processing:
    st.markdown("### ⚡ AI Processing Pipeline in Progress")
    progress_bar = st.progress(0, text="Initializing processing pipeline...")
    status_box = st.empty()

    m_stage, m_fps, m_time, m_eta = st.columns(4)
    with m_stage:
        metric_stage = st.empty()
    with m_fps:
        metric_fps = st.empty()
    with m_time:
        metric_time = st.empty()
    with m_eta:
        metric_eta = st.empty()

    def update_ui_progress(update: ProgressUpdate) -> None:
        pct = int(update.percentage)
        progress_bar.progress(pct, text=f"Stage: {update.stage} ({pct}%) - {update.message}")
        metric_stage.markdown(f"<div class='metric-box'><div class='metric-value'>{update.current_frame}/{update.total_frames}</div><div class='metric-label'>Frames Processed</div></div>", unsafe_allow_html=True)
        metric_fps.markdown(f"<div class='metric-box'><div class='metric-value'>{update.fps:.1f}</div><div class='metric-label'>Processing FPS</div></div>", unsafe_allow_html=True)
        metric_time.markdown(f"<div class='metric-box'><div class='metric-value'>{format_seconds(update.elapsed_seconds)}</div><div class='metric-label'>Elapsed Time</div></div>", unsafe_allow_html=True)
        metric_eta.markdown(f"<div class='metric-box'><div class='metric-value'>{format_seconds(update.eta_seconds)}</div><div class='metric-label'>Estimated ETA</div></div>", unsafe_allow_html=True)

    job_settings = {
        "face_replacement": opt_face_replacement,
        "lip_sync": opt_lip_sync,
        "preserve_mouth_interior": opt_teeth,
        "enhance_teeth": opt_enhance_teeth,
        "eye_sync": opt_eye_sync,
        "blinking_sync": opt_blinking,
        "expression_sync": opt_expressions,
        "head_movement_sync": opt_head_pose,
        "lighting_correction": opt_lighting,
        "seamless_blending": opt_blending,
        "blend_method": "multiband" if "Multi-Band" in opt_blend_mode else ("poisson" if "Poisson" in opt_blend_mode else "feathered"),
        "temporal_smoothing": opt_temporal,
        "preserve_audio": opt_audio,
        "output_resolution": opt_resolution,
        "video_quality": opt_quality,
        "ai_modified_label": True,
    }

    pipeline = StudioProcessingPipeline(
        settings=job_settings,
        progress_callback=update_ui_progress,
    )

    with st.spinner("Executing identity-conditioned facial animation pipeline..."):
        result = pipeline.execute(
            source_video_path=st.session_state.source_video_path,
            reference_image_path=st.session_state.reference_image_path,
        )

    if result.success and result.output_video_path and os.path.exists(result.output_video_path):
        st.session_state.last_output_video = result.output_video_path
        # Automatically generate synchronous side-by-side comparison video in ONE frame
        sbs_target = os.path.join(
            os.path.dirname(result.output_video_path),
            f"sbs_{Path(result.output_video_path).name}",
        )
        sbs_file = VideoExporter.create_side_by_side_comparison(
            original_video_path=st.session_state.source_video_path,
            modified_video_path=result.output_video_path,
            output_comparison_path=sbs_target,
        )
        st.session_state.sbs_video_path = sbs_file if (sbs_file and os.path.exists(sbs_file)) else None
        st.success("🎉 Video generation completed successfully!")
    else:
        st.error(f"❌ Processing failed: {result.error_message}")


# -------------------------------------------------------------
# SECTION 8: FINAL RESULT & SYNCHRONOUS COMPARISON
# -------------------------------------------------------------
if st.session_state.last_output_video and os.path.exists(st.session_state.last_output_video):
    st.markdown("---")
    st.markdown("### 🎬 Studio Results & Real-Time Comparison")

    tab_sbs, tab_single, tab_inspect = st.tabs([
        "📺 Synchronous Dual View (Both in One Frame)",
        "🎯 Full Resolution AI Video",
        "🔍 Frame-by-Frame Face Inspector",
    ])

    with tab_sbs:
        st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
        st.markdown("#### 📺 Side-by-Side Real-Time Comparison (100% Synchronous)")
        st.caption("Both the Original Driving Actor (Left) and AI-Modified Likeness (Right) play synchronously in a single frame.")

        sbs_file = getattr(st.session_state, "sbs_video_path", None)
        if not sbs_file or not os.path.exists(sbs_file):
            sbs_target = os.path.join(
                os.path.dirname(st.session_state.last_output_video),
                f"sbs_{Path(st.session_state.last_output_video).name}",
            )
            sbs_file = VideoExporter.create_side_by_side_comparison(
                original_video_path=st.session_state.source_video_path,
                modified_video_path=st.session_state.last_output_video,
                output_comparison_path=sbs_target,
            )
            st.session_state.sbs_video_path = sbs_file

        if sbs_file and os.path.exists(sbs_file):
            st.video(sbs_file)
            with open(sbs_file, "rb") as f_sbs:
                st.download_button(
                    label="⬇️ Download Side-by-Side Comparison Video (MP4)",
                    data=f_sbs.read(),
                    file_name=Path(sbs_file).name,
                    mime="video/mp4",
                    use_container_width=True,
                )
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original Actor**")
                st.video(st.session_state.source_video_path)
            with c2:
                st.markdown("**AI Modified**")
                st.video(st.session_state.last_output_video)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_single:
        st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Full Resolution AI-Modified Video")
        st.video(st.session_state.last_output_video)
        with open(st.session_state.last_output_video, "rb") as video_file:
            st.download_button(
                label="⬇️ Download Full-Res Video (MP4)",
                data=video_file.read(),
                file_name=Path(st.session_state.last_output_video).name,
                mime="video/mp4",
                use_container_width=True,
                type="primary",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_inspect:
        st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Frame-by-Frame Close-Up Face Inspector")
        st.caption("Scrub through any video frame to inspect high-definition face crops, lip sync, and teeth detail side-by-side.")

        cap_orig = cv2.VideoCapture(st.session_state.source_video_path)
        cap_mod = cv2.VideoCapture(st.session_state.last_output_video)
        total_f = int(min(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT) or 1, cap_mod.get(cv2.CAP_PROP_FRAME_COUNT) or 1))

        if total_f > 1:
            frame_num = st.slider("Select Video Frame to Inspect", min_value=0, max_value=total_f - 1, value=min(50, total_f - 1))
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            cap_mod.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret_o, f_o = cap_orig.read()
            ret_m, f_m = cap_mod.read()

            if ret_o and ret_m:
                detector = get_cached_detector()
                best_face = detector.detect_best_face(f_o)
                if best_face is not None:
                    bx1, by1, bx2, by2 = [int(v) for v in best_face["bbox"]]
                    pad = int(max(bx2 - bx1, by2 - by1) * 0.15)
                    y1 = max(0, by1 - pad)
                    y2 = min(f_o.shape[0], by2 + pad)
                    x1 = max(0, bx1 - pad)
                    x2 = min(f_o.shape[1], bx2 + pad)
                    crop_orig = f_o[y1:y2, x1:x2]
                    crop_mod = f_m[y1:y2, x1:x2]
                else:
                    crop_orig = f_o
                    crop_mod = f_m

                col_io1, col_io2 = st.columns(2)
                with col_io1:
                    st.image(cv2.cvtColor(crop_orig, cv2.COLOR_BGR2RGB), caption=f"Original Face (Frame {frame_num})", use_container_width=True)
                with col_io2:
                    st.image(cv2.cvtColor(crop_mod, cv2.COLOR_BGR2RGB), caption=f"AI-Modified Face (Frame {frame_num})", use_container_width=True)

        cap_orig.release()
        cap_mod.release()
        st.markdown("</div>", unsafe_allow_html=True)
