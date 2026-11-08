from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import APP_DESCRIPTION, APP_TITLE, OUTPUTS_DIR, TEMP_DIR, VIDEO_PREVIEW_WIDTH
from modules.audio_processor import extract_audio, merge_audio
from modules.face_detector import FaceDetector
from modules.face_replacement import FaceReplacementPipeline
from modules.face_tracker import FaceTracker
from modules.facial_animation import FacialAnimationPipeline
from modules.video_processor import load_video_metadata, process_video
from utils.file_handler import cleanup_directory, save_uploaded_file
from utils.helpers import ensure_directory, format_file_size
from utils.validation import validate_reference_image, validate_video_file, validate_video_file_path


st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #070b14;
            --bg-2: #0d1524;
            --panel: rgba(15, 23, 42, 0.9);
            --panel-soft: rgba(17, 24, 39, 0.8);
            --panel-strong: rgba(15, 118, 110, 0.15);
            --primary: #8b5cf6;
            --secondary: #22d3ee;
            --text: #edf2ff;
            --muted: #a4b0c3;
            --line: rgba(148, 163, 184, 0.18);
            --success: #34d399;
            --danger: #f87171;
            --shadow: 0 25px 60px rgba(2, 8, 23, 0.55);
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: radial-gradient(circle at top left, rgba(139, 92, 246, 0.26), transparent 32%),
                        radial-gradient(circle at top right, rgba(34, 211, 238, 0.18), transparent 24%),
                        linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(30, 41, 59, 0.8));
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.25rem 1.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
        }

        .eyebrow {
            display: inline-block;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--secondary);
            font-weight: 700;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(34, 211, 238, 0.12);
            border: 1px solid rgba(34, 211, 238, 0.18);
            margin-bottom: 0.75rem;
        }

        h1 {
            font-size: clamp(2rem, 3vw, 3rem) !important;
            margin: 0 !important;
            line-height: 1.1 !important;
        }

        .subtitle {
            margin-top: 0.6rem;
            color: var(--muted);
            font-size: 1rem;
        }

        .upload-card,
        .uploaded-box,
        .card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.8));
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }

        div[data-testid="stFileUploader"] section {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(139, 92, 246, 0.35);
            border-radius: 16px;
            padding: 0.35rem;
        }

        div[data-testid="stFileUploader"] > section > div > div {
            background: rgba(15, 23, 42, 0.35);
            border-radius: 12px;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--primary), #7c3aed);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.7rem 1.3rem;
            min-width: 180px;
            box-shadow: 0 12px 30px rgba(124, 58, 237, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.06);
            box-shadow: 0 15px 35px rgba(124, 58, 237, 0.45);
        }

        .stButton > button:disabled {
            background: rgba(148, 163, 184, 0.2);
            color: rgba(226, 232, 240, 0.55);
            box-shadow: none;
            opacity: 0.75;
        }

        .preview-shell {
            background: rgba(2, 6, 23, 0.45);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.5rem;
            overflow: hidden;
        }

        .small-text {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .checkbox {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }

        .stProgress > div > div {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }

        .stVideo, .stImage {
            border-radius: 16px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def show_video_metadata(video_path: str) -> None:
    try:
        metadata = load_video_metadata(video_path)
        st.caption(f"Duration: {metadata.duration:.2f}s • Resolution: {metadata.width}x{metadata.height} • FPS: {metadata.fps:.2f}")
    except Exception:
        st.caption("Unable to read video metadata.")


def preview_image(file_path: str, width: int = 480) -> None:
    image = Image.open(file_path)
    st.image(image, width=width, caption="Reference face preview")


def validate_user_consent() -> bool:
    consent = st.checkbox("I confirm that I have permission to use the reference person's likeness and that this use is lawful, consent-based, and not deceptive.", value=False)
    return consent


def resolve_output_size(settings: dict | None) -> tuple[int, int] | None:
    if settings is None:
        return None

    resolution = settings.get("output_resolution", "Original")
    if resolution == "720p":
        return (1280, 720)
    if resolution == "1080p":
        return (1920, 1080)
    return None


def run_processing(video_path: str, reference_path: str, progress_container, settings: dict | None = None) -> str:
    temp_dir = Path(TEMP_DIR) / f"job_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = str(Path(OUTPUTS_DIR) / f"processed_{uuid.uuid4().hex[:8]}.mp4")
    audio_path = str(temp_dir / "audio.aac")
    output_size = resolve_output_size(settings)

    try:
        audio_ready_path = extract_audio(video_path, str(audio_path))
        reference_image = cv2.imread(reference_path)
        if reference_image is None:
            raise ValueError("Unable to read the reference image. Please upload a valid JPG, JPEG, or PNG file.")

        detector = FaceDetector()
        pipeline = FacialAnimationPipeline(detector, settings=settings or {})
        pipeline.load_reference(reference_image)

        progress_bar = progress_container.progress(0.0)

        def update_progress(value: float) -> None:
            progress_bar.progress(value)

        process_video(video_path, pipeline, output_video_path, progress_callback=update_progress, target_size=output_size)

        final_output_path = str(Path(OUTPUTS_DIR) / f"final_{uuid.uuid4().hex[:8]}.mp4")
        merged_path = merge_audio(output_video_path, audio_ready_path, final_output_path)
        if merged_path and os.path.exists(merged_path):
            if merged_path != output_video_path:
                os.replace(merged_path, output_video_path)

        progress_bar.progress(1.0)
        return output_video_path
    finally:
        cleanup_directory(str(temp_dir))


def main() -> None:
    st.markdown("""
    <div class="hero-card">
        <div class="eyebrow">Consent-based AI editing</div>
        <h1>AI Face Replacement Studio</h1>
        <div class="subtitle">Upload a source video and a reference face to generate a seamless, lawful face replacement preview.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Advanced facial controls")
        settings = {
            "face_replacement": st.checkbox("Face Replacement", value=True),
            "lip_sync": st.checkbox("Lip Synchronization", value=True),
            "eye_sync": st.checkbox("Eye Synchronization", value=True),
            "expression_sync": st.checkbox("Facial Expression Synchronization", value=True),
            "head_movement_sync": st.checkbox("Head Movement Synchronization", value=True),
            "lighting_correction": st.checkbox("Lighting Correction", value=True),
            "ai_modified_label": st.checkbox("Label output as AI-modified", value=True),
            "output_resolution": st.selectbox("Output Resolution", ["Original", "720p", "1080p"], index=0),
        }

    upload_cols = st.columns(2)
    with upload_cols[0]:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("1. Upload source video", type=["mp4", "mov", "avi"], help="MP4, MOV, or AVI only.")
        st.markdown('</div>', unsafe_allow_html=True)

    with upload_cols[1]:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        reference_image = st.file_uploader("2. Upload reference face image", type=["jpg", "jpeg", "png"], help="JPG, JPEG, or PNG only.")
        st.markdown('</div>', unsafe_allow_html=True)

    video_valid = True
    image_valid = True
    message = ""

    if uploaded_video is not None:
        video_valid, message = validate_video_file(uploaded_video)
        if not video_valid:
            st.error(message)
    if reference_image is not None:
        image_valid, message = validate_reference_image(reference_image)
        if not image_valid:
            st.error(message)

    uploaded_video_path = None
    uploaded_image_path = None

    preview_cols = st.columns(2)
    if uploaded_video is not None and uploaded_video.name:
        with preview_cols[0]:
            uploaded_video_path = save_uploaded_file(uploaded_video, subdirectory="videos")
            st.markdown("<div class='uploaded-box preview-shell'>", unsafe_allow_html=True)
            st.video(uploaded_video_path, format="video/mp4", start_time=0)
            show_video_metadata(uploaded_video_path)
            st.markdown("</div>", unsafe_allow_html=True)

    if reference_image is not None and reference_image.name:
        with preview_cols[1]:
            uploaded_image_path = save_uploaded_file(reference_image, subdirectory="images")
            st.markdown("<div class='uploaded-box preview-shell'>", unsafe_allow_html=True)
            preview_image(uploaded_image_path, width=VIDEO_PREVIEW_WIDTH)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='checkbox'>", unsafe_allow_html=True)
    consent = validate_user_consent()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    process_button = st.button("Start Processing", disabled=not (video_valid and image_valid and consent and uploaded_video is not None and reference_image is not None))

    if process_button:
        if not consent:
            st.warning("Please confirm consent before processing.")
            return

        if uploaded_video is None or reference_image is None:
            st.warning("Please upload both a video and a reference face image before processing.")
            return

        video_path = uploaded_video_path or save_uploaded_file(uploaded_video, subdirectory="videos")
        image_path = uploaded_image_path or save_uploaded_file(reference_image, subdirectory="images")

        if not os.path.exists(video_path):
            st.error("The uploaded video file was not saved correctly.")
            return
        if not os.path.exists(image_path):
            st.error("The uploaded reference image was not saved correctly.")
            return

        video_ok, video_error = validate_video_file_path(video_path)
        if not video_ok:
            st.error(video_error)
            return

        with st.spinner("Analyzing faces and processing the video. This may take a moment..."):
            progress = st.empty()
            try:
                processed_video = run_processing(video_path, image_path, progress, settings=settings)
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.info("Make sure FFmpeg is installed and that its bin folder is added to PATH. On Windows, install it with: winget install Gyan.Dev.FFmpeg")
                return

        st.success("Processing complete.")

        st.subheader("Final Output")
        st.video(processed_video)

        with st.expander("Before and after comparison"):
            try:
                before = cv2.imread(image_path)
                if before is not None:
                    before_rgb = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
                    st.image(before_rgb, caption="Reference face", width=240)
            except Exception:
                pass

        with open(processed_video, "rb") as video_file:
            st.download_button(
                label="Download Video",
                data=video_file,
                file_name="ai_face_replacement_output.mp4",
                mime="video/mp4",
            )


if __name__ == "__main__":
    main()
