from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path so cartoon_studio can be imported cleanly
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st

from cartoon_studio.config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_DESCRIPTION,
    CARTOON_STYLES,
    RESOLUTION_PRESETS,
    CRF_PRESETS,
    ASSETS_DIR,
    OUTPUTS_DIR,
    TEMP_DIR,
)
from cartoon_studio.utils.helpers import get_device_info, format_duration, format_file_size
from cartoon_studio.utils.file_handler import FileHandler
from cartoon_studio.utils.validation import validate_video_file
from cartoon_studio.utils.progress_manager import ProgressUpdate
from cartoon_studio.modules.style_manager import StyleManager
from cartoon_studio.modules.processing_pipeline import CartoonStudioPipeline, PipelineResult

# Streamlit Page Setup - Luxurious Obsidian Mirror Glassmorphism
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom Glassmorphism & Liquid Chrome CSS
css_file = ASSETS_DIR / "css" / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Hardware Detection
hw_info = get_device_info()
device_label = hw_info["device_name"]
vram_text = f" ({hw_info['vram_total_gb']} GB VRAM)" if hw_info["cuda_available"] else ""

# --------------------------------------------------------------------------
# SIDEBAR: Obsidian Chrome Dock (System Telemetry & Controls)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-dock-header">⚡ QUANTUM DOCK</div>', unsafe_allow_html=True)
    st.caption("Studio Hardware & Telemetry Monitor")
    
    st.markdown(
        f"""
        <div style="background: rgba(13, 17, 30, 0.75); border: 1px solid rgba(220, 229, 242, 0.12); border-radius: 12px; padding: 14px; margin-bottom: 20px;">
            <div class="sidebar-telemetry-item">
                <span class="sidebar-telemetry-lbl">Accelerator:</span>
                <span class="sidebar-telemetry-val" style="color: #22D3EE;">{device_label}</span>
            </div>
            <div class="sidebar-telemetry-item">
                <span class="sidebar-telemetry-lbl">VRAM Status:</span>
                <span class="sidebar-telemetry-val">{hw_info['vram_total_gb']} GB Total</span>
            </div>
            <div class="sidebar-telemetry-item">
                <span class="sidebar-telemetry-lbl">PyTorch Core:</span>
                <span class="sidebar-telemetry-val">v{hw_info['torch_version']}</span>
            </div>
            <div class="sidebar-telemetry-item">
                <span class="sidebar-telemetry-lbl">Neural Engine:</span>
                <span class="sidebar-telemetry-val" style="color: #00ff88;">READY (15-30 FPS)</span>
            </div>
            <div class="sidebar-telemetry-item" style="border-bottom: none;">
                <span class="sidebar-telemetry-lbl">Anti-Flicker:</span>
                <span class="sidebar-telemetry-val" style="color: #8B5CF6;">Farneback Flow</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-dock-header" style="font-size: 0.95rem;">🎨 2D STYLE PALETTE</div>', unsafe_allow_html=True)
    for s_name, s_data in CARTOON_STYLES.items():
        st.markdown(
            f"""
            <div style="padding: 6px 10px; border-left: 2px solid #22D3EE; background: rgba(34, 211, 238, 0.04); margin-bottom: 6px; border-radius: 4px; font-size: 0.8rem;">
                <strong style="color: #F1F5F9;">{s_name}</strong><br>
                <span style="color: #94A3B8; font-size: 0.74rem;">{s_data['description'][:65]}...</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color: rgba(220, 229, 242, 0.1); margin: 20px 0;'>", unsafe_allow_html=True)
    if st.button("🧹 Clear Studio Cache & Temp", use_container_width=True):
        FileHandler.cleanup_directory(TEMP_DIR, ignore_errors=True)
        st.session_state.pipeline_result = None
        st.success("Studio temporary cache cleared!")

# App State Initialization
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Layout: Split into Command Console & Live Telemetry Stage
col_left, col_right = st.columns([1, 1.25], gap="large")

with col_left:
    # --------------------------------------------------------------------------
    # Futuristic Mirror Header (Left-Side Only)
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
            <div class="hw-badge">
                <span class="hw-dot"></span>
                Liquid Chrome: <strong>{device_label}{vram_text}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------
    # Glass Panel 1: Media Ingestion
    # ----------------------------------------------------------------------
    st.markdown(
        """
        <div class="glass-panel">
            <div class="panel-header">📹 1. Media Ingestion Deck</div>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader(
        "Upload Video (MP4, MOV, AVI, MKV, WebM):",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Upload a video with people, actions, or scenery to convert into 2D cartoon animation."
    )

    if uploaded_file is not None:
        if st.session_state.uploaded_video_path is None or getattr(st.session_state, "last_uploaded_name", "") != uploaded_file.name:
            saved_path = FileHandler.save_uploaded_file(uploaded_file)
            st.session_state.uploaded_video_path = saved_path
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.pipeline_result = None

    if st.session_state.uploaded_video_path:
        video_path = Path(st.session_state.uploaded_video_path)
        val = validate_video_file(video_path)
        
        if val.is_valid:
            st.video(str(video_path))
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div class='metric-chip'><div class='metric-chip-val'>{val.width}x{val.height}</div><div class='metric-chip-lbl'>Dimension</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-chip'><div class='metric-chip-val'>{val.fps:.1f}</div><div class='metric-chip-lbl'>FPS</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-chip'><div class='metric-chip-val'>{format_duration(val.duration_seconds)}</div><div class='metric-chip-lbl'>Duration</div></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-chip'><div class='metric-chip-val'>{format_file_size(val.file_size_bytes)}</div><div class='metric-chip-lbl'>Size</div></div>", unsafe_allow_html=True)
        else:
            st.error(f"Invalid video file: {val.error_message}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # Glass Panel 2: Style Synthesizer
    # ----------------------------------------------------------------------
    st.markdown(
        """
        <div class="glass-panel">
            <div class="panel-header">🎨 2. Neural Style Synthesizer</div>
        """,
        unsafe_allow_html=True
    )
    
    style_names = StyleManager.get_style_names()
    selected_style = st.selectbox("Select 2D Cartoon Style:", style_names, index=0)
    
    profile = StyleManager.get_profile(selected_style)
    st.markdown(
        f"""
        <div class="style-glass-preview">
            <strong>{profile.name} Aesthetic:</strong> {profile.description}<br>
            <span style="color: #94A3B8; font-size: 0.78rem;"><em>Prompt Signature:</em> "{profile.prompt}"</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_int, col_lw = st.columns(2)
    with col_int:
        animation_intensity = st.slider(
            "Animation Intensity:",
            min_value=0.10,
            max_value=1.00,
            value=0.85,
            step=0.05,
            help="Higher intensity generates deeper 2D cel-shaded stylization; lower retains subtle realism."
        )
    with col_lw:
        line_weight = st.slider(
            "Line Art Weight:",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Multiplier for ink outline contours and cel-edges."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # Glass Panel 3: Quantum Engine & Master Output
    # ----------------------------------------------------------------------
    st.markdown(
        """
        <div class="glass-panel">
            <div class="panel-header">⚙️ 3. Quantum Engine & Master Settings</div>
        """,
        unsafe_allow_html=True
    )
    
    engine_options = [
        "High-Performance Neural Engine (15-30 FPS, Zero OOM)",
        "AnimateDiff Video-to-Video Engine (Diffusers + ControlNet)"
    ]
    selected_engine = st.selectbox("Animation Engine:", engine_options, index=0)

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        selected_res = st.selectbox(
            "Target Resolution:",
            list(RESOLUTION_PRESETS.keys()),
            index=0,
            help="Optionally upscale to 720p, 1080p Full HD, or 4K Ultra HD."
        )
    with opt_col2:
        selected_quality = st.selectbox(
            "Encoding Quality:",
            list(CRF_PRESETS.keys()),
            index=0,
            help="H.264 Constant Rate Factor (CRF 18 provides pristine visual fidelity)."
        )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        temporal_smoothing = st.checkbox("Temporal Anti-Flicker", value=True, help="Eliminate cross-frame jitter via dense optical flow motion compensation.")
        preserve_pose = st.checkbox("Preserve Body Gestures & Pose", value=True, help="Guide animation with OpenPose skeleton motion tracking.")
    with col_t2:
        preserve_depth = st.checkbox("Preserve Scene Depth", value=True, help="Maintain original 3D perspective and camera angles.")
        preserve_audio = st.checkbox("Preserve Audio Track", value=True, help="Losslessly synchronize and preserve original audio.")

    st.markdown("<br>", unsafe_allow_html=True)
    start_btn = st.button(
        "✨ INITIATE 2D TRANSFORMATION",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.uploaded_video_path is None or st.session_state.is_processing
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # ----------------------------------------------------------------------
    # Glass Panel 4: Telemetry Stage & Live Processing
    # ----------------------------------------------------------------------
    st.markdown(
        """
        <div class="glass-panel">
            <div class="panel-header">🎬 4. Studio Telemetry & Live Monitor</div>
        """,
        unsafe_allow_html=True
    )

    progress_container = st.container()

    if start_btn and st.session_state.uploaded_video_path:
        st.session_state.is_processing = True
        st.session_state.pipeline_result = None

        with progress_container:
            status_box = st.empty()
            pbar = st.progress(0.0)
            metric_cols = st.columns(4)
            m_stage = metric_cols[0].empty()
            m_frames = metric_cols[1].empty()
            m_fps = metric_cols[2].empty()
            m_eta = metric_cols[3].empty()
            
            def progress_callback(update: ProgressUpdate):
                prog_val = min(1.0, max(0.0, float(getattr(update, "overall_progress", 0.0))))
                pct = int(round(prog_val * 100))
                stage_num = getattr(update, "stage_step", 1)
                total_stages = getattr(update, "total_stages", 10)
                stage_name = getattr(update, "stage_name", "Transforming")
                cur_frame = getattr(update, "current_frame", 0)
                tot_frames = getattr(update, "total_frames", 1)
                fps_val = getattr(update, "fps", 0.0)
                eta_val = getattr(update, "eta_seconds", 0.0)
                msg = getattr(update, "status_message", "")

                # 1. Prominent Animated Liquid Chrome Progress Bar
                pbar.progress(prog_val, text=f"✨ {pct}% Complete — {msg}")

                # 2. Glowing Mirror Status Badge
                status_box.markdown(
                    f"""
                    <div style="background: rgba(139, 92, 246, 0.08); border-left: 4px solid #22D3EE; border-top: 1px solid rgba(220, 229, 242, 0.25); border-right: 1px solid rgba(220, 229, 242, 0.1); border-bottom: 1px solid rgba(220, 229, 242, 0.1); padding: 14px 18px; border-radius: 10px; margin-bottom: 14px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: #22D3EE; font-size: 1.05rem;">Stage {stage_num}/{total_stages}: {stage_name}</span>
                            <span style="font-weight: 800; color: #FFFFFF; font-size: 1.25rem; font-family: 'JetBrains Mono', monospace;">{pct}%</span>
                        </div>
                        <div style="color: #F1F5F9; font-size: 0.88rem; margin-top: 4px;">{msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # 3. Live Metric Chips
                m_stage.markdown(f"<div class='metric-chip'><div class='metric-chip-val'>{stage_num}/{total_stages}</div><div class='metric-chip-lbl'>Stage</div></div>", unsafe_allow_html=True)
                m_frames.markdown(f"<div class='metric-chip'><div class='metric-chip-val'>{cur_frame}/{tot_frames}</div><div class='metric-chip-lbl'>Frames</div></div>", unsafe_allow_html=True)
                m_fps.markdown(f"<div class='metric-chip'><div class='metric-chip-val'>{fps_val:.1f}</div><div class='metric-chip-lbl'>Speed (FPS)</div></div>", unsafe_allow_html=True)
                m_eta.markdown(f"<div class='metric-chip'><div class='metric-chip-val'>{format_duration(eta_val) if eta_val > 0 else '00:00'}</div><div class='metric-chip-lbl'>ETA</div></div>", unsafe_allow_html=True)

            pipeline = CartoonStudioPipeline()
            result: PipelineResult = pipeline.process(
                video_path=st.session_state.uploaded_video_path,
                style_name=selected_style,
                engine_name=selected_engine,
                animation_intensity=animation_intensity,
                line_weight=line_weight,
                temporal_smoothing=temporal_smoothing,
                preserve_pose=preserve_pose,
                preserve_depth=preserve_depth,
                preserve_audio=preserve_audio,
                output_resolution=selected_res,
                quality_preset=selected_quality,
                progress_callback=progress_callback,
            )

            st.session_state.is_processing = False
            st.session_state.pipeline_result = result
            
            # Keep completed 100% progress visible
            pbar.progress(1.0, text="✅ 100% Complete — All Processing Finished!")

    # Display Results if available
    res = st.session_state.pipeline_result
    if res is not None:
        video_out = getattr(res, "cartoon_video_path", None)
        comp_out = getattr(res, "comparison_video_path", None)
        is_success = getattr(res, "success", False)
        
        if is_success and video_out and video_out.exists():
            st.markdown(
                f"""
                <div style="background: linear-gradient(90deg, rgba(34, 211, 238, 0.15) 0%, rgba(139, 92, 246, 0.18) 100%); border: 1px solid rgba(34, 211, 238, 0.4); border-top: 1px solid rgba(255, 255, 255, 0.45); border-radius: 12px; padding: 16px 20px; margin-top: 16px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);">
                    <div style="display: flex; align-items: center; gap: 14px;">
                        <span style="font-size: 2rem;">✨</span>
                        <div>
                            <strong style="font-size: 1.15rem; color: #FFFFFF;">2D Cartoon Transformation Finished Successfully!</strong><br>
                            <span style="color: #DCE5F2; font-size: 0.9rem;">Transformed {res.stats.get('frames_processed', 0)} frames in {res.stats.get('elapsed_seconds', 0)}s ({res.stats.get('avg_processing_fps', 0)} FPS) | Output: {res.stats.get('output_resolution', 'N/A')}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Prominent Liquid Chrome Download Section
            st.markdown('<div class="panel-header" style="margin-top: 18px;">📥 5. Studio Output & Downloads</div>', unsafe_allow_html=True)
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                with open(video_out, "rb") as f:
                    st.download_button(
                        label="⬇️ DOWNLOAD 2D CARTOON (.MP4)",
                        data=f.read(),
                        file_name=video_out.name,
                        mime="video/mp4",
                        key="btn_dl_main_cartoon",
                        type="primary",
                        use_container_width=True
                    )
                st.caption(f"💾 File: `{video_out.name}` ({format_file_size(video_out.stat().st_size)})")

            with dl_col2:
                if comp_out and comp_out.exists():
                    with open(comp_out, "rb") as f:
                        st.download_button(
                            label="⬇️ DOWNLOAD SIDE-BY-SIDE (.MP4)",
                            data=f.read(),
                            file_name=comp_out.name,
                            mime="video/mp4",
                            key="btn_dl_main_comp",
                            use_container_width=True
                        )
                    st.caption(f"💾 File: `{comp_out.name}` ({format_file_size(comp_out.stat().st_size)})")

            st.markdown(f"<div style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 16px;'>📍 <strong>Saved path on disk:</strong> <code style='color: #22D3EE;'>{video_out}</code></div>", unsafe_allow_html=True)

            st.markdown('<div class="panel-header" style="margin-top: 22px;">📺 6. Master Studio Preview Deck</div>', unsafe_allow_html=True)
            tab_compare, tab_cartoon, tab_original = st.tabs([
                "📺 Synchronous Dual-View (Single Player)",
                "🎨 2D Cartoon Master Video",
                "📹 Original Source Video"
            ])

            with tab_compare:
                if comp_out and comp_out.exists():
                    st.caption("Synchronized side-by-side single-player comparison (Original vs 2D Cartoon Animation):")
                    st.video(str(comp_out))
                else:
                    st.info("Comparison video not generated.")

            with tab_cartoon:
                st.video(str(video_out))

            with tab_original:
                if st.session_state.uploaded_video_path:
                    st.video(str(st.session_state.uploaded_video_path))

            logs_list = getattr(res, "logs", [])
            if logs_list:
                with st.expander("📜 Full Studio Processing Log Stream", expanded=False):
                    st.code("\n".join(logs_list), language="bash")

        else:
            err_msg = getattr(res, "error_message", "Unknown error")
            st.error(f"❌ Processing failed: {err_msg}")
            logs_list = getattr(res, "logs", [])
            if logs_list:
                with st.expander("📜 Processing Error Logs", expanded=True):
                    st.code("\n".join(logs_list), language="bash")
    elif not st.session_state.is_processing:
        st.info("Upload a video on the left and click **INITIATE 2D TRANSFORMATION** to begin.")
        
        # Immediate showcase preview if converted video already exists
        showcase_cartoon = OUTPUTS_DIR / "cartoon_animation_showcase.mp4"
        showcase_comp = OUTPUTS_DIR / "side_by_side_comparison_showcase.mp4"
        if showcase_cartoon.exists():
            with st.expander("🎬 View Converted Showcase Video & Downloads", expanded=True):
                st.caption(f"Ready converted 2D cartoon video (`{showcase_cartoon.name}`, {format_file_size(showcase_cartoon.stat().st_size)}):")
                st.video(str(showcase_cartoon))
                
                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    with open(showcase_cartoon, "rb") as f:
                        st.download_button(
                            label="⬇️ DOWNLOAD 2D CARTOON SHOWCASE (.MP4)",
                            data=f.read(),
                            file_name=showcase_cartoon.name,
                            mime="video/mp4",
                            key="dl_showcase_cartoon",
                            type="primary",
                            use_container_width=True,
                        )
                with sc_col2:
                    if showcase_comp.exists():
                        with open(showcase_comp, "rb") as f:
                            st.download_button(
                                label="⬇️ DOWNLOAD COMPARISON SHOWCASE (.MP4)",
                                data=f.read(),
                                file_name=showcase_comp.name,
                                mime="video/mp4",
                                key="dl_showcase_comp",
                                use_container_width=True,
                            )
                st.markdown(f"<small style='color: #8b949e;'>📍 Location: <code>{showcase_cartoon}</code></small>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
