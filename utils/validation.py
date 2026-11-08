from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import cv2

from config import MAX_IMAGE_SIZE_BYTES, MAX_VIDEO_SIZE_BYTES, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS


def allowed_extension(filename: str, allowed: Iterable[str]) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in {ext.lower() for ext in allowed}


def validate_file_type(filename: str, allowed: Iterable[str]) -> bool:
    return allowed_extension(filename, allowed)


def validate_file_size(file_size_bytes: int, max_bytes: int) -> bool:
    return file_size_bytes > 0 and file_size_bytes <= max_bytes


def validate_video_file(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "Please upload a source video."
    filename = getattr(uploaded_file, "name", "") or ""
    if not validate_file_type(filename, SUPPORTED_VIDEO_EXTENSIONS):
        return False, "Unsupported video format. Use MP4, MOV, or AVI."
    if not validate_file_size(getattr(uploaded_file, "size", 0) or 0, MAX_VIDEO_SIZE_BYTES):
        return False, f"Video is too large. Maximum size is {MAX_VIDEO_SIZE_BYTES / (1024 * 1024):.0f} MB."

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        content = uploaded_file.read(64)
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        if not content:
            return False, "The uploaded video is empty or unreadable."
    except Exception:
        return False, "Unable to read the uploaded video file."

    return True, "Valid video file."


def validate_video_file_path(video_path: str) -> Tuple[bool, str]:
    if not video_path or not os.path.exists(video_path):
        return False, "The uploaded video file could not be found on disk."

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return False, "The uploaded file is not a valid video, or it may be corrupted. Please upload a playable MP4, MOV, or AVI file."

    success, _ = capture.read()
    capture.release()
    if not success:
        return False, "The uploaded video could not be decoded. It may be corrupted, incomplete, or not a real video file."

    return True, "Valid video file."


def validate_reference_image(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "Please upload a reference face image."
    filename = getattr(uploaded_file, "name", "") or ""
    if not validate_file_type(filename, SUPPORTED_IMAGE_EXTENSIONS):
        return False, "Unsupported image format. Use JPG, JPEG, or PNG."
    if not validate_file_size(getattr(uploaded_file, "size", 0) or 0, MAX_IMAGE_SIZE_BYTES):
        return False, f"Image is too large. Maximum size is {MAX_IMAGE_SIZE_BYTES / (1024 * 1024):.0f} MB."
    return True, "Valid reference image."
