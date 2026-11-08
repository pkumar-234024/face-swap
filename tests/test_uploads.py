from __future__ import annotations

import io
from pathlib import Path

import pytest

from utils.file_handler import cleanup_file, sanitize_filename, save_uploaded_file
from utils.validation import allowed_extension, validate_file_size, validate_reference_image, validate_video_file


class DummyUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self.size = len(data)
        self._buf = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def seek(self, pos: int) -> int:
        return self._buf.seek(pos)

    def getvalue(self) -> bytes:
        return self._data


def test_allowed_extensions():
    assert allowed_extension("clip.mp4", {".mp4", ".mov"}) is True
    assert allowed_extension("clip.MOV", {".mp4", ".mov"}) is True
    assert allowed_extension("bad.exe", {".mp4", ".mov"}) is False
    assert allowed_extension("", {".mp4"}) is False


def test_sanitize_filename():
    assert sanitize_filename("safe_video.mp4") == "safe_video.mp4"
    assert ".." not in sanitize_filename("../../../etc/passwd.mp4")
    assert sanitize_filename("spaced video file (1).mp4") == "spaced_video_file__1_.mp4"


def test_validate_video_upload():
    valid_file = DummyUploadedFile("video.mp4", b"fake_header_data_bytes_for_testing" * 4)
    is_valid, msg = validate_video_file(valid_file)
    assert is_valid is True

    bad_ext_file = DummyUploadedFile("video.pdf", b"pdf_data")
    is_valid, msg = validate_video_file(bad_ext_file)
    assert is_valid is False
    assert "Unsupported video format" in msg

    empty_file = DummyUploadedFile("video.mp4", b"")
    is_valid, msg = validate_video_file(empty_file)
    assert is_valid is False


def test_validate_image_upload():
    valid_img = DummyUploadedFile("portrait.jpg", b"fake_image_bytes" * 5)
    is_valid, msg = validate_reference_image(valid_img)
    assert is_valid is True

    bad_img = DummyUploadedFile("script.py", b"print('hi')")
    is_valid, msg = validate_reference_image(bad_img)
    assert is_valid is False


def test_save_and_cleanup_file(tmp_path):
    dummy = DummyUploadedFile("test_file.png", b"123456789")
    saved_path = save_uploaded_file(dummy, subdirectory="test_sub")
    assert Path(saved_path).exists()
    assert Path(saved_path).read_bytes() == b"123456789"

    # Cleanup
    assert cleanup_file(saved_path) is True
    assert not Path(saved_path).exists()

