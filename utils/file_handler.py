from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from config import UPLOADS_DIR


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def save_uploaded_file(uploaded_file, subdirectory: str = "") -> str:
    if uploaded_file is None:
        raise ValueError("No uploaded file provided.")

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    target_dir = UPLOADS_DIR / subdirectory if subdirectory else UPLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = sanitize_filename(getattr(uploaded_file, "name", "upload.bin"))
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    unique_name = f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    destination = target_dir / unique_name

    with open(destination, "wb") as output_file:
        if hasattr(uploaded_file, "getvalue"):
            output_file.write(uploaded_file.getvalue())
        else:
            uploaded_file.seek(0)
            shutil.copyfileobj(uploaded_file, output_file)

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    return str(destination)


def cleanup_file(path: Optional[str]) -> None:
    if not path:
        return
    if os.path.exists(path):
        os.remove(path)


def cleanup_directory(directory: str) -> None:
    if not directory:
        return
    directory_path = Path(directory)
    if directory_path.exists() and directory_path.is_dir():
        shutil.rmtree(directory_path)
