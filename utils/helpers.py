from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None


def format_file_size(num_bytes: int) -> str:
    """Formats raw byte count into human-readable units."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(0, num_bytes))
    index = 0
    while size >= 1024.0 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    return f"{size:.2f} {units[index]}"


def format_seconds(seconds: float) -> str:
    """Formats seconds into MM:SS or HH:MM:SS format."""
    total_sec = max(0, int(round(seconds)))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def ensure_directory(path: str | Path) -> str:
    """Creates directory if missing and returns resolved absolute path."""
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def make_temp_path(base_dir: str | Path, prefix: str = "temp", extension: str = ".tmp") -> str:
    """Generates a collision-resistant unique file path in base_dir."""
    base_dir_path = Path(base_dir).resolve()
    base_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    rand_hex = os.urandom(4).hex()
    clean_ext = extension if extension.startswith(".") else f".{extension}"
    return str(base_dir_path / f"{prefix}_{timestamp}_{rand_hex}{clean_ext}")


def safe_path(path: Optional[str | Path]) -> str:
    """Returns resolved string path or empty string."""
    return str(Path(path).resolve()) if path else ""


def get_device(preference: str = "auto") -> str:
    """Determines optimal PyTorch and ONNX execution device.

    Args:
        preference: "auto", "cuda", or "cpu".

    Returns:
        "cuda" if available and preferred, else "cpu".
    """
    if preference.lower() == "cpu":
        return "cpu"

    if torch is not None and torch.cuda.is_available():
        return "cuda"

    return "cpu"
