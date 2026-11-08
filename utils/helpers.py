from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def format_file_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    return f"{size:.2f} {units[index]}"


def ensure_directory(path: str) -> str:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def make_temp_path(base_dir: str, prefix: str, extension: str) -> str:
    base_dir_path = Path(base_dir)
    base_dir_path.mkdir(parents=True, exist_ok=True)
    return str(base_dir_path / f"{prefix}_{os.urandom(4).hex()}{extension}")


def safe_path(path: Optional[str]) -> str:
    return str(Path(path).resolve()) if path else ""
