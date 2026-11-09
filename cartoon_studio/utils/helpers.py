from __future__ import annotations

import os
import platform
import shutil
import time
from typing import Dict, Any
import torch

def resolve_ffmpeg_binary() -> str:
    """Find available FFmpeg executable from system PATH or bundled imageio-ffmpeg."""
    custom = os.environ.get("FFMPEG_PATH")
    if custom and os.path.exists(custom):
        return custom

    for binary_name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(binary_name)
        if found and os.path.exists(found):
            return found

    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.exists(bundled):
            return bundled
    except Exception:
        pass

    return "ffmpeg"


def get_device_info() -> Dict[str, Any]:
    """Detect current hardware environment (GPU/CPU) and compute capabilities."""
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": "CPU",
        "vram_total_gb": 0.0,
        "vram_allocated_gb": 0.0,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_total_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        info["cuda_capability"] = f"{props.major}.{props.minor}"
        
    return info

def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    seconds = max(0.0, float(seconds))
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def format_file_size(size_bytes: int | float) -> str:
    """Format bytes into readable string (B, KB, MB, GB)."""
    bytes_val = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

class ExecutionTimer:
    """Context manager and utility to measure elapsed execution time."""
    def __init__(self, name: str = "Task"):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
