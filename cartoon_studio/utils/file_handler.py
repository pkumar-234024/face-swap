from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np

from cartoon_studio.config import TEMP_DIR, UPLOADS_DIR, OUTPUTS_DIR

class FileHandler:
    """Manages file storage, session directories, and image/video I/O."""

    @staticmethod
    def create_job_dir(job_id: Optional[str] = None) -> Path:
        """Create a dedicated workspace directory for a processing job."""
        if not job_id:
            job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_dir = TEMP_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    @staticmethod
    def save_uploaded_file(uploaded_file, destination_dir: Optional[Path] = None) -> Path:
        """Save an uploaded file from Streamlit to disk with a clean sanitized name."""
        if destination_dir is None:
            destination_dir = UPLOADS_DIR
        destination_dir.mkdir(parents=True, exist_ok=True)
        
        name = getattr(uploaded_file, "name", "uploaded_file.mp4")
        ext = Path(name).suffix.lower()
        stem = Path(name).stem
        sanitized_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_")).strip()
        if not sanitized_stem:
            sanitized_stem = "video"
        
        unique_name = f"{sanitized_stem}_{uuid.uuid4().hex[:8]}{ext}"
        target_path = destination_dir / unique_name
        
        with open(target_path, "wb") as f:
            if hasattr(uploaded_file, "getbuffer"):
                f.write(uploaded_file.getbuffer())
            elif hasattr(uploaded_file, "read"):
                f.write(uploaded_file.read())
            else:
                f.write(uploaded_file)
                
        return target_path

    @staticmethod
    def read_image_bgr(image_path: str | Path) -> Optional[np.ndarray]:
        """Read image using OpenCV with EXIF rotation support."""
        path_str = str(image_path)
        if not os.path.exists(path_str):
            return None
        # Use imdecode to support unicode and non-ascii paths on Windows
        try:
            with open(path_str, "rb") as f:
                bytes_data = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
                return img
        except Exception:
            return cv2.imread(path_str, cv2.IMREAD_COLOR)

    @staticmethod
    def write_image(image_path: str | Path, image_bgr: np.ndarray) -> bool:
        """Write image safely to disk handling unicode paths."""
        path_str = str(image_path)
        ext = Path(path_str).suffix.lower()
        if not ext:
            ext = ".png"
        try:
            success, encoded_img = cv2.imencode(ext, image_bgr)
            if success:
                with open(path_str, "wb") as f:
                    f.write(encoded_img.tobytes())
                return True
        except Exception:
            pass
        return cv2.imwrite(path_str, image_bgr)

    @staticmethod
    def cleanup_directory(directory_path: Path, ignore_errors: bool = True) -> bool:
        """Safely remove a temporary directory and its contents."""
        try:
            if directory_path.exists() and directory_path.is_dir():
                shutil.rmtree(directory_path, ignore_errors=ignore_errors)
                return True
        except Exception:
            pass
        return False

