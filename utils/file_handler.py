from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from config import TEMP_DIR, UPLOADS_DIR
from utils.logging_config import get_logger

logger = get_logger("file_handler")


def sanitize_filename(filename: str) -> str:
    """Removes unsafe characters from filename to prevent directory traversal."""
    name = Path(filename).name
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in name)
    return cleaned or "unnamed_file"


def save_uploaded_file(uploaded_file, subdirectory: str = "") -> str:
    """Saves a Streamlit UploadedFile or file-like buffer to disk securely.

    Args:
        uploaded_file: Streamlit UploadedFile or io.BytesIO object.
        subdirectory: Optional sub-folder under UPLOADS_DIR.

    Returns:
        Absolute string path to the saved file.
    """
    if uploaded_file is None:
        raise ValueError("No uploaded file was provided.")

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    target_dir = (UPLOADS_DIR / subdirectory) if subdirectory else UPLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    original_name = getattr(uploaded_file, "name", "upload.bin") or "upload.bin"
    clean_name = sanitize_filename(original_name)
    stem = Path(clean_name).stem
    suffix = Path(clean_name).suffix.lower()
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

    logger.info("Saved uploaded file to %s (%d bytes)", destination, destination.stat().st_size)
    return str(destination.resolve())


def cleanup_file(path: Optional[str | Path]) -> bool:
    """Safely removes a single file if it exists."""
    if not path:
        return False
    try:
        p = Path(path)
        if p.exists() and p.is_file():
            p.unlink()
            return True
    except Exception as exc:
        logger.warning("Failed to clean up file %s: %s", path, exc)
    return False


def cleanup_directory(directory: str | Path, keep_root: bool = False) -> None:
    """Recursively removes a directory and its contents, or empties it if keep_root is True."""
    if not directory:
        return
    try:
        dir_path = Path(directory)
        if dir_path.exists() and dir_path.is_dir():
            if keep_root:
                for item in dir_path.iterdir():
                    if item.is_file() or item.is_symlink():
                        item.unlink(missing_ok=True)
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
            else:
                shutil.rmtree(dir_path, ignore_errors=True)
    except Exception as exc:
        logger.warning("Error during directory cleanup for %s: %s", directory, exc)


def create_job_temp_dir(job_id: Optional[str] = None) -> Path:
    """Creates an isolated temporary folder for a processing job."""
    uid = job_id or uuid.uuid4().hex[:8]
    job_dir = TEMP_DIR / f"job_{uid}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def read_image_bgr(image_path: str | Path) -> Optional[np.ndarray]:
    """Safely loads an image in BGR format, auto-orienting EXIF tags and stripping transparency."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return None

    try:
        from PIL import Image, ImageOps
        import cv2
        with Image.open(str(p)) as pil_img:
            # Auto-orient based on camera EXIF tags (solves sideways smartphone portraits)
            pil_img = ImageOps.exif_transpose(pil_img)
            # Handle alpha channel (RGBA/PNG/WEBP)
            if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                if pil_img.mode != "RGBA":
                    pil_img = pil_img.convert("RGBA")
                bg.paste(pil_img, mask=pil_img.split()[3])
                rgb = np.array(bg)
            else:
                rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        import cv2
        return cv2.imread(str(p))
