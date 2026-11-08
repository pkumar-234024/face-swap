from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from config import BUFFALO_L_DIR, MODELS_DIR
from utils.logging_config import get_logger

logger = get_logger("model_manager")


class ModelManager:
    """Manages AI model paths, verification, unpacking, and licensing statuses."""

    REQUIRED_BUFFALO_MODELS = {
        "det_10g.onnx": "Face Detection (SCRFD 10G)",
        "w600k_r50.onnx": "Identity Feature Recognition (ArcFace ResNet-50)",
        "1k3d68.onnx": "3D 68-Point Facial Landmarks Network",
        "2d106det.onnx": "Dense 2D 106-Point Facial Landmarks Network",
        "genderage.onnx": "Facial Attribute Analysis (Gender & Age)",
    }

    OPTIONAL_MODELS = {
        "inswapper_128.onnx": "Identity-Conditioned InSwapper (128x128)",
    }

    def __init__(self, models_dir: Path | str = MODELS_DIR):
        self.models_dir = Path(models_dir).resolve()
        self.buffalo_dir = self.models_dir / "buffalo_l"
        self._ensure_models_unpacked()

    def _ensure_models_unpacked(self) -> None:
        """Extracts buffalo_l.zip if folder is missing or incomplete."""
        zip_path = self.models_dir / "buffalo_l.zip"
        if zip_path.exists() and (not self.buffalo_dir.exists() or not any(self.buffalo_dir.iterdir())):
            logger.info("Extracting %s into %s...", zip_path, self.buffalo_dir)
            try:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    archive.extractall(self.models_dir)
                logger.info("Extraction completed successfully.")
            except Exception as exc:
                logger.error("Failed to unpack %s: %s", zip_path, exc)

    def check_buffalo_models(self) -> Dict[str, Any]:
        """Verifies presence of all required Buffalo_L models."""
        status: Dict[str, Any] = {
            "all_present": True,
            "models_dir": str(self.buffalo_dir),
            "found": {},
            "missing": [],
        }

        for filename, desc in self.REQUIRED_BUFFALO_MODELS.items():
            model_path = self.buffalo_dir / filename
            if model_path.exists() and model_path.stat().st_size > 1024:
                status["found"][filename] = {
                    "description": desc,
                    "path": str(model_path),
                    "size_bytes": model_path.stat().st_size,
                }
            else:
                status["all_present"] = False
                status["missing"].append(filename)

        return status

    def check_inswapper_model(self) -> Dict[str, Any]:
        """Checks if optional inswapper_128.onnx is available."""
        inswapper_path = self.models_dir / "inswapper_128.onnx"
        exists = inswapper_path.exists() and inswapper_path.stat().st_size > 1024
        return {
            "present": exists,
            "path": str(inswapper_path) if exists else None,
            "description": self.OPTIONAL_MODELS["inswapper_128.onnx"],
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Generates a complete summary of AI models and engine capabilities."""
        buffalo_status = self.check_buffalo_models()
        inswapper_status = self.check_inswapper_model()

        return {
            "buffalo_l": buffalo_status,
            "inswapper": inswapper_status,
            "facial_animation_ready": buffalo_status["all_present"],
            "deep_swap_ready": inswapper_status["present"],
            "licensing": {
                "insightface": "Academic & Non-Commercial Research License",
                "media_engine": "Open Source Computer Vision / PyTorch Engine",
            },
        }

