from __future__ import annotations

import logging
import sys
from typing import Optional

_LOGGERS = {}


def setup_logging(level: int = logging.INFO) -> None:
    """Configures global logging format and handler."""
    root_logger = logging.getLogger("face_studio")
    if not root_logger.handlers:
        root_logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a named logger child of face_studio."""
    setup_logging()
    full_name = f"face_studio.{name}" if name else "face_studio"
    if full_name not in _LOGGERS:
        _LOGGERS[full_name] = logging.getLogger(full_name)
    return _LOGGERS[full_name]

