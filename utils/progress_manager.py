from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from utils.helpers import format_seconds


@dataclass
class ProgressUpdate:
    stage: str
    percentage: float
    current_frame: int
    total_frames: int
    fps: float
    elapsed_seconds: float
    eta_seconds: float
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ProgressManager:
    """Tracks and calculates frame processing progress, FPS, and ETA."""

    def __init__(
        self,
        total_frames: int = 0,
        callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ):
        self.total_frames = max(0, total_frames)
        self.current_frame = 0
        self.current_stage = "Initializing"
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.callback = callback
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def set_stage(self, stage_name: str, message: str = "") -> None:
        """Transitions to a new pipeline stage."""
        self.current_stage = stage_name
        self._notify(message=message)

    def set_total_frames(self, total_frames: int) -> None:
        """Updates total expected frames."""
        self.total_frames = max(1, total_frames)

    def advance_frame(self, count: int = 1, message: str = "") -> None:
        """Increments processed frame counter and computes metrics."""
        self.current_frame += count
        self._notify(message=message)

    def add_warning(self, warning_msg: str) -> None:
        """Records a non-fatal warning."""
        self.warnings.append(warning_msg)

    def add_error(self, error_msg: str) -> None:
        """Records an error."""
        self.errors.append(error_msg)

    def _notify(self, message: str = "") -> None:
        now = time.time()
        elapsed = max(0.001, now - self.start_time)
        fps = self.current_frame / elapsed if self.current_frame > 0 else 0.0

        if self.total_frames > 0:
            percentage = min(100.0, (self.current_frame / self.total_frames) * 100.0)
            remaining_frames = max(0, self.total_frames - self.current_frame)
            eta = (remaining_frames / fps) if fps > 0.01 else 0.0
        else:
            percentage = 0.0
            eta = 0.0

        update = ProgressUpdate(
            stage=self.current_stage,
            percentage=percentage,
            current_frame=self.current_frame,
            total_frames=self.total_frames,
            fps=round(fps, 1),
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            message=message,
            warnings=list(self.warnings),
            errors=list(self.errors),
        )

        if self.callback is not None:
            self.callback(update)

