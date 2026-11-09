from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple

# Weighted progress percentage ranges for realistic multi-stage execution
STAGE_PROGRESS_RANGES: Dict[int, Tuple[float, float]] = {
    1: (0.00, 0.04),  # Video analysis & scene cut detection
    2: (0.04, 0.08),  # Audio track extraction
    3: (0.08, 0.12),  # Pose & depth guidance setup
    4: (0.12, 0.15),  # Engine & style shader setup
    5: (0.15, 0.80),  # Frame-by-frame 2D cartoonization & temporal stabilization (65% of work)
    6: (0.80, 0.88),  # H.264 video track encoding
    7: (0.88, 0.93),  # Frame-accurate audio remuxing
    8: (0.93, 0.98),  # Synchronous side-by-side comparison rendering
    9: (0.98, 0.99),  # Finalization & verification
    10: (0.99, 1.00), # Studio complete
}

@dataclass
class ProgressUpdate:
    stage_name: str
    stage_step: int
    total_stages: int
    current_frame: int
    total_frames: int
    overall_progress: float  # 0.0 to 1.0
    fps: float
    eta_seconds: float
    status_message: str
    timestamp: str = ""
    logs: List[str] = field(default_factory=list)

ProgressCallback = Callable[[ProgressUpdate], None]

class ProgressManager:
    """Tracks and calculates multi-stage pipeline progress, processing speed, ETA, and console/UI logs."""

    def __init__(self, total_stages: int = 10, total_frames: int = 1, echo_console: bool = True):
        self.total_stages = max(1, total_stages)
        self.total_frames = max(1, total_frames)
        self.current_stage_step = 1
        self.current_stage_name = "Initializing"
        self.current_frame = 0
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.last_update_time = time.time()
        self.frames_in_stage = 0
        self.echo_console = echo_console
        self.logs: List[str] = []

    def log(self, message: str, level: str = "INFO") -> str:
        """Record a timestamped log message and echo to console."""
        ts = time.strftime("%H:%M:%S")
        prefix = f"[{ts}] [{level}]"
        formatted_entry = f"{prefix} {message}"
        self.logs.append(formatted_entry)
        if self.echo_console:
            print(formatted_entry, flush=True)
        return formatted_entry

    def start_stage(self, stage_step: int, stage_name: str, total_frames: Optional[int] = None) -> None:
        """Begin a new pipeline stage and log event."""
        self.current_stage_step = stage_step
        self.current_stage_name = stage_name
        if total_frames is not None:
            self.total_frames = max(1, total_frames)
        self.current_frame = 0
        self.frames_in_stage = 0
        self.stage_start_time = time.time()
        
        self.log(f"Stage {stage_step}/{self.total_stages}: {stage_name}", level="STAGE")

    def update_frame(self, current_frame: int, message: str = "") -> ProgressUpdate:
        """Update progress within current frame processing loop with weighted progress scaling."""
        self.current_frame = current_frame
        self.frames_in_stage = current_frame
        now = time.time()
        ts = time.strftime("%H:%M:%S")

        elapsed_stage = max(0.001, now - self.stage_start_time)
        fps = self.frames_in_stage / elapsed_stage if self.frames_in_stage > 0 else 0.0

        remaining_frames = max(0, self.total_frames - current_frame)
        eta_seconds = (remaining_frames / fps) if fps > 0.05 else 0.0

        # Calculate fractional stage progress
        stage_fraction = min(1.0, current_frame / self.total_frames) if self.total_frames > 0 else 0.0
        
        # Use realistic weighted stage intervals
        s_min, s_max = STAGE_PROGRESS_RANGES.get(self.current_stage_step, (0.0, 1.0))
        overall_progress = min(0.99, s_min + stage_fraction * (s_max - s_min))

        if not message:
            message = f"Processing frame {current_frame}/{self.total_frames} ({fps:.1f} fps) | ETA: {int(eta_seconds)}s"

        if current_frame > 0 and (current_frame % 5 == 0 or current_frame == self.total_frames):
            self.log(f"[Stage {self.current_stage_step}] {message}", level="FRAME")

        return ProgressUpdate(
            stage_name=self.current_stage_name,
            stage_step=self.current_stage_step,
            total_stages=self.total_stages,
            current_frame=current_frame,
            total_frames=self.total_frames,
            overall_progress=overall_progress,
            fps=fps,
            eta_seconds=eta_seconds,
            status_message=message,
            timestamp=ts,
            logs=list(self.logs),
        )

    def complete_stage(self, message: str = "") -> ProgressUpdate:
        """Mark current stage as finished and append completion log."""
        ts = time.strftime("%H:%M:%S")
        _, s_max = STAGE_PROGRESS_RANGES.get(self.current_stage_step, (0.0, 1.0))
        overall_progress = min(1.0, s_max)
        
        if not message:
            message = f"Completed stage {self.current_stage_step}/{self.total_stages}: {self.current_stage_name}"

        self.log(message, level="DONE")

        return ProgressUpdate(
            stage_name=self.current_stage_name,
            stage_step=self.current_stage_step,
            total_stages=self.total_stages,
            current_frame=self.total_frames,
            total_frames=self.total_frames,
            overall_progress=overall_progress,
            fps=0.0,
            eta_seconds=0.0,
            status_message=message,
            timestamp=ts,
            logs=list(self.logs),
        )
