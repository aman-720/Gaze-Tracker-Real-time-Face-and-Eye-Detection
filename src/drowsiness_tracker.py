"""
Drowsiness state tracker with temporal logic.

Tracks how long the driver's eyes have been closed and determines
whether to trigger an alarm. Uses frame counting against a configurable
threshold (default: ~3 seconds at 30 FPS = 90 frames).

State machine:
    AWAKE  →  eyes closed for N consecutive frames  →  DROWSY (alarm!)
    DROWSY →  eyes open again                       →  AWAKE  (alarm stops)
"""

import time
from enum import Enum, auto
from dataclasses import dataclass


class DriverState(Enum):
    """Current assessed state of the driver."""
    AWAKE = auto()
    DROWSY = auto()
    YAWNING = auto()
    NO_FACE = auto()


@dataclass
class DrowsinessEvent:
    """Record of a drowsiness event for logging."""
    start_time: float
    end_time: float | None = None
    duration_sec: float = 0.0
    frame_count: int = 0


class DrowsinessTracker:
    """Tracks consecutive eye-closure frames and determines drowsiness."""

    def __init__(
        self,
        ear_threshold: float = 0.22,
        closed_frames_threshold: int = 90,
        mar_threshold: float = 0.6,
        yawn_frames_threshold: int = 15,
        yawn_count_warning: int = 3,
    ):
        self.ear_threshold = ear_threshold
        self.closed_frames_threshold = closed_frames_threshold
        self.mar_threshold = mar_threshold
        self.yawn_frames_threshold = yawn_frames_threshold
        self.yawn_count_warning = yawn_count_warning

        # Internal state — eye closure
        self._closed_frame_count: int = 0
        self._state: DriverState = DriverState.AWAKE
        self._current_event: DrowsinessEvent | None = None
        self._events: list[DrowsinessEvent] = []
        self._total_blinks: int = 0
        self._blink_frame_count: int = 0

        # Internal state — yawn tracking
        self._yawn_frame_count: int = 0
        self._total_yawns: int = 0
        self._is_yawning: bool = False

    @property
    def state(self) -> DriverState:
        return self._state

    @property
    def closed_frame_count(self) -> int:
        return self._closed_frame_count

    @property
    def events(self) -> list[DrowsinessEvent]:
        return self._events.copy()

    @property
    def total_blinks(self) -> int:
        return self._total_blinks

    @property
    def total_yawns(self) -> int:
        return self._total_yawns

    @property
    def yawn_warning(self) -> bool:
        """True when yawn count has reached the warning threshold."""
        return self._total_yawns >= self.yawn_count_warning

    def update(self, avg_ear: float, face_detected: bool, mar: float = 0.0) -> DriverState:
        """Update tracker with a new frame's EAR and MAR values.

        Args:
            avg_ear: Average Eye Aspect Ratio for both eyes.
            face_detected: Whether a face was detected in this frame.
            mar: Mouth Aspect Ratio (0.0 if mouth not tracked).

        Returns:
            Current DriverState after this update.
        """
        if not face_detected:
            self._state = DriverState.NO_FACE
            return self._state

        # ── Eye closure tracking ──────────────────────────────────
        if avg_ear < self.ear_threshold:
            self._closed_frame_count += 1

            if (
                self._closed_frame_count >= self.closed_frames_threshold
                and self._state != DriverState.DROWSY
            ):
                self._state = DriverState.DROWSY
                self._current_event = DrowsinessEvent(
                    start_time=time.time(),
                    frame_count=self._closed_frame_count,
                )
        else:
            if self._state == DriverState.DROWSY and self._current_event:
                self._current_event.end_time = time.time()
                self._current_event.duration_sec = (
                    self._current_event.end_time - self._current_event.start_time
                )
                self._events.append(self._current_event)
                self._current_event = None

            if 2 <= self._closed_frame_count <= 8:
                self._total_blinks += 1

            self._closed_frame_count = 0
            # Only set AWAKE if not currently in a yawn
            if self._state != DriverState.YAWNING:
                self._state = DriverState.AWAKE

        # ── Yawn tracking ─────────────────────────────────────────
        # DROWSY takes priority — don't downgrade to YAWNING
        if mar > self.mar_threshold:
            self._yawn_frame_count += 1

            if (
                self._yawn_frame_count >= self.yawn_frames_threshold
                and not self._is_yawning
            ):
                self._is_yawning = True
                self._total_yawns += 1
                if self._state != DriverState.DROWSY:
                    self._state = DriverState.YAWNING
        else:
            self._yawn_frame_count = 0
            self._is_yawning = False
            # Return to AWAKE from YAWNING (but not from DROWSY)
            if self._state == DriverState.YAWNING:
                self._state = DriverState.AWAKE

        return self._state

    def reset(self):
        """Reset all tracking state."""
        self._closed_frame_count = 0
        self._state = DriverState.AWAKE
        self._current_event = None
        self._events.clear()
        self._total_blinks = 0
        self._yawn_frame_count = 0
        self._total_yawns = 0
        self._is_yawning = False

    def get_closure_duration_sec(self, fps: float = 30.0) -> float:
        """Get current eye closure duration in seconds."""
        if fps <= 0:
            return 0.0
        return self._closed_frame_count / fps

    def get_summary(self) -> dict:
        """Return a summary of the session statistics."""
        return {
            "total_drowsiness_events": len(self._events),
            "total_blinks": self._total_blinks,
            "total_yawns": self._total_yawns,
            "events": [
                {
                    "start": e.start_time,
                    "end": e.end_time,
                    "duration_sec": round(e.duration_sec, 2),
                }
                for e in self._events
            ],
        }
