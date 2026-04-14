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
    ):
        self.ear_threshold = ear_threshold
        self.closed_frames_threshold = closed_frames_threshold

        # Internal state
        self._closed_frame_count: int = 0
        self._state: DriverState = DriverState.AWAKE
        self._current_event: DrowsinessEvent | None = None
        self._events: list[DrowsinessEvent] = []
        self._total_blinks: int = 0
        self._blink_frame_count: int = 0

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

    def update(self, avg_ear: float, face_detected: bool) -> DriverState:
        """Update tracker with a new frame's EAR value.

        Args:
            avg_ear: Average Eye Aspect Ratio for both eyes.
            face_detected: Whether a face was detected in this frame.

        Returns:
            Current DriverState after this update.
        """
        if not face_detected:
            self._state = DriverState.NO_FACE
            return self._state

        if avg_ear < self.ear_threshold:
            # Eyes are closed this frame
            self._closed_frame_count += 1

            if (
                self._closed_frame_count >= self.closed_frames_threshold
                and self._state != DriverState.DROWSY
            ):
                # Transition to DROWSY
                self._state = DriverState.DROWSY
                self._current_event = DrowsinessEvent(
                    start_time=time.time(),
                    frame_count=self._closed_frame_count,
                )
        else:
            # Eyes are open this frame
            if self._state == DriverState.DROWSY and self._current_event:
                # End the drowsiness event
                self._current_event.end_time = time.time()
                self._current_event.duration_sec = (
                    self._current_event.end_time - self._current_event.start_time
                )
                self._events.append(self._current_event)
                self._current_event = None

            # Count short closures as blinks (roughly 2–6 frames at 30 FPS)
            if 2 <= self._closed_frame_count <= 8:
                self._total_blinks += 1

            self._closed_frame_count = 0
            self._state = DriverState.AWAKE

        return self._state

    def reset(self):
        """Reset all tracking state."""
        self._closed_frame_count = 0
        self._state = DriverState.AWAKE
        self._current_event = None
        self._events.clear()
        self._total_blinks = 0

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
            "events": [
                {
                    "start": e.start_time,
                    "end": e.end_time,
                    "duration_sec": round(e.duration_sec, 2),
                }
                for e in self._events
            ],
        }
