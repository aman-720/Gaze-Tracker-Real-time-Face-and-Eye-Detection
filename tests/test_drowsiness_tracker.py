"""Tests for the drowsiness tracking state machine."""

import pytest
from src.drowsiness_tracker import DrowsinessTracker, DriverState


class TestDrowsinessTracker:
    """Tests for DrowsinessTracker state transitions."""

    def setup_method(self):
        self.tracker = DrowsinessTracker(
            ear_threshold=0.22,
            closed_frames_threshold=10,  # small for testing
        )

    def test_initial_state_is_awake(self):
        assert self.tracker.state == DriverState.AWAKE

    def test_open_eyes_stay_awake(self):
        for _ in range(50):
            state = self.tracker.update(avg_ear=0.30, face_detected=True)
        assert state == DriverState.AWAKE

    def test_no_face_detected(self):
        state = self.tracker.update(avg_ear=0.0, face_detected=False)
        assert state == DriverState.NO_FACE

    def test_closed_eyes_trigger_drowsy(self):
        # Close eyes for 10+ frames (threshold)
        for _ in range(15):
            state = self.tracker.update(avg_ear=0.15, face_detected=True)
        assert state == DriverState.DROWSY

    def test_eyes_below_threshold_not_enough_frames(self):
        # Close for fewer than threshold frames
        for _ in range(5):
            state = self.tracker.update(avg_ear=0.15, face_detected=True)
        assert state == DriverState.AWAKE

    def test_recovery_from_drowsy_to_awake(self):
        # Go drowsy
        for _ in range(15):
            self.tracker.update(avg_ear=0.15, face_detected=True)
        assert self.tracker.state == DriverState.DROWSY

        # Open eyes
        state = self.tracker.update(avg_ear=0.30, face_detected=True)
        assert state == DriverState.AWAKE

    def test_closed_frame_count_resets_on_open(self):
        for _ in range(5):
            self.tracker.update(avg_ear=0.15, face_detected=True)
        assert self.tracker.closed_frame_count == 5

        self.tracker.update(avg_ear=0.30, face_detected=True)
        assert self.tracker.closed_frame_count == 0

    def test_blink_counted_correctly(self):
        # Simulate a blink: 3 frames closed then open
        for _ in range(3):
            self.tracker.update(avg_ear=0.15, face_detected=True)
        self.tracker.update(avg_ear=0.30, face_detected=True)
        assert self.tracker.total_blinks == 1

    def test_drowsiness_event_logged(self):
        # Go drowsy
        for _ in range(15):
            self.tracker.update(avg_ear=0.15, face_detected=True)
        # Recover
        self.tracker.update(avg_ear=0.30, face_detected=True)

        events = self.tracker.events
        assert len(events) == 1
        assert events[0].duration_sec > 0

    def test_reset_clears_state(self):
        for _ in range(15):
            self.tracker.update(avg_ear=0.15, face_detected=True)
        self.tracker.reset()
        assert self.tracker.state == DriverState.AWAKE
        assert self.tracker.closed_frame_count == 0
        assert self.tracker.total_blinks == 0
        assert len(self.tracker.events) == 0

    def test_get_summary(self):
        summary = self.tracker.get_summary()
        assert "total_drowsiness_events" in summary
        assert "total_blinks" in summary
        assert "events" in summary

    def test_closure_duration_calculation(self):
        for _ in range(30):
            self.tracker.update(avg_ear=0.15, face_detected=True)
        duration = self.tracker.get_closure_duration_sec(fps=30.0)
        assert duration == pytest.approx(1.0)
