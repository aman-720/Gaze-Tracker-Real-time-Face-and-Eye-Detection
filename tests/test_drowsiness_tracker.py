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


class TestYawnTracking:
    """Tests for yawn detection state machine."""

    def setup_method(self):
        self.tracker = DrowsinessTracker(
            ear_threshold=0.22,
            closed_frames_threshold=10,
            mar_threshold=0.6,
            yawn_frames_threshold=5,   # small for testing
            yawn_count_warning=3,
        )

    def test_no_yawn_when_mouth_closed(self):
        for _ in range(20):
            state = self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.2)
        assert state == DriverState.AWAKE
        assert self.tracker.total_yawns == 0

    def test_yawn_detected_after_threshold_frames(self):
        for _ in range(10):
            state = self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.8)
        assert state == DriverState.YAWNING
        assert self.tracker.total_yawns == 1

    def test_yawn_not_counted_below_threshold_frames(self):
        # Open mouth for fewer frames than threshold
        for _ in range(3):
            self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.8)
        # Close mouth
        self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.2)
        assert self.tracker.total_yawns == 0

    def test_recovery_from_yawning_to_awake(self):
        # Trigger yawn
        for _ in range(10):
            self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.8)
        assert self.tracker.state == DriverState.YAWNING

        # Close mouth
        state = self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.2)
        assert state == DriverState.AWAKE

    def test_yawn_warning_after_count_threshold(self):
        assert not self.tracker.yawn_warning

        for yawn in range(3):
            # Yawn
            for _ in range(10):
                self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.8)
            # Close mouth between yawns
            for _ in range(5):
                self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.2)

        assert self.tracker.total_yawns == 3
        assert self.tracker.yawn_warning

    def test_drowsy_takes_priority_over_yawning(self):
        # Close eyes long enough to trigger drowsy
        for _ in range(15):
            self.tracker.update(avg_ear=0.15, face_detected=True, mar=0.8)
        # DROWSY should take priority even though mouth is open
        assert self.tracker.state == DriverState.DROWSY

    def test_single_yawn_counted_once(self):
        # Sustained open mouth should only count as one yawn
        for _ in range(30):
            self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.8)
        assert self.tracker.total_yawns == 1

    def test_yawn_count_in_summary(self):
        for _ in range(10):
            self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.8)
        summary = self.tracker.get_summary()
        assert summary["total_yawns"] == 1

    def test_reset_clears_yawn_state(self):
        for _ in range(10):
            self.tracker.update(avg_ear=0.30, face_detected=True, mar=0.8)
        self.tracker.reset()
        assert self.tracker.total_yawns == 0
        assert not self.tracker.yawn_warning
