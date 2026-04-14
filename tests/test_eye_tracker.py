"""Tests for the eye tracking (EAR computation) module."""

import numpy as np
import pytest
from src.eye_tracker import compute_ear, compute_avg_ear


def _make_eye(vertical_gap: float, horizontal_gap: float = 20.0) -> np.ndarray:
    """Create a synthetic 6-point eye with controlled dimensions.

    Points layout (approximation):
        p0 ---- p3
       / p1  p2  \\
       \\ p5  p4  /
    """
    half_h = vertical_gap / 2
    return np.array([
        [0, 0],                          # p0: left corner
        [5, -half_h],                    # p1: upper-left lid
        [15, -half_h],                   # p2: upper-right lid
        [horizontal_gap, 0],             # p3: right corner
        [15, half_h],                    # p4: lower-right lid
        [5, half_h],                     # p5: lower-left lid
    ], dtype=np.float64)


class TestComputeEAR:
    """Tests for the compute_ear function."""

    def test_open_eye_returns_positive_ear(self):
        eye = _make_eye(vertical_gap=8.0, horizontal_gap=20.0)
        ear = compute_ear(eye)
        assert ear > 0.2, f"Open eye should have EAR > 0.2, got {ear}"

    def test_closed_eye_returns_low_ear(self):
        eye = _make_eye(vertical_gap=0.5, horizontal_gap=20.0)
        ear = compute_ear(eye)
        assert ear < 0.1, f"Closed eye should have EAR < 0.1, got {ear}"

    def test_zero_horizontal_returns_zero(self):
        eye = _make_eye(vertical_gap=8.0, horizontal_gap=0.0)
        ear = compute_ear(eye)
        assert ear == 0.0

    def test_ear_increases_with_openness(self):
        ear_small = compute_ear(_make_eye(vertical_gap=2.0))
        ear_large = compute_ear(_make_eye(vertical_gap=10.0))
        assert ear_large > ear_small


class TestComputeAvgEAR:
    """Tests for the average EAR function."""

    def test_symmetric_eyes_match_average(self):
        eye = _make_eye(vertical_gap=8.0)
        l_ear, r_ear, avg_ear = compute_avg_ear(eye, eye)
        assert l_ear == pytest.approx(r_ear)
        assert avg_ear == pytest.approx(l_ear)

    def test_asymmetric_eyes_average_correctly(self):
        open_eye = _make_eye(vertical_gap=10.0)
        closed_eye = _make_eye(vertical_gap=1.0)
        l_ear, r_ear, avg_ear = compute_avg_ear(open_eye, closed_eye)
        assert avg_ear == pytest.approx((l_ear + r_ear) / 2.0)
