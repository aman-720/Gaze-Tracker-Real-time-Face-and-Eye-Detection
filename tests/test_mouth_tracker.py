"""Tests for the mouth tracking (MAR computation) module."""

import numpy as np
import pytest
from src.mouth_tracker import compute_mar


def _make_mouth(vertical_gap: float, horizontal_gap: float = 30.0) -> np.ndarray:
    """Create a synthetic 8-point mouth with controlled dimensions.

    Points layout (inner lips, frontal view):
        p6 (left corner) ---- p7 (right corner)
             p1    p2   (upper inner)
              p0         (top center)
              p3         (bottom center)
             p4    p5   (lower inner)

    vertical_gap controls the distance between top/bottom center points.
    """
    half_h = vertical_gap / 2
    third_w = horizontal_gap / 3

    return np.array([
        [horizontal_gap / 2, -half_h],          # p0: top center
        [third_w, -half_h * 0.8],               # p1: upper-left
        [2 * third_w, -half_h * 0.8],           # p2: upper-right
        [horizontal_gap / 2, half_h],            # p3: bottom center
        [third_w, half_h * 0.8],                 # p4: lower-left
        [2 * third_w, half_h * 0.8],             # p5: lower-right
        [0, 0],                                  # p6: left corner
        [horizontal_gap, 0],                     # p7: right corner
    ], dtype=np.float64)


class TestComputeMAR:
    """Tests for the compute_mar function."""

    def test_closed_mouth_returns_low_mar(self):
        mouth = _make_mouth(vertical_gap=1.0, horizontal_gap=30.0)
        mar = compute_mar(mouth)
        assert mar < 0.1, f"Closed mouth should have MAR < 0.1, got {mar}"

    def test_open_mouth_returns_high_mar(self):
        mouth = _make_mouth(vertical_gap=25.0, horizontal_gap=30.0)
        mar = compute_mar(mouth)
        assert mar > 0.5, f"Wide open mouth (yawn) should have MAR > 0.5, got {mar}"

    def test_zero_horizontal_returns_zero(self):
        mouth = _make_mouth(vertical_gap=10.0, horizontal_gap=0.0)
        mar = compute_mar(mouth)
        assert mar == 0.0

    def test_mar_increases_with_openness(self):
        mar_small = compute_mar(_make_mouth(vertical_gap=2.0))
        mar_large = compute_mar(_make_mouth(vertical_gap=20.0))
        assert mar_large > mar_small

    def test_mar_is_non_negative(self):
        mouth = _make_mouth(vertical_gap=5.0)
        mar = compute_mar(mouth)
        assert mar >= 0.0
