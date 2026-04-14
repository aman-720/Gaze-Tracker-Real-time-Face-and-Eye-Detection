"""
Eye state detection using the Eye Aspect Ratio (EAR) algorithm.

The EAR is a scalar value computed from the 6 landmark points of each eye.
When the eye is open, EAR is roughly 0.25–0.35. When closed, it drops
below ~0.20. This simple geometric measure is remarkably reliable for
blink and eye-closure detection in real time.

Reference:
    Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks"
    (2016 Computer Vision Winter Workshop)

EAR formula:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

Where p1..p6 are the 6 eye landmarks in order.
"""

import numpy as np
from scipy.spatial import distance as dist


def compute_ear(eye: np.ndarray) -> float:
    """Compute the Eye Aspect Ratio for a single eye.

    Args:
        eye: (6, 2) array of eye landmark points, ordered as:
             p1 (outer corner), p2, p3 (upper lid),
             p4 (inner corner), p5, p6 (lower lid)

    Returns:
        Float EAR value. Higher = more open; lower = more closed.
    """
    # Vertical distances (between upper and lower lid landmarks)
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])

    # Horizontal distance (between eye corners)
    h = dist.euclidean(eye[0], eye[3])

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear


def compute_avg_ear(
    left_eye: np.ndarray, right_eye: np.ndarray
) -> tuple[float, float, float]:
    """Compute EAR for both eyes and return the average.

    Args:
        left_eye: (6, 2) landmarks for left eye.
        right_eye: (6, 2) landmarks for right eye.

    Returns:
        Tuple of (left_ear, right_ear, average_ear).
    """
    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, avg_ear
