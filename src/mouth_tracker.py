"""
Yawn detection using the Mouth Aspect Ratio (MAR) algorithm.

MAR is computed from 8 inner-lip landmark points extracted by MediaPipe.
When the mouth is closed, MAR stays in the range of 0.1–0.3. During a
yawn it spikes above ~0.6. By tracking this value frame-by-frame, the
system detects sustained mouth opening as a yawn event.

MAR formula:
    MAR = (||p_top - p_bottom|| + ||p_ul - p_ll|| + ||p_ur - p_lr||)
          / (3 * ||p_left - p_right||)

Where the 8 points are inner lip landmarks ordered as:
    p0 = top center       p1 = upper-left      p2 = upper-right
    p3 = bottom center    p4 = lower-left       p5 = lower-right
    p6 = left corner      p7 = right corner
"""

import numpy as np
from scipy.spatial import distance as dist


def compute_mar(mouth: np.ndarray) -> float:
    """Compute the Mouth Aspect Ratio for inner lip landmarks.

    Args:
        mouth: (8, 2) array of inner lip landmark points, ordered as:
               p0 (top), p1 (upper-left), p2 (upper-right),
               p3 (bottom), p4 (lower-left), p5 (lower-right),
               p6 (left corner), p7 (right corner)

    Returns:
        Float MAR value. Higher = mouth more open; lower = mouth closed.
    """
    # Vertical distances (three pairs across the mouth opening)
    v1 = dist.euclidean(mouth[0], mouth[3])  # top center to bottom center
    v2 = dist.euclidean(mouth[1], mouth[4])  # upper-left to lower-left
    v3 = dist.euclidean(mouth[2], mouth[5])  # upper-right to lower-right

    # Horizontal distance (mouth width)
    h = dist.euclidean(mouth[6], mouth[7])   # left corner to right corner

    if h == 0:
        return 0.0

    mar = (v1 + v2 + v3) / (3.0 * h)
    return mar
