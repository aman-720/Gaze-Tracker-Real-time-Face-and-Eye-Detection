"""
UI overlay and visualization module.

Draws real-time information onto the video frame:
  - Face bounding box
  - Eye landmarks (contours)
  - EAR value display
  - Driver state indicator (AWAKE / DROWSY)
  - FPS counter
  - Drowsiness progress bar
  - Alert flash effect when drowsy
"""

import cv2
import numpy as np
from src.drowsiness_tracker import DriverState


# ── Color palette (BGR) ───────────────────────────────────────────
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 140, 255)


def draw_overlay(
    frame: np.ndarray,
    state: DriverState,
    ear: float,
    fps: float,
    closed_frames: int,
    threshold_frames: int,
    landmarks: np.ndarray | None = None,
    left_eye: np.ndarray | None = None,
    right_eye: np.ndarray | None = None,
    face_rect=None,
    show_landmarks: bool = True,
    show_ear: bool = True,
    show_fps: bool = True,
) -> np.ndarray:
    """Draw all UI elements onto the frame.

    Args:
        frame: BGR image to draw on (modified in place and returned).
        state: Current DriverState.
        ear: Current average EAR value.
        fps: Current frames per second.
        closed_frames: How many consecutive frames eyes have been closed.
        threshold_frames: Frame threshold for drowsiness.
        landmarks: Full 68-point landmarks (optional).
        left_eye: (6, 2) left eye landmarks.
        right_eye: (6, 2) right eye landmarks.
        face_rect: dlib rectangle for the detected face.
        show_landmarks: Whether to draw eye contours.
        show_ear: Whether to show EAR value.
        show_fps: Whether to show FPS counter.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # ── Face bounding box ──────────────────────────────────────────
    # face_rect is a dict {"x1", "y1", "x2", "y2"} from MediaPipe
    if face_rect is not None:
        color = RED if state == DriverState.DROWSY else GREEN
        cv2.rectangle(
            frame,
            (face_rect["x1"], face_rect["y1"]),
            (face_rect["x2"], face_rect["y2"]),
            color, 2,
        )

    # ── Eye contours ───────────────────────────────────────────────
    if show_landmarks:
        if left_eye is not None:
            hull = cv2.convexHull(left_eye)
            cv2.drawContours(frame, [hull], -1, GREEN, 1)
        if right_eye is not None:
            hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [hull], -1, GREEN, 1)

    # ── Status bar background ──────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), BLACK, -1)

    # ── Driver state label ─────────────────────────────────────────
    if state == DriverState.DROWSY:
        label = "DROWSY - WAKE UP!"
        label_color = RED
        # Flash effect: semi-transparent red border
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), RED, 20)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    elif state == DriverState.NO_FACE:
        label = "NO FACE DETECTED"
        label_color = YELLOW
    else:
        label = "AWAKE"
        label_color = GREEN

    cv2.putText(
        frame, label, (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2,
    )

    # ── EAR value ──────────────────────────────────────────────────
    if show_ear:
        ear_text = f"EAR: {ear:.3f}"
        cv2.putText(
            frame, ear_text, (w - 180, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2,
        )

    # ── FPS counter ────────────────────────────────────────────────
    if show_fps:
        fps_text = f"FPS: {fps:.0f}"
        cv2.putText(
            frame, fps_text, (w - 180, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
        )

    # ── Drowsiness progress bar ───────────────────────────────────
    if threshold_frames > 0:
        progress = min(closed_frames / threshold_frames, 1.0)
        bar_x, bar_y, bar_w, bar_h = 10, h - 30, w - 20, 15

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

        # Fill
        fill_w = int(bar_w * progress)
        if progress < 0.5:
            bar_color = GREEN
        elif progress < 0.8:
            bar_color = ORANGE
        else:
            bar_color = RED

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)

        # Label
        bar_label = f"Eye closure: {progress * 100:.0f}%"
        cv2.putText(
            frame, bar_label, (bar_x + 5, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1,
        )

    return frame
