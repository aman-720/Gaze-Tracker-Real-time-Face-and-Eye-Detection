"""
Face detection and landmark extraction using the MediaPipe Tasks API.

MediaPipe 0.10+ removed the legacy `mp.solutions` namespace entirely in favour
of the Tasks API. This module uses `mp.tasks.vision.FaceLandmarker`, which:
  - Requires a small .task model file (~1.5 MB, auto-downloaded on first run)
  - Returns 478 face-mesh landmarks (468 face + 10 iris points)
  - Runs in VIDEO mode for smooth temporal tracking
  - Works on Python 3.13, Apple Silicon, and all modern platforms

Eye landmark indices (MediaPipe 478-point convention — same as the old 468):
  Right eye (from camera): 33, 160, 158, 133, 153, 144
  Left eye  (from camera): 362, 385, 387, 263, 373, 380

These 6 indices per eye are ordered for the EAR formula:
    p0 = outer corner  p1 = upper-outer  p2 = upper-inner
    p3 = inner corner  p4 = lower-inner  p5 = lower-outer

Mouth landmark indices (MediaPipe 478-point convention):
  Inner lips used for yawn detection — vertical and horizontal extents.
  Top: 13  Bottom: 14  Left corner: 78  Right corner: 308
  Upper inner: 82, 312  Lower inner: 87, 317
"""

import time
import urllib.request
import sys

import cv2
import numpy as np
import mediapipe as mp

from src.config import DetectionConfig, ASSETS_DIR


# ── Model setup ────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = ASSETS_DIR / "face_landmarker.task"


def _ensure_model() -> str:
    """Download the FaceLandmarker .task model if not already present."""
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)

    print(f"Downloading MediaPipe face landmark model (~1.5 MB)...")
    print(f"  Source: {_MODEL_URL}")

    def _progress(block, block_size, total):
        downloaded = block * block_size
        if total > 0:
            pct = min(downloaded * 100 / total, 100)
            sys.stdout.write(f"\r  {pct:4.0f}% ({downloaded / 1024:.0f} KB / {total / 1024:.0f} KB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH), _progress)
        print(f"\nSaved to: {_MODEL_PATH}\n")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download face landmark model: {e}\n"
            f"Please download manually from:\n  {_MODEL_URL}\n"
            f"and place it at:\n  {_MODEL_PATH}"
        ) from e

    return str(_MODEL_PATH)


# ── Eye landmark indices ───────────────────────────────────────────
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

# Inner lip landmarks for Mouth Aspect Ratio (MAR) — yawn detection
# Ordered: top, upper-left, upper-right, bottom, lower-left, lower-right,
#          left corner, right corner
MOUTH_IDX = [13, 82, 312, 14, 87, 317, 78, 308]


class FaceDetector:
    """Detects faces and extracts eye landmarks using MediaPipe Tasks API."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()

        model_path = _ensure_model()

        BaseOptions          = mp.tasks.BaseOptions
        FaceLandmarker       = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode    = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_presence_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray):
        """Run full detection on one frame.

        Args:
            frame: BGR image from OpenCV (H x W x 3, uint8).

        Returns:
            Tuple of:
              - face_detected (bool)
              - left_eye  (6, 2) int32 array or None
              - right_eye (6, 2) int32 array or None
              - face_rect dict {"x1","y1","x2","y2"} in pixels, or None
              - mouth (8, 2) int32 array or None
        """
        h, w = frame.shape[:2]

        # MediaPipe Tasks expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires a strictly monotonically increasing timestamp (ms)
        timestamp_ms = int(time.monotonic() * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return False, None, None, None, None

        # Use first detected face (max_num_faces=1 anyway)
        lm = result.face_landmarks[0]

        # Denormalise landmark coordinates → pixel space
        pts = np.array(
            [(p.x * w, p.y * h) for p in lm],
            dtype=np.float32,
        )

        left_eye  = pts[LEFT_EYE_IDX].astype(np.int32)
        right_eye = pts[RIGHT_EYE_IDX].astype(np.int32)
        mouth     = pts[MOUTH_IDX].astype(np.int32)

        # Loose face bounding box from all mesh points
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        face_rect = {
            "x1": max(0, int(x_min)),
            "y1": max(0, int(y_min)),
            "x2": min(w, int(x_max)),
            "y2": min(h, int(y_max)),
        }

        return True, left_eye, right_eye, face_rect, mouth

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()
