#!/usr/bin/env python3
"""
EAR Calibration Tool.

Helps you find the optimal EAR threshold for your face and camera setup.
Shows a live video feed with real-time EAR values while you alternate
between opening and closing your eyes.

Usage:
    python scripts/calibrate_ear.py

Instructions:
    1. Run this script and look at the camera.
    2. Keep your eyes open normally for ~5 seconds — note the EAR values.
    3. Close your eyes for ~5 seconds — note the EAR values.
    4. A good threshold is roughly halfway between the two averages.
    5. Press 'q' to quit and see recommended threshold.
"""

import cv2
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_detector import FaceDetector
from src.eye_tracker import compute_avg_ear
from src.config import DetectionConfig


def main():
    config = DetectionConfig()
    detector = FaceDetector(config)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        sys.exit(1)

    print("\n=== EAR Calibration Tool ===")
    print("Look at the camera. The EAR value is shown on screen.")
    print("Open your eyes normally, then close them, and note the values.")
    print("Press 'q' to quit and see the recommended threshold.\n")

    ear_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        face_detected, left_eye, right_eye, _ = detector.process(frame)

        ear = 0.0
        if face_detected:
            _, _, ear = compute_avg_ear(left_eye, right_eye)
            ear_history.append(ear)

            # Draw eye contours
            cv2.drawContours(frame, [cv2.convexHull(left_eye)],  -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        # Display current EAR
        color = (0, 255, 0) if ear > 0.22 else (0, 0, 255)
        cv2.putText(
            frame, f"EAR: {ear:.3f}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
        )

        if len(ear_history) >= 10:
            avg = np.mean(ear_history[-60:])  # rolling ~2s window at 30fps
            cv2.putText(
                frame, f"Rolling avg: {avg:.3f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )

        cv2.putText(
            frame, "Open eyes normally, then close them. Press 'q' to finish.",
            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )

        cv2.imshow("EAR Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if ear_history:
        print(f"\n=== Calibration Results ===")
        print(f"  Samples:  {len(ear_history)} frames")
        print(f"  Min EAR:  {min(ear_history):.3f}  ← approx closed-eye value")
        print(f"  Max EAR:  {max(ear_history):.3f}  ← approx open-eye value")
        print(f"  Mean EAR: {np.mean(ear_history):.3f}")
        suggested = round((min(ear_history) + np.mean(ear_history)) / 2, 3)
        print(f"\n  Suggested threshold: {suggested}")
        print(f"\n  Run with:  python main.py --ear-threshold {suggested}")
    else:
        print("\nNo EAR data collected — was a face detected?")


if __name__ == "__main__":
    main()
