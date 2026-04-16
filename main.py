"""
Driver Drowsiness Detection System — Main Entry Point

Real-time detection pipeline:
  1. Capture frame from webcam or video file
  2. Detect face using MediaPipe Face Mesh
  3. Extract eye landmarks (6 pts/eye) and mouth landmarks (8 pts)
  4. Compute Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
  5. Track consecutive closed-eye frames and yawn events
  6. Trigger alarm if eyes closed > 3 seconds; warn on repeated yawns
  7. Display annotated video with status overlay

Usage:
    python main.py                        # webcam (default)
    python main.py --source video.mp4     # video file
    python main.py --ear-threshold 0.25   # custom EAR threshold
    python main.py --no-display           # headless mode (alarm only)
"""

import argparse
import time
import sys

import cv2

from src.config import AppConfig
from src.face_detector import FaceDetector
from src.eye_tracker import compute_avg_ear
from src.mouth_tracker import compute_mar
from src.drowsiness_tracker import DrowsinessTracker, DriverState
from src.alarm import AlarmSystem
from src.visualizer import draw_overlay
from src.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time Driver Drowsiness Detection System"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: 0 for webcam, or path to video file",
    )
    parser.add_argument(
        "--ear-threshold", type=float, default=0.22,
        help="EAR threshold below which eyes are considered closed (default: 0.22)",
    )
    parser.add_argument(
        "--closed-time", type=float, default=3.0,
        help="Seconds of continuous eye closure to trigger alarm (default: 3.0)",
    )
    parser.add_argument(
        "--mar-threshold", type=float, default=0.6,
        help="MAR threshold above which mouth is considered open/yawning (default: 0.6)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without GUI window (alarm-only mode)",
    )
    parser.add_argument(
        "--no-alarm", action="store_true",
        help="Disable audio alarm (visual alert only)",
    )
    parser.add_argument(
        "--record", type=str, default=None,
        help="Path to save output video (e.g., output.avi)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Configuration ──────────────────────────────────────────────
    config = AppConfig()
    config.detection.ear_threshold = args.ear_threshold
    config.detection.mar_threshold = args.mar_threshold

    # Parse video source
    source = int(args.source) if args.source.isdigit() else args.source

    # Calculate frame threshold from time and estimated FPS
    estimated_fps = config.camera.fps
    closed_frames_threshold = int(args.closed_time * estimated_fps)
    config.detection.closed_frames_threshold = closed_frames_threshold

    # ── Initialize components ──────────────────────────────────────
    logger = setup_logger()
    logger.info("Starting Driver Drowsiness Detection System")
    logger.info(f"Source: {source}")
    logger.info(f"EAR threshold: {args.ear_threshold}")
    logger.info(f"MAR threshold: {args.mar_threshold}")
    logger.info(f"Closed-eye time trigger: {args.closed_time}s ({closed_frames_threshold} frames)")

    detector = FaceDetector(config.detection)

    tracker = DrowsinessTracker(
        ear_threshold=config.detection.ear_threshold,
        closed_frames_threshold=closed_frames_threshold,
        mar_threshold=config.detection.mar_threshold,
        yawn_frames_threshold=config.detection.yawn_frames_threshold,
        yawn_count_warning=config.detection.yawn_count_warning,
    )

    alarm = None
    if not args.no_alarm:
        alarm = AlarmSystem(
            sound_file=config.alarm.sound_file,
            cooldown_sec=config.alarm.alarm_cooldown_sec,
        )

    # ── Video capture ──────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.frame_height)

    # ── Video writer (optional) ────────────────────────────────────
    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(
            args.record, fourcc, estimated_fps,
            (config.camera.frame_width, config.camera.frame_height),
        )
        logger.info(f"Recording to: {args.record}")

    # ── Main loop ──────────────────────────────────────────────────
    logger.info("Detection running. Press 'q' to quit.")
    fps_timer = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and source != "0":
                    logger.info("End of video file.")
                else:
                    logger.error("Failed to read frame from camera.")
                break

            frame = cv2.resize(
                frame,
                (config.camera.frame_width, config.camera.frame_height),
            )

            # ── Detection pipeline ─────────────────────────────────
            face_detected, left_eye, right_eye, face_rect, mouth = detector.process(frame)

            ear = 0.0
            mar = 0.0
            if face_detected:
                _, _, ear = compute_avg_ear(left_eye, right_eye)
                mar = compute_mar(mouth)

            # ── State update ───────────────────────────────────────
            state = tracker.update(ear, face_detected, mar)

            # ── Alarm logic ────────────────────────────────────────
            if state == DriverState.DROWSY:
                if alarm:
                    alarm.trigger()
                logger.warning(
                    f"DROWSY ALERT! Eyes closed for "
                    f"{tracker.get_closure_duration_sec(fps if fps > 0 else estimated_fps):.1f}s"
                )

            if state == DriverState.YAWNING:
                logger.info(f"Yawn detected (total: {tracker.total_yawns})")

            if tracker.yawn_warning:
                logger.warning(f"FATIGUE WARNING: {tracker.total_yawns} yawns detected")

            # ── Visualization ──────────────────────────────────────
            if not args.no_display:
                frame = draw_overlay(
                    frame=frame,
                    state=state,
                    ear=ear,
                    fps=fps,
                    closed_frames=tracker.closed_frame_count,
                    threshold_frames=closed_frames_threshold,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    face_rect=face_rect,
                    mouth=mouth,
                    mar=mar,
                    yawn_count=tracker.total_yawns,
                    yawn_warning=tracker.yawn_warning,
                    show_landmarks=config.display.show_landmarks,
                    show_ear=config.display.show_ear_value,
                    show_fps=config.display.show_fps,
                )
                cv2.imshow(config.display.window_name, frame)

            if writer:
                writer.write(frame)

            # ── FPS calculation ────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # ── Exit check ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or Escape
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        # ── Cleanup ────────────────────────────────────────────────
        logger.info("Shutting down...")
        summary = tracker.get_summary()
        logger.info(
            f"Session summary: {summary['total_drowsiness_events']} drowsiness events, "
            f"{summary['total_blinks']} blinks, {summary['total_yawns']} yawns detected"
        )
        for i, evt in enumerate(summary["events"], 1):
            logger.info(f"  Event {i}: {evt['duration_sec']}s")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        detector.close()
        if alarm:
            alarm.stop()
            alarm.cleanup()

    logger.info("Session ended.")


if __name__ == "__main__":
    main()
