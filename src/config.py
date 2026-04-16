"""
Configuration module for Driver Drowsiness Detection System.

Central place for all tunable parameters. Adjust these values
to calibrate the system for different cameras, lighting conditions,
and sensitivity requirements.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class DetectionConfig:
    """Face and eye detection parameters."""

    # MediaPipe Face Mesh detection confidence thresholds
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Eye Aspect Ratio (EAR) threshold — below this, eye is "closed"
    # Typical open-eye EAR: 0.25–0.35 | closed-eye EAR: <0.20
    ear_threshold: float = 0.22

    # Number of consecutive frames eyes must be closed to trigger alarm
    # At 30 FPS, 90 frames ≈ 3 seconds
    closed_frames_threshold: int = 90

    # Mouth Aspect Ratio (MAR) threshold — above this, mouth is "open wide" (yawn)
    # Typical closed-mouth MAR: 0.1–0.3 | yawn MAR: >0.6
    mar_threshold: float = 0.6

    # Number of consecutive frames mouth must be open to count as a yawn
    # At 30 FPS, 15 frames ≈ 0.5 seconds (yawns are typically 2–4 seconds)
    yawn_frames_threshold: int = 15

    # Number of yawns within a session window that triggers a yawn warning
    yawn_count_warning: int = 3


@dataclass
class CameraConfig:
    """Video source settings."""

    source: int | str = 0  # 0 = default webcam, or path to video file
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30


@dataclass
class AlarmConfig:
    """Alert system settings."""

    sound_file: str = str(ASSETS_DIR / "alarm" / "alarm.wav")
    alarm_cooldown_sec: float = 5.0  # min seconds between alarm triggers
    visual_alert_color: tuple = (0, 0, 255)  # BGR red


@dataclass
class DisplayConfig:
    """UI overlay settings."""

    show_landmarks: bool = True
    show_ear_value: bool = True
    show_fps: bool = True
    show_status_bar: bool = True
    window_name: str = "Driver Drowsiness Detection"


@dataclass
class LoggingConfig:
    """Logging and recording settings."""

    log_to_file: bool = True
    log_file: str = str(LOGS_DIR / "session.log")
    record_video: bool = False
    output_video: str = str(DATA_DIR / "output.avi")


@dataclass
class AppConfig:
    """Top-level application configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
