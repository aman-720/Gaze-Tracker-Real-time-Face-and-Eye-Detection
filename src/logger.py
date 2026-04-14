"""
Session logger for drowsiness detection events.

Logs key events (drowsiness alerts, session stats) to both
console and an optional log file with timestamps.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from src.config import LOGS_DIR


def setup_logger(
    name: str = "drowsiness",
    log_to_file: bool = True,
    log_file: str | None = None,
) -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name.
        log_to_file: Whether to also write to a file.
        log_file: Path to the log file. Defaults to logs/session_<date>.log.

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-init
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if log_file is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = str(LOGS_DIR / f"session_{date_str}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
