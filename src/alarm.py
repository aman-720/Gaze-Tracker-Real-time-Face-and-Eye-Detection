"""
Alarm module for drowsiness alerts.

Provides both audio and visual alert capabilities:
  - Audio: plays a .wav alarm sound in a non-blocking thread
  - Visual: returns alert color/text for the UI overlay

The alarm uses a cooldown period to prevent rapid re-triggering.
If no custom alarm sound is found, falls back to a system beep.
"""

import time
import threading
from pathlib import Path

try:
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


class AlarmSystem:
    """Non-blocking audio alarm with cooldown logic."""

    def __init__(
        self,
        sound_file: str | None = None,
        cooldown_sec: float = 5.0,
    ):
        self.sound_file = sound_file
        self.cooldown_sec = cooldown_sec
        self._last_alarm_time: float = 0.0
        self._is_playing: bool = False
        self._initialized = False

        if _PYGAME_AVAILABLE and sound_file and Path(sound_file).exists():
            try:
                pygame.mixer.init()
                self._alarm_sound = pygame.mixer.Sound(sound_file)
                self._initialized = True
            except Exception:
                self._initialized = False

    def trigger(self) -> bool:
        """Trigger the alarm if cooldown has elapsed.

        Returns:
            True if alarm was triggered, False if still in cooldown.
        """
        now = time.time()
        if now - self._last_alarm_time < self.cooldown_sec:
            return False

        self._last_alarm_time = now
        self._play_async()
        return True

    def _play_async(self):
        """Play alarm sound in a background thread."""
        thread = threading.Thread(target=self._play, daemon=True)
        thread.start()

    def _play(self):
        """Play the alarm sound."""
        if self._initialized and _PYGAME_AVAILABLE:
            try:
                self._alarm_sound.play()
                return
            except Exception:
                pass
        # Fallback: terminal beep
        print("\a" * 3, end="", flush=True)

    def stop(self):
        """Stop any currently playing alarm."""
        if self._initialized and _PYGAME_AVAILABLE:
            try:
                pygame.mixer.stop()
            except Exception:
                pass

    def cleanup(self):
        """Release audio resources."""
        if _PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
