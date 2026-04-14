#!/usr/bin/env python3
"""
Generate a simple alarm .wav file using numpy and wave.

This avoids requiring users to source an alarm sound externally.
Creates a dual-tone alarm that alternates between two frequencies.

Usage:
    python scripts/generate_alarm.py
"""

import wave
import struct
import math
from pathlib import Path

ALARM_DIR = Path(__file__).resolve().parent.parent / "assets" / "alarm"
ALARM_FILE = ALARM_DIR / "alarm.wav"

# Audio parameters
SAMPLE_RATE = 44100
DURATION_SEC = 3.0
FREQ_HIGH = 1200  # Hz
FREQ_LOW = 800    # Hz
AMPLITUDE = 0.7
SWITCH_INTERVAL = 0.15  # seconds between tone switches


def generate_alarm():
    """Generate a two-tone alarm sound and save as .wav."""
    ALARM_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = int(SAMPLE_RATE * DURATION_SEC)
    samples = []

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # Alternate between high and low frequency
        cycle = int(t / SWITCH_INTERVAL) % 2
        freq = FREQ_HIGH if cycle == 0 else FREQ_LOW
        value = AMPLITUDE * math.sin(2 * math.pi * freq * t)
        # Apply fade-in/fade-out envelope
        env = min(t / 0.05, 1.0) * min((DURATION_SEC - t) / 0.05, 1.0)
        value *= env
        samples.append(int(value * 32767))

    with wave.open(str(ALARM_FILE), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        for s in samples:
            wav.writeframes(struct.pack("<h", max(-32768, min(32767, s))))

    print(f"Alarm sound generated: {ALARM_FILE}")
    print(f"  Duration: {DURATION_SEC}s, Sample rate: {SAMPLE_RATE} Hz")


if __name__ == "__main__":
    generate_alarm()
