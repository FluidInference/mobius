#!/usr/bin/env python3
"""Create a simple test audio file for benchmarking."""

import numpy as np
import soundfile as sf
from pathlib import Path

# Create 5 seconds of 16kHz audio with a simple tone pattern
sample_rate = 16000
duration = 5.0
t = np.linspace(0, duration, int(sample_rate * duration))

# Create a simple pattern: 440Hz + 880Hz tones
audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)

# Add some amplitude modulation to make it more interesting
audio *= (1 + 0.3 * np.sin(2 * np.pi * 2 * t))

# Save
output_path = Path("test-audio/synthetic-test.wav")
output_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(output_path, audio, sample_rate)

print(f"Created test audio: {output_path}")
print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz")
