#!/usr/bin/env python3
"""Compare prosody (F0 pitch contour) across TTS outputs.

Extracts fundamental frequency (F0) from WAV files and compares:
- F0 range (min/max of voiced frames)
- F0 standard deviation (higher = more expressive)
- F0 mean
- Voiced frame ratio (how much of the audio is voiced)
- F0 contour shape (visual comparison)

Usage:
    python compare_prosody.py file1.wav file2.wav file3.wav ...
"""

import sys
import struct
import numpy as np
import parselmouth
from parselmouth.praat import call


def read_wav_info(path):
    """Read WAV and return basic info."""
    with open(path, "rb") as f:
        f.read(4)  # RIFF
        f.read(4)  # size
        f.read(4)  # WAVE
        sr = 24000
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                sr = struct.unpack("<I", fmt_data[4:8])[0]
            else:
                f.read(chunk_size)
    return sr


def analyze_f0(wav_path, floor=75, ceiling=500):
    """Extract F0 and compute prosody metrics.

    Args:
        wav_path: Path to WAV file
        floor: Minimum F0 in Hz (75 default for speech)
        ceiling: Maximum F0 in Hz (500 for wide range)

    Returns:
        dict with prosody metrics and raw F0 values
    """
    snd = parselmouth.Sound(wav_path)
    duration = snd.duration

    # Extract pitch using autocorrelation method
    pitch = call(snd, "To Pitch", 0.0, floor, ceiling)

    # Get F0 values at regular intervals
    n_frames = call(pitch, "Get number of frames")
    f0_values = []
    times = []
    for i in range(1, n_frames + 1):
        t = call(pitch, "Get time from frame number", i)
        f0 = call(pitch, "Get value in frame", i, "Hertz")
        times.append(t)
        f0_values.append(f0 if f0 == f0 else 0.0)  # NaN check

    f0_all = np.array(f0_values)
    times = np.array(times)

    # Voiced frames only (F0 > 0)
    voiced_mask = f0_all > 0
    f0_voiced = f0_all[voiced_mask]

    if len(f0_voiced) == 0:
        return {
            "duration": duration,
            "n_frames": n_frames,
            "voiced_ratio": 0.0,
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_min": 0.0,
            "f0_max": 0.0,
            "f0_range": 0.0,
            "f0_range_semitones": 0.0,
            "f0_all": f0_all,
            "f0_voiced": f0_voiced,
            "times": times,
            "voiced_mask": voiced_mask,
        }

    f0_mean = np.mean(f0_voiced)
    f0_std = np.std(f0_voiced)
    f0_min = np.min(f0_voiced)
    f0_max = np.max(f0_voiced)
    f0_range = f0_max - f0_min

    # Range in semitones (perceptually meaningful)
    f0_range_st = 12 * np.log2(f0_max / f0_min) if f0_min > 0 else 0

    # Coefficient of variation (normalized std)
    f0_cv = f0_std / f0_mean if f0_mean > 0 else 0

    return {
        "duration": duration,
        "n_frames": n_frames,
        "voiced_ratio": np.sum(voiced_mask) / len(voiced_mask),
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_cv": f0_cv,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_range": f0_range,
        "f0_range_semitones": f0_range_st,
        "f0_all": f0_all,
        "f0_voiced": f0_voiced,
        "times": times,
        "voiced_mask": voiced_mask,
    }


def print_comparison(results):
    """Print comparison table."""
    # Header
    max_name = max(len(name) for name in results)
    name_w = max(max_name, 10)

    print()
    print("=" * 90)
    print("PROSODY (F0) COMPARISON")
    print("=" * 90)
    print()

    header = (f"{'File':<{name_w}}  {'Dur':>5}  {'Voiced%':>7}  "
              f"{'F0 Mean':>7}  {'F0 Std':>6}  {'F0 CV':>5}  "
              f"{'F0 Min':>6}  {'F0 Max':>6}  {'Range':>6}  {'Range ST':>8}")
    print(header)
    print("-" * len(header))

    for name, m in results.items():
        print(f"{name:<{name_w}}  {m['duration']:>5.2f}  {m['voiced_ratio']*100:>6.1f}%  "
              f"{m['f0_mean']:>7.1f}  {m['f0_std']:>6.1f}  {m['f0_cv']:>5.3f}  "
              f"{m['f0_min']:>6.1f}  {m['f0_max']:>6.1f}  {m['f0_range']:>6.1f}  {m['f0_range_semitones']:>7.1f}st")

    print()
    print("Key metrics for prosody:")
    print("  F0 Std     — pitch variation in Hz (higher = more expressive)")
    print("  F0 CV      — coefficient of variation (std/mean, normalized expressiveness)")
    print("  Range ST   — pitch range in semitones (perceptual units, >6st = expressive)")
    print()

    # Pairwise comparisons
    names = list(results.keys())
    if len(names) >= 2:
        print("Pairwise comparison:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = results[names[i]], results[names[j]]
                std_ratio = a["f0_std"] / b["f0_std"] if b["f0_std"] > 0 else float("inf")
                range_ratio = a["f0_range_semitones"] / b["f0_range_semitones"] if b["f0_range_semitones"] > 0 else float("inf")
                print(f"  {names[i]} vs {names[j]}:")
                print(f"    F0 Std ratio:   {std_ratio:.2f}x  ({'more' if std_ratio > 1 else 'less'} expressive)")
                print(f"    Range ST ratio: {range_ratio:.2f}x")
        print()


def print_contour_ascii(results, width=60):
    """Print ASCII pitch contours for visual comparison."""
    print("F0 CONTOURS (voiced frames only)")
    print("=" * 70)

    for name, m in results.items():
        f0 = m["f0_voiced"]
        if len(f0) == 0:
            print(f"\n{name}: NO VOICED FRAMES")
            continue

        # Resample to fixed width
        indices = np.linspace(0, len(f0) - 1, width).astype(int)
        f0_resampled = f0[indices]

        # Find global min/max for consistent scaling
        f0_min = min(r["f0_min"] for r in results.values() if r["f0_min"] > 0)
        f0_max = max(r["f0_max"] for r in results.values())

        height = 8
        print(f"\n{name} (mean={m['f0_mean']:.0f}Hz, std={m['f0_std']:.1f}Hz, range={m['f0_range_semitones']:.1f}st):")

        for row in range(height - 1, -1, -1):
            row_min = f0_min + (f0_max - f0_min) * row / height
            row_max = f0_min + (f0_max - f0_min) * (row + 1) / height
            label = f"{row_max:>5.0f}|" if row == height - 1 else f"{row_min:>5.0f}|"
            chars = []
            for v in f0_resampled:
                if row_min <= v < row_max:
                    chars.append("█")
                elif row == 0 and v < row_max:
                    chars.append("▄")
                else:
                    chars.append(" ")
            print(f"  {label}{''.join(chars)}|")

        print(f"       {'─' * width}")
    print()


def main():
    files = sys.argv[1:]
    if not files:
        # Default: compare the 4 test files
        files = [
            "/tmp/pytorch_ref_Hello_world,_this_is.wav",
            "/tmp/coreml_hello_test.wav",
            "/tmp/pytorch_xvec_hello.wav",
            "/tmp/coreml_xvec_hello.wav",
        ]
        print("Using default comparison files:")
        for f in files:
            print(f"  {f}")

    results = {}
    for path in files:
        import os
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        name = os.path.basename(path).replace(".wav", "")
        # Shorten names for display
        name = (name.replace("pytorch_ref_Hello_world,_this_is", "pytorch_no_spk")
                    .replace("coreml_hello_test", "coreml_no_spk")
                    .replace("pytorch_xvec_hello", "pytorch_xvec")
                    .replace("coreml_xvec_hello", "coreml_xvec"))
        print(f"Analyzing: {name}...")
        results[name] = analyze_f0(path)

    if not results:
        print("No files to analyze")
        return

    print_comparison(results)
    print_contour_ascii(results)


if __name__ == "__main__":
    main()
