#!/usr/bin/env python3
"""Debug encoder outputs for different audio lengths.

Check if encoder produces degraded outputs for longer audio.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset

print("="*70)
print("Debug: Encoder Output Analysis")
print("="*70)

# Load encoder
print("\n[1/2] Loading encoder...")
encoder = ct.models.MLModel(
    "build/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
print("   ✓ Encoder loaded")

# Get samples of different lengths
print("\n[2/2] Analyzing encoder outputs...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
mel_processor = CohereMelSpectrogram()

samples_to_test = {
    "Short (5s)": (4.5, 5.5),
    "Medium (10s)": (9.5, 10.5),
    "Long (20s)": (19.5, 20.5),
}

results = {}

for label, (min_dur, max_dur) in samples_to_test.items():
    # Find one sample in this range
    for sample in dataset:
        duration = len(sample['audio']['array']) / 16000.0
        if min_dur <= duration <= max_dur:
            audio = sample['audio']['array'].astype(np.float32)

            # Compute mel
            mel = mel_processor(audio)
            if mel.shape[2] > 3500:
                mel_padded = mel[:, :, :3500]
                actual_len = 3500
            else:
                mel_padded = np.pad(
                    mel,
                    ((0, 0), (0, 0), (0, 3500 - mel.shape[2])),
                    mode='constant',
                    constant_values=0
                )
                actual_len = mel.shape[2]

            # Encode
            encoder_output = encoder.predict({
                "input_features": mel_padded.astype(np.float32),
                "feature_length": np.array([actual_len], dtype=np.int32)
            })

            # Extract encoder hidden states
            encoder_hidden = None
            for key, value in encoder_output.items():
                if hasattr(value, 'shape') and len(value.shape) == 3:
                    encoder_hidden = value
                    break

            if encoder_hidden is not None:
                results[label] = {
                    'duration': duration,
                    'mel_frames': actual_len,
                    'encoder_shape': encoder_hidden.shape,
                    'encoder_mean': float(encoder_hidden.mean()),
                    'encoder_std': float(encoder_hidden.std()),
                    'encoder_min': float(encoder_hidden.min()),
                    'encoder_max': float(encoder_hidden.max()),
                    'encoder_has_nan': bool(np.isnan(encoder_hidden).any()),
                    'encoder_has_inf': bool(np.isinf(encoder_hidden).any()),
                }

                print(f"\n{label}:")
                print(f"  Duration: {duration:.2f}s")
                print(f"  Mel frames: {actual_len}")
                print(f"  Encoder shape: {encoder_hidden.shape}")
                print(f"  Encoder stats:")
                print(f"    Mean: {results[label]['encoder_mean']:.6f}")
                print(f"    Std:  {results[label]['encoder_std']:.6f}")
                print(f"    Min:  {results[label]['encoder_min']:.6f}")
                print(f"    Max:  {results[label]['encoder_max']:.6f}")
                print(f"  Has NaN: {results[label]['encoder_has_nan']}")
                print(f"  Has Inf: {results[label]['encoder_has_inf']}")

                # Check encoder attention distribution (first few tokens)
                print(f"  First 5 frame stats:")
                for i in range(min(5, encoder_hidden.shape[1])):
                    frame_mean = encoder_hidden[0, i, :].mean()
                    frame_std = encoder_hidden[0, i, :].std()
                    print(f"    Frame {i}: mean={frame_mean:.6f}, std={frame_std:.6f}")

            break  # Found one sample for this length

    # Reset dataset iterator for next search
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

if len(results) >= 2:
    labels = list(results.keys())
    short = results[labels[0]]
    long = results[labels[-1]]

    print(f"\nComparing {labels[0]} vs {labels[-1]}:")
    print(f"  Mean change: {short['encoder_mean']:.6f} → {long['encoder_mean']:.6f}")
    print(f"  Std change:  {short['encoder_std']:.6f} → {long['encoder_std']:.6f}")

    mean_diff_pct = abs(long['encoder_mean'] - short['encoder_mean']) / abs(short['encoder_mean']) * 100
    std_diff_pct = abs(long['encoder_std'] - short['encoder_std']) / abs(short['encoder_std']) * 100

    print(f"\n  Mean difference: {mean_diff_pct:.1f}%")
    print(f"  Std difference:  {std_diff_pct:.1f}%")

    if mean_diff_pct > 20 or std_diff_pct > 20:
        print(f"\n⚠️  Significant encoder output change for longer audio!")
        print(f"    This could explain decoder quality degradation")
    else:
        print(f"\n✓ Encoder outputs are similar across audio lengths")
        print(f"   Issue is likely in decoder, not encoder")

print(f"\n{'='*70}")
