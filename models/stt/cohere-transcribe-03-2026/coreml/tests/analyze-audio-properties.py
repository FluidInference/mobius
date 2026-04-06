#!/usr/bin/env python3
"""Analyze audio properties of working vs failing samples."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
import librosa

print("="*70)
print("Audio Properties Analysis: Working vs Failing Samples")
print("="*70)

# Load encoder
print("\n[1/3] Loading encoder...")
encoder = ct.models.MLModel(
    "build/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
mel_processor = CohereMelSpectrogram()
print("   ✓ Encoder loaded")

# Find specific samples we know about
print("\n[2/3] Finding samples...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

# Known working and failing samples
targets = [
    ("Working", 19.5, 20.5, "for general service"),
    ("Failing 1", 23.0, 23.5, "from the respect paid"),
    ("Failing 2", 23.0, 23.5, "thus saying and pressing"),
    ("Failing 3", 22.0, 23.0, "just then leocadia"),
]

samples = {}
for label, min_dur, max_dur, text_snippet in targets:
    for sample in dataset:
        duration = len(sample['audio']['array']) / 16000.0
        if min_dur <= duration <= max_dur and text_snippet in sample['text'].lower():
            samples[label] = sample
            print(f"   ✓ Found {label}: {duration:.2f}s")
            break
    # Reset
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

print(f"\n[3/3] Analyzing {len(samples)} samples...")

def analyze_audio(audio, sr=16000):
    """Extract audio features."""
    # Temporal features
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Pitch/fundamental frequency
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    # Energy distribution
    stft = np.abs(librosa.stft(audio))
    low_energy = np.mean(stft[0:stft.shape[0]//4, :])  # Low frequencies
    mid_energy = np.mean(stft[stft.shape[0]//4:3*stft.shape[0]//4, :])
    high_energy = np.mean(stft[3*stft.shape[0]//4:, :])

    return {
        'duration': len(audio) / sr,
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
        'spec_cent_mean': float(np.mean(spec_cent)),
        'spec_cent_std': float(np.std(spec_cent)),
        'spec_bw_mean': float(np.mean(spec_bw)),
        'spec_rolloff_mean': float(np.mean(spec_rolloff)),
        'mfcc_means': [float(np.mean(mfcc)) for mfcc in mfccs[:5]],
        'mfcc_stds': [float(np.std(mfcc)) for mfcc in mfccs[:5]],
        'pitch_mean': float(np.mean(pitch_values)) if pitch_values else 0.0,
        'pitch_std': float(np.std(pitch_values)) if pitch_values else 0.0,
        'low_energy': float(low_energy),
        'mid_energy': float(mid_energy),
        'high_energy': float(high_energy),
        'energy_ratio': float(high_energy / (low_energy + 1e-10)),
    }

results = {}

for label, sample in samples.items():
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")

    audio = sample['audio']['array'].astype(np.float32)

    # Audio analysis
    audio_features = analyze_audio(audio)

    # Encoder analysis
    mel = mel_processor(audio)
    if mel.shape[2] > 3001:
        mel_padded = mel[:, :, :3001]
        actual_frames = 3001
    else:
        mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant', constant_values=0)
        actual_frames = mel.shape[2]

    encoder_output = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([actual_frames], dtype=np.int32)
    })

    encoder_hidden = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden = value
            break

    results[label] = {
        'audio': audio_features,
        'encoder_std': float(encoder_hidden.std()),
        'encoder_max': float(encoder_hidden.max()),
        'encoder_mean': float(encoder_hidden.mean()),
        'quality': 'WEAK' if encoder_hidden.std() < 0.4 else 'GOOD',
    }

    print(f"\nAudio Properties:")
    print(f"  Duration: {audio_features['duration']:.2f}s")
    print(f"  RMS (volume): mean={audio_features['rms_mean']:.4f}, std={audio_features['rms_std']:.4f}")
    print(f"  Zero-crossing rate: mean={audio_features['zcr_mean']:.4f}")
    print(f"  Spectral centroid: {audio_features['spec_cent_mean']:.1f} Hz")
    print(f"  Spectral bandwidth: {audio_features['spec_bw_mean']:.1f} Hz")
    print(f"  Spectral rolloff: {audio_features['spec_rolloff_mean']:.1f} Hz")
    print(f"  Pitch: mean={audio_features['pitch_mean']:.1f} Hz, std={audio_features['pitch_std']:.1f}")
    print(f"  Energy distribution:")
    print(f"    Low: {audio_features['low_energy']:.4f}")
    print(f"    Mid: {audio_features['mid_energy']:.4f}")
    print(f"    High: {audio_features['high_energy']:.4f}")
    print(f"    High/Low ratio: {audio_features['energy_ratio']:.2f}")

    print(f"\nEncoder Output:")
    print(f"  Std: {results[label]['encoder_std']:.6f} ({results[label]['quality']})")
    print(f"  Max: {results[label]['encoder_max']:.6f}")

# Compare working vs failing
print(f"\n{'='*70}")
print("COMPARISON: Working vs Failing")
print(f"{'='*70}")

if "Working" in results:
    working = results["Working"]
    failing_samples = [v for k, v in results.items() if k.startswith("Failing")]

    print(f"\nWorking sample:")
    print(f"  Encoder std: {working['encoder_std']:.6f}")
    print(f"  RMS: {working['audio']['rms_mean']:.4f}")
    print(f"  Pitch: {working['audio']['pitch_mean']:.1f} Hz")
    print(f"  Spectral centroid: {working['audio']['spec_cent_mean']:.1f} Hz")
    print(f"  Energy ratio (high/low): {working['audio']['energy_ratio']:.2f}")

    if failing_samples:
        print(f"\nFailing samples (average of {len(failing_samples)}):")
        avg_encoder_std = np.mean([f['encoder_std'] for f in failing_samples])
        avg_rms = np.mean([f['audio']['rms_mean'] for f in failing_samples])
        avg_pitch = np.mean([f['audio']['pitch_mean'] for f in failing_samples])
        avg_spec_cent = np.mean([f['audio']['spec_cent_mean'] for f in failing_samples])
        avg_energy_ratio = np.mean([f['audio']['energy_ratio'] for f in failing_samples])

        print(f"  Encoder std: {avg_encoder_std:.6f}")
        print(f"  RMS: {avg_rms:.4f}")
        print(f"  Pitch: {avg_pitch:.1f} Hz")
        print(f"  Spectral centroid: {avg_spec_cent:.1f} Hz")
        print(f"  Energy ratio (high/low): {avg_energy_ratio:.2f}")

        print(f"\nKey differences:")
        rms_diff = ((avg_rms - working['audio']['rms_mean']) / working['audio']['rms_mean']) * 100
        pitch_diff = ((avg_pitch - working['audio']['pitch_mean']) / working['audio']['pitch_mean']) * 100
        spec_diff = ((avg_spec_cent - working['audio']['spec_cent_mean']) / working['audio']['spec_cent_mean']) * 100
        energy_diff = ((avg_energy_ratio - working['audio']['energy_ratio']) / working['audio']['energy_ratio']) * 100

        print(f"  RMS: {rms_diff:+.1f}% {'(quieter)' if rms_diff < 0 else '(louder)'}")
        print(f"  Pitch: {pitch_diff:+.1f}% {'(lower)' if pitch_diff < 0 else '(higher)'}")
        print(f"  Spectral centroid: {spec_diff:+.1f}% {'(darker)' if spec_diff < 0 else '(brighter)'}")
        print(f"  Energy ratio: {energy_diff:+.1f}% {'(less high-freq)' if energy_diff < 0 else '(more high-freq)'}")

print(f"\n{'='*70}")
