#!/usr/bin/env python3
"""Test PyTorch Cohere model end-to-end on long audio."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datasets import load_dataset
import soundfile as sf
import torch

print("="*70)
print("Test: PyTorch Cohere Model on Long Audio")
print("="*70)

# Load PyTorch model
print("\n[1/3] Loading PyTorch model...")
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True
    )
    print("   ✓ PyTorch model loaded")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Get long audio samples
print("\n[2/3] Finding long audio samples (20-23s)...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []

for sample in dataset:
    duration = len(sample['audio']['array']) / 16000.0
    if 20.0 <= duration <= 23.5:
        samples.append(sample)
        print(f"   Found sample {len(samples)}: {duration:.2f}s")
        if len(samples) >= 3:
            break

# Test each sample
print(f"\n[3/3] Testing {len(samples)} samples...")

for idx, sample in enumerate(samples):
    print(f"\n{'='*70}")
    print(f"Sample {idx + 1}/{len(samples)}")
    print(f"{'='*70}")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"\nDuration: {duration:.2f}s")
    print(f"Ground truth: \"{ground_truth[:80]}...\"")

    # Save to temp file
    wav_path = f"/tmp/cohere_test_{idx}.wav"
    sf.write(wav_path, audio, 16000)

    # Transcribe with PyTorch
    with torch.no_grad():
        result = model.transcribe(
            audio_files=[wav_path],
            language="en",
            processor=processor
        )
        hypothesis = result['text'][0].lower().strip()

    print(f"\nPyTorch output: \"{hypothesis[:100]}...\"")

    # Check quality
    gt_start = ground_truth[:50].replace(".", "").replace(",", "").strip()
    hyp_start = hypothesis[:50].replace(".", "").replace(",", "").strip()

    matches = gt_start in hyp_start or hyp_start in gt_start

    if matches:
        print(f"\n✅ CORRECT transcription")
    else:
        print(f"\n❌ INCORRECT transcription (garbage)")
        print(f"   Expected: \"{gt_start}...\"")
        print(f"   Got: \"{hyp_start}...\"")

print(f"\n{'='*70}")
