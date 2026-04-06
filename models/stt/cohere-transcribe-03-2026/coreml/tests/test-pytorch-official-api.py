#!/usr/bin/env python3
"""Test PyTorch model using Cohere's official API (the way they intended)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datasets import load_dataset
import soundfile as sf
import tempfile
import torch

print("="*70)
print("Test: PyTorch Cohere Model - Official API")
print("="*70)

# Load model using official method
print("\n[1/3] Loading model with official API...")
try:
    from transformers import pipeline

    # Use pipeline API (the recommended way)
    pipe = pipeline(
        "automatic-speech-recognition",
        model="CohereLabs/cohere-transcribe-03-2026",
        torch_dtype=torch.float32,
        device="cpu",
        trust_remote_code=True
    )
    print("   ✓ Model loaded via pipeline")
    model = None
    processor = None
except Exception as e:
    print(f"   ❌ Pipeline failed: {e}")
    print("\n   Trying direct model loading...")

    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "CohereLabs/cohere-transcribe-03-2026",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()

        # Load processor (required for transcribe() method)
        try:
            processor = AutoProcessor.from_pretrained(
                "CohereLabs/cohere-transcribe-03-2026",
                trust_remote_code=True
            )
            print("   ✓ Model and processor loaded directly")
        except Exception as proc_err:
            print(f"   ⚠️ Model loaded but processor failed: {proc_err}")
            processor = None

        pipe = None
    except Exception as e2:
        print(f"   ❌ Direct loading also failed: {e2}")
        import traceback
        traceback.print_exc()
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
print(f"\n[3/3] Testing {len(samples)} samples with official API...")

for idx, sample in enumerate(samples):
    print(f"\n{'='*70}")
    print(f"Sample {idx + 1}/{len(samples)}")
    print(f"{'='*70}")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"\nDuration: {duration:.2f}s")
    print(f"Ground truth: \"{ground_truth[:80]}...\"")

    # Method 1: Try pipeline API (if available)
    if pipe is not None:
        print("\nUsing pipeline API...")
        try:
            result = pipe(audio, return_timestamps=False)
            hypothesis = result['text'].lower().strip()
            print(f"✓ Pipeline succeeded")
        except Exception as e:
            print(f"✗ Pipeline failed: {e}")
            hypothesis = None
    else:
        hypothesis = None

    # Method 2: Try model's transcribe() method
    if hypothesis is None and processor is not None:
        print("\nUsing model.transcribe() method with processor...")

        # Save audio to temp file (transcribe() might need file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
            sf.write(tmpf.name, audio, 16000)
            tmpfile = tmpf.name

        try:
            # Try different transcribe() signatures

            # Signature 1: File path with processor
            try:
                result = model.transcribe(
                    audio_files=[tmpfile],
                    language="en",
                    processor=processor
                )
                hypothesis = result['text'][0].lower().strip()
                print(f"✓ transcribe(audio_files=[...], processor=...) succeeded")
            except Exception as e1:
                print(f"✗ transcribe(audio_files=[...], processor=...) failed: {e1}")

                # Signature 2: Audio array with processor
                try:
                    result = model.transcribe(
                        audio_arrays=[audio],
                        language="en",
                        processor=processor
                    )
                    hypothesis = result['text'][0].lower().strip()
                    print(f"✓ transcribe(audio_arrays=[...], processor=...) succeeded")
                except Exception as e2:
                    print(f"✗ transcribe(audio_arrays=[...], processor=...) failed: {e2}")
                    hypothesis = None
        finally:
            Path(tmpfile).unlink()
    elif hypothesis is None:
        print("\nCannot test model.transcribe() - processor not available")

    # Show result
    if hypothesis:
        print(f"\nPyTorch output: \"{hypothesis[:100]}...\"")

        # Check quality
        gt_start = ground_truth[:50].replace(".", "").replace(",", "").strip()
        hyp_start = hypothesis[:50].replace(".", "").replace(",", "").strip()

        matches = gt_start in hyp_start or hyp_start in gt_start

        if matches:
            print(f"\n✅ CORRECT transcription")
        else:
            print(f"\n❌ INCORRECT transcription")
            print(f"   Expected: \"{gt_start}...\"")
            print(f"   Got: \"{hyp_start}...\"")
    else:
        print(f"\n⚠️  Could not get transcription using official API")
        print(f"   All methods failed - this suggests the official API has issues")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("""
If official API works and produces CORRECT transcriptions:
  → Our manual implementation may have bugs
  → We should use the official API instead

If official API works and produces GARBAGE transcriptions:
  → Confirms model limitation
  → Both official and manual implementations fail

If official API doesn't work at all:
  → Validates our manual implementation approach
  → Official API has bugs, manual implementation was necessary
""")
print(f"{'='*70}")
