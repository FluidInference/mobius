#!/usr/bin/env python3
"""Test using EXACT official API from Cohere README."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datasets import load_dataset
import torch
import soundfile as sf
import tempfile

print("="*70)
print("Test: Official Cohere API (Exact README Example)")
print("="*70)

# Load model using EXACT method from README
print("\n[1/3] Loading model (official method from README)...")
try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    model_id = "CohereLabs/cohere-transcribe-03-2026"
    device = "cpu"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()

    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
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

    # Method 1: audio_files (from README example 1)
    print("\n[Method 1] Using audio_files (README Example 1 approach)...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
        sf.write(tmpf.name, audio, 16000)
        tmpfile = tmpf.name

    try:
        texts = model.transcribe(
            processor=processor,
            audio_files=[tmpfile],
            language="en"
        )
        hypothesis = texts[0].lower().strip() if texts else None
        print(f"✓ Success!")
    except Exception as e:
        print(f"✗ Failed: {e}")
        hypothesis = None
    finally:
        Path(tmpfile).unlink()

    # Method 2: audio_arrays (from README example 2)
    if hypothesis is None:
        print("\n[Method 2] Using audio_arrays (README Example 2 approach)...")
        try:
            texts = model.transcribe(
                processor=processor,
                audio_arrays=[audio],
                sample_rates=[16000],
                language="en"
            )
            hypothesis = texts[0].lower().strip() if texts else None
            print(f"✓ Success!")
        except Exception as e:
            print(f"✗ Failed: {e}")
            hypothesis = None

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
        print(f"\n⚠️  Both methods failed - official API has bugs")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("""
If this works:
  → Official API is functional
  → If outputs are CORRECT: our manual implementation has bugs
  → If outputs are GARBAGE: confirms model limitation

If this fails:
  → Official API has bugs (processor loading issue)
  → Validates our manual implementation approach
""")
print(f"{'='*70}")
