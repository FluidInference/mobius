#!/usr/bin/env python3
"""Compare CoreML encoder vs PyTorch encoder on failing sample.

This determines if weak encoder embeddings are due to CoreML conversion
or if PyTorch encoder also produces weak outputs for these samples.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
import torch

print("="*70)
print("Compare: PyTorch vs CoreML Encoder on Failing Sample")
print("="*70)

# Load CoreML encoder
print("\n[1/4] Loading CoreML encoder...")
coreml_encoder = ct.models.MLModel(
    "build/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
print("   ✓ CoreML encoder loaded")

# Load PyTorch model
print("\n[2/4] Loading PyTorch model...")
try:
    from transformers import AutoModelForSpeechSeq2Seq
    pytorch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    pytorch_model.eval()
    print("   ✓ PyTorch model loaded")
except Exception as e:
    print(f"   ❌ Failed to load PyTorch model: {e}")
    print("   This test requires the PyTorch model")
    exit(1)

# Find the failing sample
print("\n[3/4] Finding failing sample...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
for sample in dataset:
    duration = len(sample['audio']['array']) / 16000.0
    if 23.0 <= duration <= 23.5 and "from the respect paid" in sample['text'].lower():
        break

audio = sample['audio']['array'].astype(np.float32)
ground_truth = sample['text'].lower()
duration = len(audio) / 16000.0

print(f"   ✓ Found sample: {duration:.2f}s")
print(f"   Text: \"{ground_truth[:60]}...\"")

# Process with mel spectrogram
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)
actual_frames = mel.shape[2]

if mel.shape[2] > 3500:
    mel_padded = mel[:, :, :3500]
    actual_frames = 3500
else:
    mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])), mode='constant', constant_values=0)

print(f"   Mel shape: {mel.shape}, padded: {mel_padded.shape}")

# Run CoreML encoder
print("\n[4/4] Comparing encoders...")
print("\n--- CoreML Encoder ---")
coreml_output = coreml_encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([actual_frames], dtype=np.int32)
})

coreml_hidden = None
for key, value in coreml_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3:
        coreml_hidden = value
        break

print(f"Output shape: {coreml_hidden.shape}")
print(f"  Mean: {coreml_hidden.mean():.6f}")
print(f"  Std:  {coreml_hidden.std():.6f}")
print(f"  Min:  {coreml_hidden.min():.6f}")
print(f"  Max:  {coreml_hidden.max():.6f}")

# Run PyTorch encoder
print("\n--- PyTorch Encoder ---")
with torch.no_grad():
    # Convert mel to torch tensor
    mel_torch = torch.from_numpy(mel_padded).float()
    feature_length_torch = torch.tensor([actual_frames], dtype=torch.long)

    # Run encoder
    pytorch_output = pytorch_model.encoder(
        input_features=mel_torch,
        feature_length=feature_length_torch
    )

    # Get hidden states
    if hasattr(pytorch_output, 'last_hidden_state'):
        pytorch_hidden = pytorch_output.last_hidden_state
    else:
        pytorch_hidden = pytorch_output[0]

    # Apply encoder_decoder_proj (1280 → 1024) to match CoreML
    pytorch_hidden = pytorch_model.encoder_decoder_proj(pytorch_hidden)
    pytorch_hidden = pytorch_hidden.numpy()

print(f"Output shape: {pytorch_hidden.shape}")
print(f"  Mean: {pytorch_hidden.mean():.6f}")
print(f"  Std:  {pytorch_hidden.std():.6f}")
print(f"  Min:  {pytorch_hidden.min():.6f}")
print(f"  Max:  {pytorch_hidden.max():.6f}")

# Compare
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")

diff = np.abs(coreml_hidden - pytorch_hidden)
print(f"\nAbsolute difference:")
print(f"  Mean: {diff.mean():.6f}")
print(f"  Max:  {diff.max():.6f}")
print(f"  > 0.1: {(diff > 0.1).sum()} values")
print(f"  > 0.5: {(diff > 0.5).sum()} values")

# Check if both produce weak outputs
coreml_weak = coreml_hidden.std() < 0.4
pytorch_weak = pytorch_hidden.std() < 0.4

print(f"\nEncoder output quality:")
print(f"  CoreML std: {coreml_hidden.std():.6f} {'(WEAK)' if coreml_weak else '(OK)'}")
print(f"  PyTorch std: {pytorch_hidden.std():.6f} {'(WEAK)' if pytorch_weak else '(OK)'}")

if coreml_weak and pytorch_weak:
    print(f"\n✅ Both encoders produce weak outputs for this sample")
    print(f"   → This is a MODEL LIMITATION, not a CoreML conversion issue")
    print(f"   → Certain audio characteristics cause encoder to produce flat embeddings")
elif coreml_weak and not pytorch_weak:
    print(f"\n⚠️  Only CoreML encoder produces weak outputs")
    print(f"   → This IS a CoreML conversion/quantization issue")
    print(f"   → PyTorch encoder produces healthy embeddings")
    print(f"   → Check encoder export precision/quantization")
else:
    print(f"\n✓ Both encoders produce healthy outputs")
    print(f"   → Issue must be elsewhere")

print(f"\n{'='*70}")
