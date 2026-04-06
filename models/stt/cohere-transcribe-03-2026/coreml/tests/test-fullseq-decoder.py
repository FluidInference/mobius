#!/usr/bin/env python3
"""Quick test with fullseq_masked decoder."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram

print("Testing fullseq_masked decoder...")

# Load sample
from datasets import load_dataset
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
sample = next(iter(dataset))
audio = sample['audio']['array'].astype(np.float32)
ground_truth = sample['text'].lower()

print(f"Ground truth: \"{ground_truth}\"")

# Load models
encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
decoder = ct.models.MLModel("barathwaj-models/cohere_decoder_fullseq_masked.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)

# Check decoder inputs
print("\nDecoder inputs:")
spec = decoder.get_spec()
for input_desc in spec.description.input:
    print(f"  {input_desc.name}: {input_desc.type}")

