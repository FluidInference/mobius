#!/usr/bin/env python3
"""Attempt to compile Q8 models to .mlmodelc format."""

import coremltools as ct
from pathlib import Path

print("="*70)
print("Attempting to Compile Q8 Models to .mlmodelc")
print("="*70)

# Try encoder
print("\n[1/2] Trying encoder...")
try:
    encoder = ct.models.MLModel("q8/cohere_encoder.mlpackage")
    print(f"   Model type: {encoder.get_spec().WhichOneof('Type')}")
    print("   Attempting to save as .mlmodelc...")
    encoder.save("q8/cohere_encoder.mlmodelc")
    print("   ✓ SUCCESS - encoder saved as .mlmodelc")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Try decoder
print("\n[2/2] Trying decoder...")
try:
    decoder = ct.models.MLModel("q8/cohere_decoder_stateful.mlpackage")
    print(f"   Model type: {decoder.get_spec().WhichOneof('Type')}")
    print("   Attempting to save as .mlmodelc...")
    decoder.save("q8/cohere_decoder_stateful.mlmodelc")
    print("   ✓ SUCCESS - decoder saved as .mlmodelc")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

print("\n" + "="*70)
print("COMPILATION ATTEMPT COMPLETE")
print("="*70)
