#!/usr/bin/env python3
"""Compile encoder .mlpackage to .mlmodelc format."""

import coremltools as ct
from pathlib import Path
import shutil

print("="*70)
print("Compiling Encoder to .mlmodelc")
print("="*70)

# Load the .mlpackage
print("\n[1/3] Loading encoder.mlpackage...")
encoder = ct.models.MLModel("f16/cohere_encoder.mlpackage")
print("   ✓ Loaded")

# Save as .mlmodelc
print("\n[2/3] Compiling to .mlmodelc...")
output_path = "f16/cohere_encoder.mlmodelc"

# Remove existing if present
if Path(output_path).exists():
    shutil.rmtree(output_path)
    print("   ✓ Removed existing .mlmodelc")

encoder.save(output_path)
print(f"   ✓ Saved to {output_path}")

# Verify it loads
print("\n[3/3] Verifying .mlmodelc loads...")
encoder_mlmodelc = ct.models.MLModel(output_path)
print("   ✓ Successfully loaded .mlmodelc")

# Check size
mlpackage_size = sum(f.stat().st_size for f in Path("f16/cohere_encoder.mlpackage").rglob('*') if f.is_file())
mlmodelc_size = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file())

print("\n" + "="*70)
print("COMPILATION COMPLETE")
print("="*70)
print(f"\n.mlpackage size: {mlpackage_size / 1024**3:.2f} GB")
print(f".mlmodelc size:  {mlmodelc_size / 1024**3:.2f} GB")
print(f"\nThe encoder can now be used in .mlmodelc format for instant loading!")
print(f"The decoder MUST remain .mlpackage (State API requirement).")
