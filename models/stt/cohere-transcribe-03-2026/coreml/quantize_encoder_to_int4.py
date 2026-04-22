#!/usr/bin/env python3
"""Quantize FP16 encoder to INT4 (4-bit weights)."""

import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil.mil import types
from pathlib import Path

print("=" * 70)
print("Quantizing Cohere Encoder to INT4 (4-bit weights)")
print("=" * 70)
print()

# Load FP16 encoder (iOS 18)
print("Loading FP16 encoder (iOS 18)...")
fp16_encoder = ct.models.MLModel("ios18/cohere_encoder.mlpackage")
print(f"  ✓ Loaded from ios18/cohere_encoder.mlpackage")
print()

# Configure INT4 quantization
print("Configuring INT4 quantization...")
op_config = cto.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype=types.uint4,  # 4-bit unsigned integer
    weight_threshold=512
)

config = cto.OptimizationConfig(global_config=op_config)

print(f"  ✓ Mode: {op_config.mode}")
print(f"  ✓ Dtype: UINT4 ({op_config.nbits} bits)")
print(f"  ✓ Weight threshold: {op_config.weight_threshold}")
print()

# Quantize
print("Quantizing model (this may take a few minutes)...")
quantized_encoder = cto.linear_quantize_weights(fp16_encoder, config)
print("  ✓ Quantization complete")
print()

# Save
output_dir = Path("int4")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "cohere_encoder_int4.mlpackage"

print(f"Saving to {output_path}...")
quantized_encoder.save(str(output_path))
print("  ✓ Saved")
print()

# Compare sizes
import subprocess
fp16_size = subprocess.check_output(["du", "-sh", "ios18/cohere_encoder.mlpackage"]).decode().split()[0]
int4_size = subprocess.check_output(["du", "-sh", str(output_path)]).decode().split()[0]

print("=" * 70)
print("Results")
print("=" * 70)
print(f"FP16 encoder: {fp16_size}")
print(f"INT4 encoder: {int4_size}")
print()
print("Expected size reduction: ~75% (16 bits → 4 bits)")
print()
print("Next: Test with test_int4enc_fp16dec_10_en.py")
