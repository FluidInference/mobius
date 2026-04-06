#!/usr/bin/env python3
"""Quantize FP16 models to INT8 (W8A16) for smaller size and faster inference."""

import coremltools as ct
from coremltools.optimize.coreml import OptimizationConfig, OpLinearQuantizerConfig
from pathlib import Path
import numpy as np

print("="*70)
print("Quantizing Cohere Models to INT8")
print("="*70)

# Create output directory
output_dir = Path("q8")
output_dir.mkdir(exist_ok=True)

# Create quantization config
config = OptimizationConfig(
    global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=np.int8,
        granularity="per_channel",
        weight_threshold=2048
    )
)

# Quantize encoder
print("\n[1/2] Quantizing encoder...")
print("   Loading FP16 encoder...")
encoder_fp16 = ct.models.MLModel("f16/cohere_encoder.mlpackage")
print("   Quantizing weights to INT8...")
encoder_q8 = ct.optimize.coreml.linear_quantize_weights(encoder_fp16, config)
print("   Saving...")
encoder_q8.save("q8/cohere_encoder.mlpackage")
print(f"   ✓ Saved to: q8/cohere_encoder.mlpackage")

# Get sizes
fp16_size = sum(f.stat().st_size for f in Path("f16/cohere_encoder.mlpackage").rglob('*') if f.is_file()) / 1024**3
q8_size = sum(f.stat().st_size for f in Path("q8/cohere_encoder.mlpackage").rglob('*') if f.is_file()) / 1024**3
print(f"   FP16 size: {fp16_size:.2f} GB")
print(f"   Q8 size:   {q8_size:.2f} GB")
print(f"   Reduction: {(1 - q8_size/fp16_size)*100:.1f}%")

# Quantize decoder
print("\n[2/2] Quantizing decoder...")
print("   Loading FP16 decoder...")
decoder_fp16 = ct.models.MLModel("f16/cohere_decoder_stateful.mlpackage")
print("   Quantizing weights to INT8...")
decoder_q8 = ct.optimize.coreml.linear_quantize_weights(decoder_fp16, config)
print("   Saving...")
decoder_q8.save("q8/cohere_decoder_stateful.mlpackage")
print(f"   ✓ Saved to: q8/cohere_decoder_stateful.mlpackage")

# Get sizes
fp16_size = sum(f.stat().st_size for f in Path("f16/cohere_decoder_stateful.mlpackage").rglob('*') if f.is_file()) / 1024**3
q8_size = sum(f.stat().st_size for f in Path("q8/cohere_decoder_stateful.mlpackage").rglob('*') if f.is_file()) / 1024**3
print(f"   FP16 size: {fp16_size:.2f} GB")
print(f"   Q8 size:   {q8_size:.2f} GB")
print(f"   Reduction: {(1 - q8_size/fp16_size)*100:.1f}%")

print("\n" + "="*70)
print("QUANTIZATION COMPLETE")
print("="*70)
print("\nOutput directory: q8/")
print("Models:")
print("  - cohere_encoder.mlpackage")
print("  - cohere_decoder_stateful.mlpackage")
print()
