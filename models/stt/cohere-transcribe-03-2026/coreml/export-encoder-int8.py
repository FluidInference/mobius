#!/usr/bin/env python3
"""Export Cohere encoder with INT8 quantization.

Usage:
    uv run python export-encoder-int8.py --output-dir build-int8
"""

import argparse
import time
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Export Cohere encoder with INT8 quantization")
    parser.add_argument("--output-dir", default="build-int8", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Cohere Encoder INT8 Export")
    print("="*70)
    print(f"Output: {output_dir}")
    print()

    # Step 1: Export FP16 encoder first
    print("[1/3] Exporting FP16 encoder...")
    temp_dir = output_dir / "temp_fp16_encoder"
    temp_dir.mkdir(exist_ok=True)

    # Use standard export script (3500 frames, 35 seconds)
    cmd = [
        "python", "export-encoder.py",
        "--output-dir", str(temp_dir),
        "--precision", "float16"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FP16 export failed:")
        print(result.stderr)
        sys.exit(1)

    print("   ✓ FP16 encoder exported")

    # Step 2: Quantize to INT8
    print("\n[2/3] Quantizing to INT8...")
    import coremltools as ct
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

    # Load FP16 model
    fp16_path = temp_dir / "cohere_encoder.mlpackage"

    print(f"   Loading FP16 model from {fp16_path}...")
    fp16_model = ct.models.MLModel(str(fp16_path))

    # Configure INT8 quantization (wrapped in OptimizationConfig)
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",
    )
    config = OptimizationConfig(global_config=op_config)

    print("   Quantizing weights to INT8 (per-channel, symmetric)...")
    t0 = time.time()
    int8_model = linear_quantize_weights(fp16_model, config=config)
    print(f"   ✓ Quantized in {time.time() - t0:.1f}s")

    # Step 3: Save INT8 model
    print("\n[3/3] Saving INT8 model...")
    int8_path = output_dir / "cohere_encoder_int8.mlpackage"

    int8_model.save(str(int8_path))

    # Calculate sizes
    fp16_size = sum(f.stat().st_size for f in fp16_path.rglob('*') if f.is_file()) / 1024**3
    int8_size = sum(f.stat().st_size for f in int8_path.rglob('*') if f.is_file()) / 1024**3

    print(f"   ✓ Saved to: {int8_path}")
    print(f"   FP16 size: {fp16_size:.2f} GB")
    print(f"   INT8 size: {int8_size:.2f} GB")
    print(f"   Compression: {fp16_size/int8_size:.2f}x")

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    print(f"   ✓ Cleaned up temp directory")

    print("\n" + "="*70)
    print("INT8 ENCODER EXPORT COMPLETE")
    print("="*70)
    print(f"\nOutput: {int8_path}")
    print(f"Size: {int8_size:.2f} GB (was {fp16_size:.2f} GB FP16)")
    print()

if __name__ == "__main__":
    main()
