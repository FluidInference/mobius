#!/usr/bin/env python3
"""Export Cohere stateful decoder with INT8 quantization.

This exports the working stateful decoder and quantizes it to INT8 for smaller size.
Based on export-decoder-stateful.py but adds INT8 quantization step.

Usage:
    uv run python export-stateful-int8.py --output-dir build-int8
"""

import argparse
import time
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Export Cohere stateful decoder with INT8 quantization")
    parser.add_argument("--output-dir", default="build-int8", help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=108, help="Max sequence length")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Cohere Transcribe INT8 Export")
    print("="*70)
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Export FP16 stateful decoder first
    print("[1/3] Exporting FP16 stateful decoder...")
    temp_dir = output_dir / "temp_fp16"
    temp_dir.mkdir(exist_ok=True)

    cmd = [
        "python", "export-decoder-stateful.py",
        "--output-dir", str(temp_dir),
        "--max-seq-len", str(args.max_seq_len),
        "--skip-validation"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FP16 export failed:")
        print(result.stderr)
        sys.exit(1)

    print("   ✓ FP16 model exported")

    # Step 2: Quantize to INT8
    print("\n[2/3] Quantizing to INT8...")
    import coremltools as ct
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

    # Load FP16 model
    if args.max_seq_len == 108:
        fp16_path = temp_dir / "cohere_decoder_stateful.mlpackage"
    else:
        fp16_path = temp_dir / f"cohere_decoder_stateful_{args.max_seq_len}.mlpackage"

    print(f"   Loading FP16 model from {fp16_path}...")
    fp16_model = ct.models.MLModel(str(fp16_path))

    # Configure INT8 quantization (wrapped in OptimizationConfig)
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",  # INT8 symmetric quantization
        dtype="int8",
        granularity="per_channel",  # Better quality than per_tensor
    )
    config = OptimizationConfig(global_config=op_config)

    print("   Quantizing weights to INT8 (per-channel, symmetric)...")
    t0 = time.time()
    int8_model = linear_quantize_weights(fp16_model, config=config)
    print(f"   ✓ Quantized in {time.time() - t0:.1f}s")

    # Step 3: Save INT8 model
    print("\n[3/3] Saving INT8 model...")
    if args.max_seq_len == 108:
        int8_path = output_dir / "cohere_decoder_stateful_int8.mlpackage"
    else:
        int8_path = output_dir / f"cohere_decoder_stateful_{args.max_seq_len}_int8.mlpackage"

    int8_model.save(str(int8_path))

    # Calculate sizes
    fp16_size = sum(f.stat().st_size for f in fp16_path.rglob('*') if f.is_file()) / 1024**2
    int8_size = sum(f.stat().st_size for f in int8_path.rglob('*') if f.is_file()) / 1024**2

    print(f"   ✓ Saved to: {int8_path}")
    print(f"   FP16 size: {fp16_size:.1f} MB")
    print(f"   INT8 size: {int8_size:.1f} MB")
    print(f"   Compression: {fp16_size/int8_size:.2f}x")

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    print(f"   ✓ Cleaned up temp directory")

    print("\n" + "="*70)
    print("INT8 EXPORT COMPLETE")
    print("="*70)
    print(f"\nOutput: {int8_path}")
    print(f"Size: {int8_size:.1f} MB (was {fp16_size:.1f} MB FP16)")
    print(f"\nNext steps:")
    print(f"  1. Test with: python test_int8_stateful.py")
    print(f"  2. If quality is good, upload to HuggingFace")
    print()

if __name__ == "__main__":
    main()
