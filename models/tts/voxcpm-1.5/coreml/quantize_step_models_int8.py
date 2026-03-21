#!/usr/bin/env python3
"""Quantize existing FP16 step .mlpackage models to INT8.

Usage:
    cd ~/.cache/fluidaudio/models/voxcpm-1.5
    python /path/to/quantize_step_models_int8.py

This reads the FP16 .mlpackage files in the current directory, applies
INT8 linear-symmetric weight quantization, saves _int8.mlpackage variants,
and compiles them to .mlmodelc.

Models quantized:
  - base_lm_step
  - residual_lm_step
  - locdit_step
  - feat_encoder
"""

import gc
import os
import subprocess
import sys
import time


MODELS = [
    "base_lm_step",
    "residual_lm_step",
    "locdit_step",
    "feat_encoder",
]


def quantize_model(name: str):
    """Quantize a single model from FP16 to INT8."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    fp16_path = f"{name}.mlpackage"
    int8_path = f"{name}_int8.mlpackage"

    if not os.path.exists(fp16_path):
        print(f"  SKIP: {fp16_path} not found")
        return False

    print(f"  Loading {fp16_path}...")
    mlmodel = ct.models.MLModel(fp16_path, skip_model_load=True)

    print(f"  Quantizing to INT8...")
    t0 = time.time()
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config = OptimizationConfig(global_config=op_config)
    mlmodel_int8 = linear_quantize_weights(mlmodel, config=config)
    print(f"  Quantization took {time.time() - t0:.1f}s")

    del mlmodel
    gc.collect()

    mlmodel_int8.save(int8_path)
    print(f"  Saved {int8_path}")

    # Compile to .mlmodelc
    print(f"  Compiling to .mlmodelc...")
    compiled_path = f"{name}_int8.mlmodelc"
    if os.path.exists(compiled_path):
        subprocess.run(["rm", "-rf", compiled_path], check=True)

    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", int8_path, "."],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Compile failed: {result.stderr}")
        return False

    print(f"  Compiled {compiled_path}")

    # Show size comparison
    fp16_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fnames in os.walk(f"{name}.mlmodelc")
        for f in fnames
    ) if os.path.exists(f"{name}.mlmodelc") else 0
    int8_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fnames in os.walk(compiled_path)
        for f in fnames
    )
    print(f"  Size: FP16={fp16_size / 1e6:.0f}MB -> INT8={int8_size / 1e6:.0f}MB "
          f"({int8_size / fp16_size * 100:.0f}%)" if fp16_size > 0 else
          f"  Size: INT8={int8_size / 1e6:.0f}MB")

    del mlmodel_int8
    gc.collect()
    return True


def main():
    print("=== INT8 Quantization of VoxCPM Step Models ===\n")

    for name in MODELS:
        print(f"\n[{MODELS.index(name) + 1}/{len(MODELS)}] {name}")
        quantize_model(name)

    print("\n=== Done ===")
    print("\nTo use INT8 models, replace the .mlmodelc directories:")
    for name in MODELS:
        print(f"  mv {name}.mlmodelc {name}_fp16.mlmodelc && mv {name}_int8.mlmodelc {name}.mlmodelc")


if __name__ == "__main__":
    main()
