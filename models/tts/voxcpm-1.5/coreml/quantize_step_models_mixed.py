#!/usr/bin/env python3
"""Mixed-precision quantization: INT8 bulk + FP16 stop head layers.

Usage:
    cd ~/.cache/fluidaudio/models/voxcpm-1.5
    python /path/to/quantize_step_models_mixed.py

For base_lm_step (172 linear ops):
  - Transformer layers 0-23 (168 ops): INT8
  - Head projections: lm_hidden + fsq + stop_proj + stop_head (4 ops): FP16

  The stop head path (lm_hidden -> stop_proj -> SiLU -> stop_head) is
  sensitive to quantization noise. Keeping the head projections in FP16
  preserves stop-head accuracy while quantizing 98% of weights to INT8.
  (Transformer layers share bias params so they must all be the same dtype.)

For other models (residual_lm_step, locdit_step, feat_encoder):
  - Full INT8 (no stop head to protect)
"""

import gc
import os
import subprocess
import time


VENV_PYTHON = os.path.abspath(__file__).replace(
    "quantize_step_models_mixed.py", ".venv/bin/python"
)

# base_lm_step architecture:
#   24 layers × 7 linears = 168 transformer ops
#   + linear_168 (lm_hidden output proj)
#   + linear_169 (fsq_layer)
#   + linear_170 (stop_proj)
#   + linear_171 (stop_head)
#
# MiniCPM shares bias params across transformer layers, so we can't
# set different quantization for individual transformer layers.
# Keep FP16: only the 4 head projections (ops 168-171) which have
# unique weights/biases not shared with transformer layers.
BASE_LM_FP16_OPS = [f"linear_{i}_cast_fp16" for i in range(168, 172)]


def quantize_base_lm_mixed():
    """Mixed-precision quantization for base_lm_step."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    name = "base_lm_step"
    fp16_path = f"{name}.mlpackage"
    out_path = f"{name}_mixed.mlpackage"

    if not os.path.exists(fp16_path):
        print(f"  SKIP: {fp16_path} not found")
        return False

    print(f"  Loading {fp16_path}...")
    mlmodel = ct.models.MLModel(fp16_path, skip_model_load=True)

    print(f"  Mixed quantization: INT8 global, FP16 for {len(BASE_LM_FP16_OPS)} stop-head ops...")
    t0 = time.time()

    int8_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    # None means "skip quantization" = keep FP16
    fp16_overrides = {op_name: None for op_name in BASE_LM_FP16_OPS}

    config = OptimizationConfig(
        global_config=int8_config,
        op_name_configs=fp16_overrides,
    )
    mlmodel_mixed = linear_quantize_weights(mlmodel, config=config)
    print(f"  Quantization took {time.time() - t0:.1f}s")

    del mlmodel
    gc.collect()

    mlmodel_mixed.save(out_path)
    print(f"  Saved {out_path}")

    # Compile
    print(f"  Compiling to .mlmodelc...")
    compiled_path = f"{name}_mixed.mlmodelc"
    if os.path.exists(compiled_path):
        subprocess.run(["rm", "-rf", compiled_path], check=True)

    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", out_path, "."],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Compile failed: {result.stderr}")
        return False
    print(f"  Compiled {compiled_path}")

    del mlmodel_mixed
    gc.collect()
    return True


def quantize_full_int8(name: str):
    """Full INT8 quantization for models without a stop head."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    fp16_path = f"{name}.mlpackage"
    out_path = f"{name}_mixed.mlpackage"

    if not os.path.exists(fp16_path):
        print(f"  SKIP: {fp16_path} not found")
        return False

    print(f"  Loading {fp16_path}...")
    mlmodel = ct.models.MLModel(fp16_path, skip_model_load=True)

    print(f"  Full INT8 quantization...")
    t0 = time.time()
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config = OptimizationConfig(global_config=op_config)
    mlmodel_int8 = linear_quantize_weights(mlmodel, config=config)
    print(f"  Quantization took {time.time() - t0:.1f}s")

    del mlmodel
    gc.collect()

    mlmodel_int8.save(out_path)
    print(f"  Saved {out_path}")

    # Compile
    print(f"  Compiling to .mlmodelc...")
    compiled_path = f"{name}_mixed.mlmodelc"
    if os.path.exists(compiled_path):
        subprocess.run(["rm", "-rf", compiled_path], check=True)

    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", out_path, "."],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Compile failed: {result.stderr}")
        return False
    print(f"  Compiled {compiled_path}")

    del mlmodel_int8
    gc.collect()
    return True


MODELS = [
    ("base_lm_step", "mixed"),
    ("residual_lm_step", "int8"),
    ("locdit_step", "int8"),
    ("feat_encoder", "int8"),
]


def main():
    print("=== Mixed-Precision Quantization of VoxCPM Step Models ===")
    print("  base_lm_step: INT8 bulk + FP16 head projections (4 ops)")
    print("  others: full INT8\n")

    for i, (name, mode) in enumerate(MODELS):
        print(f"\n[{i + 1}/{len(MODELS)}] {name} ({mode})")
        if mode == "mixed":
            quantize_base_lm_mixed()
        else:
            quantize_full_int8(name)

    print("\n=== Done ===")
    print("\nTo use mixed models, swap the .mlmodelc directories:")
    for name, _ in MODELS:
        print(f"  mv {name}.mlmodelc {name}_old.mlmodelc && mv {name}_mixed.mlmodelc {name}.mlmodelc")


if __name__ == "__main__":
    main()
