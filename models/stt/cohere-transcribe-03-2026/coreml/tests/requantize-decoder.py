"""Re-quantize the FP16 stateful decoder with three strategies and save
each to a separate mlpackage for A/B benchmarking:

  q8_skip_lmhead     -> skip linear_80_cast_fp16 (tied embedding / lm_head)
  q8_per_tensor_lmh  -> per-channel everywhere, per-tensor on lm_head only
  q8_threshold_big   -> weight_threshold=2_000_000 (skip anything <=2M;
                        that skips QKV projections AND lm_head — blunt but
                        simple — useful as a size/quality baseline)

Baseline for comparison: the shipped q8 downloaded from HF (aggressive
per-channel INT8 on everything including lm_head).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
F16_DECODER = ROOT / "hf-upload/f16-download/f16/cohere_decoder_stateful.mlpackage"
OUT_DIR = ROOT / "hf-upload/q8-variants"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# The op that takes the tied embedding and does hidden -> logits:
LM_HEAD_OP_NAME = "linear_80_cast_fp16"
# The gather op that uses the same tied const for input embedding lookup.
# Because the weight is shared, any per-op config override must be consistent
# for BOTH consumers of the const — otherwise coremltools raises
# "compression config conflict detected between ops".
EMBED_GATHER_OP_NAME = "op_341_cast_fp16_cast_uint16"


def variant_skip_lmhead(model) -> ct.models.MLModel:
    print("[variant_skip_lmhead] per-channel INT8 everywhere, skip tied embedding (lm_head + input embed)")
    global_cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=np.int8,
        granularity="per_channel",
        weight_threshold=2048,
    )
    # Tied embedding: one const feeds both gather (input embed) and linear (lm_head).
    # Must skip both consumers so they agree on "don't quantize this const".
    config = OptimizationConfig(
        global_config=global_cfg,
        op_name_configs={
            LM_HEAD_OP_NAME: None,
            EMBED_GATHER_OP_NAME: None,
        },
    )
    return linear_quantize_weights(model, config)


def variant_per_tensor_lmhead(model) -> ct.models.MLModel:
    print("[variant_per_tensor_lmhead] per-channel everywhere, per-tensor on tied embedding")
    global_cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=np.int8,
        granularity="per_channel",
        weight_threshold=2048,
    )
    # Per-tensor on the tied embedding — keeps it INT8 but with a single shared
    # scale so there's no row-to-row scale variance across vocab entries.
    # Must apply to BOTH consumers of the shared const (gather + linear).
    shared_cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=np.int8,
        granularity="per_tensor",
        weight_threshold=2048,
    )
    config = OptimizationConfig(
        global_config=global_cfg,
        op_name_configs={
            LM_HEAD_OP_NAME: shared_cfg,
            EMBED_GATHER_OP_NAME: shared_cfg,
        },
    )
    return linear_quantize_weights(model, config)


def variant_threshold_big(model) -> ct.models.MLModel:
    print("[variant_threshold_big] weight_threshold=2_000_000 (only huge weights)")
    global_cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=np.int8,
        granularity="per_channel",
        weight_threshold=2_000_000,
    )
    # We ALSO explicitly skip the tied embedding (both consumers) even though
    # it passes the threshold, since 16M > 2M.
    config = OptimizationConfig(
        global_config=global_cfg,
        op_name_configs={
            LM_HEAD_OP_NAME: None,
            EMBED_GATHER_OP_NAME: None,
        },
    )
    return linear_quantize_weights(model, config)


def maybe_dir_size_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 ** 3)


def main():
    print(f"Loading FP16 decoder from {F16_DECODER}...")
    t0 = time.time()
    model = ct.models.MLModel(str(F16_DECODER))
    print(f"  loaded in {time.time() - t0:.1f}s")

    variants = {
        "skip_lmhead": variant_skip_lmhead,
        "per_tensor_lmhead": variant_per_tensor_lmhead,
        "threshold_big": variant_threshold_big,
    }

    for name, fn in variants.items():
        out_path = OUT_DIR / f"cohere_decoder_stateful_{name}.mlpackage"
        if out_path.exists():
            print(f"  {name}: already exists at {out_path}, skipping")
            continue
        print(f"\n=== {name} ===")
        t0 = time.time()
        # Reload model each time since linear_quantize_weights may modify it.
        fresh = ct.models.MLModel(str(F16_DECODER))
        q = fn(fresh)
        print(f"  quantized in {time.time() - t0:.1f}s")
        q.save(str(out_path))
        size = maybe_dir_size_gb(out_path)
        print(f"  saved {out_path.name}  {size:.2f} GB")

    print("\nAll variants saved to:", OUT_DIR)
    for p in sorted(OUT_DIR.glob("cohere_decoder_stateful_*.mlpackage")):
        print(f"  {p.name}  {maybe_dir_size_gb(p):.3f} GB")


if __name__ == "__main__":
    main()
