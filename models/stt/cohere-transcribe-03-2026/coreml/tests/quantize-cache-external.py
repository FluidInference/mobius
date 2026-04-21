"""Quantize the FP16 cache-external decoder to INT8 (per-channel, weight-only).

The stateful decoder uploaded to HF has a systemic over-generation bug —
both f16 and q8 variants hallucinate continuations past EOS. The
cache-external decoder at hf-upload/cohere-transcribe-cache-external-coreml/
is the one that actually lands at EN 10.6% / ES 4.9% / FR 16.8% / ZH 14.1%
WER on the 12-sample FLEURS slice, so we want a q8 version of *that*
decoder, not the stateful one.

Op names discovered via inspection of the MIL program:
  lm_head op     : linear_80_cast_fp16
  input embed op : var_339_cast_fp16_cast_uint16
Both consume the tied const embedding_token_embedding_weight_to_fp16, so
any per-op config override must agree across both or coremltools raises
"compression config conflict detected between ops".

We produce two variants:
  q8                 -> per-channel everywhere, tied embedding included
  q8_per_tensor_lmh  -> per-channel everywhere, per-tensor on tied embedding
                        (the variant that worked best for the stateful decoder)
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
F16_DECODER = ROOT / "build-test/cohere_decoder_cache_external.mlpackage"
OUT_DIR = ROOT / "build-cache-external-q8"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Cache-external specific names (differ from stateful decoder):
LM_HEAD_OP_NAME = "linear_80_cast_fp16"
EMBED_GATHER_OP_NAME = "var_339_cast_fp16_cast_uint16"


def variant_q8(model) -> ct.models.MLModel:
    print("[q8] per-channel INT8 everywhere (including tied embedding)")
    cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=np.int8,
        granularity="per_channel",
        weight_threshold=2048,
    )
    config = OptimizationConfig(global_config=cfg)
    return linear_quantize_weights(model, config)


def variant_q8_per_tensor_lmh(model) -> ct.models.MLModel:
    print("[q8_per_tensor_lmh] per-channel everywhere, per-tensor on tied embedding")
    global_cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=np.int8,
        granularity="per_channel",
        weight_threshold=2048,
    )
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


def main():
    if not F16_DECODER.exists():
        raise SystemExit(f"Missing {F16_DECODER}")

    print(f"Loading base FP16 cache-external decoder: {F16_DECODER}")
    t0 = time.time()
    base = ct.models.MLModel(str(F16_DECODER))
    print(f"  loaded in {time.time() - t0:.1f}s")

    variants = [
        ("q8", variant_q8),
        ("q8_per_tensor_lmh", variant_q8_per_tensor_lmh),
    ]

    for name, fn in variants:
        out = OUT_DIR / f"cohere_decoder_cache_external_{name}.mlpackage"
        print(f"\n--- {name} → {out} ---")
        t0 = time.time()
        model = fn(base)
        print(f"  quantized in {time.time() - t0:.1f}s")
        if out.exists():
            import shutil
            shutil.rmtree(out)
        model.save(str(out))
        size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / (1024 * 1024)
        print(f"  saved ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
