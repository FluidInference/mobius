"""Quantize the FP32 companion encoder to INT8 (per-channel, weight-only).

The cache-external decoder at
  hf-upload/cohere-transcribe-cache-external-coreml/cohere_decoder_cache_external.mlpackage
is paired with
  hf-upload/cohere-transcribe-cache-external-coreml/cohere_encoder.mlpackage   (7.0 GB FP32).

The f16-download (3.6 GB) and q8-download (1.8 GB) encoders on HF are from a
*different export pass* and are not drop-in replacements — pairing them with
the cache-external decoder produces 58%+ WER. So to get the encoder smaller
we have to quantize the companion encoder's own weights in place.

From tests/inspect-encoder-ops.py:
  - 729 weight tensors >= 2048 numel
  - 1.87 B params total (7.48 GB FP32 → 3.74 GB FP16 → 1.87 GB INT8)
  - 0 shared consts — default per-channel config should not hit config-conflict

We produce:
  cohere_encoder_q8.mlpackage   (per-channel INT8 everywhere, weight_threshold=2048)

skip_model_load=True avoids the BNNS compile pass during MLModel() — keeps
disk / memory pressure low, and we don't need a compiled model for the
quantization pass itself (only for predict, which happens later in the bench).
"""
from __future__ import annotations

import shutil
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
FP32_ENCODER = ROOT / "hf-upload/cohere-transcribe-cache-external-coreml/cohere_encoder.mlpackage"
OUT_DIR = ROOT / "build-cache-external-enc-q8"
OUT = OUT_DIR / "cohere_encoder_q8.mlpackage"


def main():
    if not FP32_ENCODER.exists():
        raise SystemExit(f"Missing {FP32_ENCODER}")

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    if OUT.exists():
        print(f"Removing existing {OUT}")
        shutil.rmtree(OUT)

    print(f"Loading FP32 companion encoder (skip_model_load=True): {FP32_ENCODER}")
    t0 = time.time()
    base = ct.models.MLModel(str(FP32_ENCODER), skip_model_load=True)
    print(f"  loaded spec in {time.time() - t0:.1f}s")

    print("Quantizing: per-channel symmetric INT8, weight_threshold=2048")
    cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=np.int8,
        granularity="per_channel",
        weight_threshold=2048,
    )
    config = OptimizationConfig(global_config=cfg)

    t0 = time.time()
    q = linear_quantize_weights(base, config)
    print(f"  quantized in {time.time() - t0:.1f}s")

    t0 = time.time()
    q.save(str(OUT))
    print(f"  saved in {time.time() - t0:.1f}s")

    size_mb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / (1024 * 1024)
    print(f"\n{OUT}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
