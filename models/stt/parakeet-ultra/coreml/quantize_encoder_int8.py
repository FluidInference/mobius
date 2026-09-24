#!/usr/bin/env python3
"""Data-free int8 linear (per-channel, symmetric) weight quantization of the parakeet-ultra fp16 encoder.

Same encoding as the v3 `Encoder_v2.mlmodelc` (FluidAudio #760): the shipped v3 6-bit LUT flips tokens on some
windows, int8-linear per-channel does not.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fp16", type=Path, help="fp16 encoder .mlpackage")
    ap.add_argument("--out", type=Path, required=True, help="Output .mlpackage (compiled .mlmodelc written beside it)")
    args = ap.parse_args()

    model = ct.models.MLModel(str(args.fp16), skip_model_load=True)
    cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel",
                                                  weight_threshold=2048)
    )
    t0 = time.time()
    q = cto.linear_quantize_weights(model, config=cfg)
    q.short_description = model.short_description.replace("fp16", "int8 linear per-channel")
    q.author = "Fluid Inference"
    q.save(str(args.out))
    print(f"quantized in {time.time() - t0:.0f}s")

    compiled = Path(ct.utils.compile_model(str(args.out)))
    dst = args.out.with_suffix(".mlmodelc")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(compiled), str(dst))
    print(f"mlpackage {dir_size(args.out) / 1e6:.1f} MB, mlmodelc {dir_size(dst) / 1e6:.1f} MB "
          f"(fp16 source {dir_size(args.fp16) / 1e6:.1f} MB) -> {dst}")


if __name__ == "__main__":
    main()
