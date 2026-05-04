#!/usr/bin/env python3
"""
Post-conversion weight quantization for the g2pW mlpackage.

Takes the fp16 baseline `build/g2pw/g2pw.mlpackage` and emits int8
linear-quantized variant(s) under `build/g2pw-int8/`. BERT-base is
~85% linear layer params, so int8 weights cut the on-disk footprint
roughly in half (~303 MB → ~150 MB) with minimal accuracy impact for
classification heads.

Activations stay fp16 (matching the baseline's `compute_precision`).
This is weight-only post-training quantization — no calibration data
needed.

Usage:
    uv run python quantize.py \\
        --in-dir ./build/g2pw \\
        --out-dir ./build/g2pw-int8 \\
        --mode linear-per-channel
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SIDE_FILES = [
    "POLYPHONIC_CHARS.txt",
    "MONOPHONIC_CHARS.txt",
    "config.py",
    "version",
]


def _quantize(in_pkg: Path, out_pkg: Path, mode: str) -> None:
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OpPalettizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
        palettize_weights,
    )

    print(f"[load] {in_pkg}")
    mlmodel = ct.models.MLModel(str(in_pkg))

    print(f"[quantize] mode={mode}")
    if mode == "linear-per-channel":
        op_cfg = OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
        )
        quantized = linear_quantize_weights(
            mlmodel, config=OptimizationConfig(global_config=op_cfg)
        )
    elif mode == "linear-per-tensor":
        op_cfg = OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_tensor",
        )
        quantized = linear_quantize_weights(
            mlmodel, config=OptimizationConfig(global_config=op_cfg)
        )
    elif mode == "palettize-4bit":
        op_cfg = OpPalettizerConfig(nbits=4, mode="kmeans")
        quantized = palettize_weights(
            mlmodel, config=OptimizationConfig(global_config=op_cfg)
        )
    elif mode == "palettize-6bit":
        op_cfg = OpPalettizerConfig(nbits=6, mode="kmeans")
        quantized = palettize_weights(
            mlmodel, config=OptimizationConfig(global_config=op_cfg)
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    quantized.save(str(out_pkg))
    print(f"[write] {out_pkg}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, default=Path("./build/g2pw"))
    p.add_argument("--out-dir", type=Path, default=Path("./build/g2pw-int8"))
    p.add_argument(
        "--mode",
        choices=[
            "linear-per-channel",
            "linear-per-tensor",
            "palettize-6bit",
            "palettize-4bit",
        ],
        default="linear-per-channel",
        help=(
            "linear-per-channel: int8 sym per-channel (~2x shrink, near-lossless)."
            " linear-per-tensor: int8 sym per-tensor (~2x, slightly more drift)."
            " palettize-6bit: 6-bit kmeans palette (~2.7x)."
            " palettize-4bit: 4-bit kmeans palette (~4x, BERT may degrade)."
        ),
    )
    args = p.parse_args()

    in_pkg = args.in_dir / "g2pw.mlpackage"
    if not in_pkg.exists():
        print(f"missing {in_pkg} — run convert-coreml.py first", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pkg = args.out_dir / "g2pw.mlpackage"
    _quantize(in_pkg, out_pkg, args.mode)

    for fname in SIDE_FILES:
        src = args.in_dir / fname
        if src.exists():
            shutil.copy2(src, args.out_dir / fname)
            print(f"[copy] {fname}")

    print("[done] int8 mlpackage ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
