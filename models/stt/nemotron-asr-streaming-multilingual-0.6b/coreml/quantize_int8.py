#!/usr/bin/env python3
"""
Weight-only int8 quantization of the multilingual Nemotron CoreML pipeline.

Uses `coremltools.optimize.coreml.linear_quantize_weights` with the
default per-channel linear-symmetric scheme:
  - mode: linear_symmetric
  - dtype: int8
  - granularity: PER_CHANNEL
  - weight_threshold: 2048 (skip tiny weights; not worth the overhead)

Activations stay fp16/fp32; only the on-disk weight const ops are
replaced with `constexpr_affine_dequantize` ops that emit fp16 at
runtime. ANE residency is preserved.

Usage:
    cd conversion_scripts && .venv/bin/python ../quantize_int8.py \
        --in-dir ../build_fp16 --out-dir ../build_int8
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as ctc


def quantize_one(src: Path, dst: Path) -> tuple[int, int]:
    """Quantize a single mlpackage. Returns (src_bytes, dst_bytes)."""
    mlmodel = ct.models.MLModel(str(src))
    config = ctc.OptimizationConfig(
        global_config=ctc.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
        )
    )
    mlmodel_q = ctc.linear_quantize_weights(mlmodel, config=config)
    if dst.exists():
        shutil.rmtree(dst)
    mlmodel_q.save(str(dst))

    def _du(p: Path) -> int:
        total = 0
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    return _du(src), _du(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="build_fp16")
    ap.add_argument("--out-dir", default="build_int8")
    ap.add_argument(
        "--components",
        default="encoder",
        help="Comma list: encoder,preprocessor,decoder,joint or 'all'",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.components == "all":
        components = ["encoder", "preprocessor", "decoder", "joint"]
    else:
        components = [c.strip() for c in args.components.split(",") if c.strip()]

    # Copy metadata + tokenizer through untouched
    for name in ("metadata.json", "tokenizer.json"):
        src = in_dir / name
        if src.exists():
            shutil.copy(src, out_dir / name)

    # Pass-through any component we don't quantize
    all_components = ["encoder", "preprocessor", "decoder", "joint"]
    for name in all_components:
        if name in components:
            continue
        src = in_dir / f"{name}.mlpackage"
        dst = out_dir / f"{name}.mlpackage"
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  [copy ] {name}.mlpackage (not quantized)")

    print(f"Quantizing: {components}")
    total_src = 0
    total_dst = 0
    for name in components:
        src = in_dir / f"{name}.mlpackage"
        dst = out_dir / f"{name}.mlpackage"
        if not src.exists():
            print(f"  [skip ] {name}: source missing")
            continue
        print(f"  [int8 ] {name}.mlpackage ...")
        sb, db = quantize_one(src, dst)
        total_src += sb
        total_dst += db
        ratio = (db / sb) if sb else 0.0
        print(f"          {sb/1e6:.1f} MB -> {db/1e6:.1f} MB  (ratio {ratio:.2f})")

    if total_src:
        print(
            f"Total (quantized only): {total_src/1e6:.1f} MB -> "
            f"{total_dst/1e6:.1f} MB  (ratio {total_dst/total_src:.2f})"
        )
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
