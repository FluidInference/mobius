#!/usr/bin/env python3
"""Layer-position mixed-precision: first/last 2 conformer layers INT8, middle 20 pal6.

Variant of mixed_quantize_inv.py that adds layer-position awareness on the
linear ops:
  - linear_0..15   (first 2 layers, 8 linears each)        -> INT8  (precision)
  - linear_16..175 (middle 20 layers, 160 ops)             -> pal6  (size win)
  - linear_176..194 (last 2 layers + post-encoder prompt MLP) -> INT8  (precision)
  - conv + matmul (all)                                    -> INT8  (matches mixed-inv)

Hypothesis: first/last layers carry the highest WER sensitivity (entry
features and pre-output projection); protecting them at INT8 may close the
+0.03pp residual gap mixed-inv has vs INT8 while keeping the size+RTFx wins.

Requires iOS18 target.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as ctc


# Option B: shift boundary out to layer 3 to avoid shared-bias conflict.
# 24 conformer layers x 8 linear ops/layer = 192; +3 post-encoder = 195 total
# Option A failed because linear_1 (layer 0, FFN1) shares a bias with linear_17
# (layer 1, FFN2); coremltools refuses configs that differ on shared consts.
# Wider INT8 cuff: first 3 layers + last 3 layers INT8, middle 18 layers pal6.
FIRST_LAYERS_END = 24    # linear_0..23  -> first 3 layers
LAST_LAYERS_START = 168  # linear_168..194 -> last 3 layers + post-encoder prompt MLP
TOTAL_LINEAR = 195


def quantize(src: Path, dst: Path) -> tuple[int, int]:
    mlmodel = ct.models.MLModel(str(src))

    # Stage 1: palettize the middle 20 layers' linear ops
    pal6 = ctc.OpPalettizerConfig(mode="kmeans", nbits=6, granularity="per_tensor")
    middle_names = [f"linear_{i}_cast_fp16" for i in range(FIRST_LAYERS_END, LAST_LAYERS_START)]
    pal_cfg = ctc.OptimizationConfig(
        op_name_configs={name: pal6 for name in middle_names}
    )
    mlmodel = ctc.palettize_weights(mlmodel, config=pal_cfg)

    # Stage 2: INT8 the first+last linear ops (still FP16) + all conv + all matmul
    int8 = ctc.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    edge_names = (
        [f"linear_{i}_cast_fp16" for i in range(FIRST_LAYERS_END)]
        + [f"linear_{i}_cast_fp16" for i in range(LAST_LAYERS_START, TOTAL_LINEAR)]
    )
    q_cfg = ctc.OptimizationConfig(
        op_type_configs={"conv": int8, "matmul": int8},
        op_name_configs={name: int8 for name in edge_names},
    )
    mlmodel = ctc.linear_quantize_weights(mlmodel, config=q_cfg)

    if dst.exists():
        shutil.rmtree(dst)
    mlmodel.save(str(dst))

    def _du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return _du(src), _du(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "tokenizer.json"):
        s = in_dir / name
        if s.exists():
            shutil.copy(s, out_dir / name)
    for c in ["preprocessor", "encoder", "decoder", "joint", "decoder_joint"]:
        src = in_dir / f"{c}.mlpackage"
        dst = out_dir / f"{c}.mlpackage"
        if not src.exists():
            continue
        if c == "encoder":
            print(f"  [layerpos-mixed] {c}.mlpackage ...")
            s_sz, d_sz = quantize(src, dst)
            print(f"           {s_sz/1e6:.1f} MB -> {d_sz/1e6:.1f} MB  ratio {d_sz/s_sz:.2f}")
        else:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  [copy   ] {c}.mlpackage")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
