"""LAYERPOS quantization variant with configurable weight_threshold.

Standard LAYERPOS:
  - Middle 18 conformer layers' linears: pal6 (per-tensor 6-bit)
  - First/last 3 layers + all conv + all matmul: INT8 (per-tensor symmetric)
  - Default weight_threshold (coremltools default 2048): smaller weights skipped

This variant lets weight_threshold be tuned to test whether quantizing more
smaller tensors helps (smaller model, more INT8 dispatch) or hurts (extra
dequant per small op).

Usage:
    .venv/bin/python conversion_scripts/mixed_layerpos_threshold.py \\
        --in-dir /path/to/build_fp16_engprune_42_13_4480ms_v3 \\
        --out-dir /path/to/build_lp_engprune_42_13_4480ms_v4_wt512 \\
        --weight-threshold 512
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as ctc

TOTAL_LINEAR = 24  # 24 conformer layers
FIRST_LAYERS_END = 3  # first 3 → INT8 cuff
LAST_LAYERS_START = 21  # last 3 → INT8 cuff


def quantize(src: Path, dst: Path, weight_threshold: int) -> tuple[int, int]:
    mlmodel = ct.models.MLModel(str(src))

    # Stage 1: palettize the middle 18 layers' linear ops at the SAME threshold
    pal6 = ctc.OpPalettizerConfig(
        mode="kmeans", nbits=6, granularity="per_tensor",
        weight_threshold=weight_threshold,
    )
    middle_names = [f"linear_{i}_cast_fp16" for i in range(FIRST_LAYERS_END, LAST_LAYERS_START)]
    pal_cfg = ctc.OptimizationConfig(
        op_name_configs={name: pal6 for name in middle_names}
    )
    mlmodel = ctc.palettize_weights(mlmodel, config=pal_cfg)

    # Stage 2: INT8 the first+last linear ops + all conv + all matmul
    int8 = ctc.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8",
        weight_threshold=weight_threshold,
    )
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
    ap.add_argument("--in-dir", required=True, help="FP16 engprune build dir (source)")
    ap.add_argument("--out-dir", required=True, help="Output build dir for the quantized variant")
    ap.add_argument("--weight-threshold", type=int, default=2048)
    args = ap.parse_args()
    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "tokenizer.json"):
        s = in_dir / name
        if s.exists():
            shutil.copy(s, out_dir / name)
    for c in ["preprocessor", "encoder", "decoder", "joint"]:
        src = in_dir / f"{c}.mlpackage"
        dst = out_dir / f"{c}.mlpackage"
        if not src.exists():
            continue
        if c == "encoder":
            print(f"  [layerpos-mixed wt={args.weight_threshold}] {c}.mlpackage ...")
            s_sz, d_sz = quantize(src, dst, args.weight_threshold)
            print(f"           {s_sz/1e6:.1f} MB -> {d_sz/1e6:.1f} MB  ratio {d_sz/s_sz:.2f}")
        else:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  [copy   ] {c}.mlpackage")


if __name__ == "__main__":
    main()
