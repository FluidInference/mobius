"""Post-training weight compression for the NeuTTS-2E CoreML LM.

The decode step is memory-bandwidth-bound (every token streams ~470 MB of
fp16 weights), so weight compression is the main speed lever. Produces
variants of the prefill + stateful-decode mlpackages:

    int8  — linear_symmetric per-channel (W8A16)
    int4  — 4-bit palettization (kmeans LUT, per-grouped-channel)

Usage:
    uv run python compress-lm.py --lm-dir ./build/lm-fp16 --output-dir ./build/lm-int8 --mode int8
    uv run python compress-lm.py --lm-dir ./build/lm-fp16 --output-dir ./build/lm-int4 --mode int4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto


def compress(src: Path, dst: Path, mode: str) -> None:
    print(f"[{mode}] {src.name} ...")
    model = ct.models.MLModel(str(src), skip_model_load=True)
    if mode == "int8":
        cfg = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8", granularity="per_channel"
            )
        )
        compressed = cto.coreml.linear_quantize_weights(model, cfg)
    elif mode == "int4":
        cfg = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=16
            )
        )
        compressed = cto.coreml.palettize_weights(model, cfg)
    else:
        raise SystemExit(f"unknown mode {mode}")
    compressed.save(str(dst))
    import subprocess

    size = subprocess.run(["du", "-sh", str(dst)], capture_output=True, text=True).stdout.split()[0]
    print(f"      saved {dst.name} ({size})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lm-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["int8", "int4"], required=True)
    args = p.parse_args()

    lm_dir, out_dir = Path(args.lm_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(lm_dir.glob("*.mlpackage")):
        if "-M2048-fp16.mlpackage" in src.name and "Decode" in src.name:
            continue  # skip the pass-through decode; stateful is the fast path
        compress(src, out_dir / src.name.replace("fp16", args.mode), args.mode)


if __name__ == "__main__":
    main()
