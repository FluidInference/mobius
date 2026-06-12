#!/usr/bin/env python3
"""Int8-quantize the parakeet-unified encoders (per-channel linear symmetric).

Same recipe as the nemotron pipeline's int8_quantize_encoder.py. Only the two
encoders are quantized (1.1 GB fp16 each — everything else is < 15 MB). The
output directory mirrors the fp16 layout so compare-models.py / benchmark_wer.py
can point at it directly: quantized encoders + untouched preprocessor /
decoder / joint copies.

Usage:
    uv run --no-sync python quantize_int8.py \
        --coreml-dir ./build/parakeet_unified_coreml --output-dir ./build/parakeet_unified_coreml_int8
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

ENCODERS = [
    "parakeet_unified_encoder.mlpackage",
    "parakeet_unified_encoder_streaming_70_13_13.mlpackage",
]
PASSTHROUGH = [
    "parakeet_unified_preprocessor.mlpackage",
    "parakeet_unified_decoder.mlpackage",
    "parakeet_unified_joint_decision_single_step.mlpackage",
    "parakeet_unified_joint.mlpackage",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-dir", type=Path, default=Path("build/parakeet_unified_coreml"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/parakeet_unified_coreml_int8"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", granularity="per_channel", dtype="int8"
        )
    )

    for name in ENCODERS:
        src = args.coreml_dir / name
        dst = args.output_dir / name
        if dst.exists():
            print(f"exists, skipping: {dst.name}")
            continue
        print(f"quantizing {name}…")
        model = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_ONLY)
        quantized = linear_quantize_weights(model, cfg)
        quantized.save(str(dst))
        print(f"saved {dst}")

    for name in PASSTHROUGH:
        src = args.coreml_dir / name
        dst = args.output_dir / name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
    metadata = args.coreml_dir / "metadata.json"
    if metadata.exists():
        shutil.copy(metadata, args.output_dir / "metadata.json")

    print(f"Done. Validate with: uv run --no-sync python benchmark_wer.py --coreml-dir {args.output_dir}")


if __name__ == "__main__":
    main()
