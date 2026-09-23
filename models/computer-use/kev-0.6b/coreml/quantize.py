"""Compress the Kev Core ML package and leave the FP16 source untouched."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto

from assets import ROOT


def package_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--precision", choices=("e8", "w8", "w8mlp", "w8mlp12", "w4"), required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    stem = f"kev_0_6b_fp16_L{args.length}_options{args.max_options}"
    source = args.build_dir / f"{stem}.mlpackage"
    target = args.build_dir / f"{stem.replace('fp16', args.precision)}.mlpackage"
    model = ct.models.MLModel(str(source), compute_units=ct.ComputeUnit.CPU_ONLY)
    metadata = cto.get_weights_metadata(model, weight_threshold=2048)
    embeddings = [
        name for name, weight in metadata.items() if weight.child_ops and weight.child_ops[0].op_type == "gather"
    ]
    linear = [
        name
        for name, weight in metadata.items()
        if len(weight.val.shape) == 2 and weight.child_ops and weight.child_ops[0].op_type == "linear"
    ]
    if args.precision == "e8":
        names = embeddings
        config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
        compressed = cto.linear_quantize_weights(
            model, cto.OptimizationConfig(op_name_configs={name: config for name in names})
        )
    elif args.precision in ("w8", "w8mlp", "w8mlp12"):
        names = embeddings + linear
        if args.precision == "w8mlp":
            names = embeddings + [name for name in linear if "_mlp_" in name]
        elif args.precision == "w8mlp12":
            names = embeddings + [
                name
                for name in linear
                if "_mlp_" in name and int(name.split("_", 2)[1]) % 2 == 0
            ]
        config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
        compressed = cto.linear_quantize_weights(
            model, cto.OptimizationConfig(op_name_configs={name: config for name in names})
        )
    else:
        names = embeddings + linear
        config = cto.OpPalettizerConfig(mode="kmeans", nbits=4, granularity="per_tensor", num_kmeans_workers=8)
        compressed = cto.palettize_weights(
            model, cto.OptimizationConfig(op_name_configs={name: config for name in names})
        )
    compressed.user_defined_metadata["precision"] = args.precision
    compressed.short_description = (model.short_description or "") + f" [{args.precision}]"
    started = time.perf_counter()
    compressed.save(str(target))
    print(
        f"Saved {target.name}: {package_bytes(target) / 1e6:.0f} MB from "
        f"{package_bytes(source) / 1e6:.0f} MB; {len(names)} compressed constants; "
        f"save {time.perf_counter() - started:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
