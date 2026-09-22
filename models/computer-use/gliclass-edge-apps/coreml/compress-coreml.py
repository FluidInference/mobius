"""Palettize a converted GLiClass Edge Core ML bucket and record its size."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights


def package_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", default="build/coreml")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=25)
    parser.add_argument("--bits", type=int, choices=[4, 6, 8], required=True)
    parser.add_argument("--mode", choices=["kmeans", "uniform"], default="kmeans")
    parser.add_argument("--granularity", choices=["per_tensor", "per_grouped_channel"], default="per_tensor")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.packages)
    source = root / f"gliclass_edge_apps_fp16_L{args.length}_options{args.max_options}.mlpackage"
    suffix = f"lut{args.bits}_{args.mode}_{args.granularity}"
    output = root / f"gliclass_edge_apps_{suffix}_L{args.length}_options{args.max_options}.mlpackage"
    report = root / f"compression-{suffix}-L{args.length}.json"
    if not source.exists():
        raise FileNotFoundError(source)

    started = time.perf_counter()
    model = ct.models.MLModel(str(source), skip_model_load=True)
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            mode=args.mode,
            nbits=args.bits,
            granularity=args.granularity,
            group_size=args.group_size,
            enable_per_channel_scale=args.granularity == "per_grouped_channel",
            num_kmeans_workers=args.workers,
        )
    )
    compressed = palettize_weights(model, config=config)
    compressed.short_description = f"{model.short_description}; {args.bits}-bit weight LUT"
    compressed.author = model.author
    compressed.license = model.license
    compressed.user_defined_metadata.update(model.user_defined_metadata)
    compressed.user_defined_metadata.update(
        {
            "weight_compression": f"{args.bits}-bit {args.mode} LUT",
            "weight_granularity": args.granularity,
            "weight_group_size": str(args.group_size),
        }
    )
    compressed.save(str(output))
    result = {
        "source": str(source),
        "output": str(output),
        "length": args.length,
        "bits": args.bits,
        "mode": args.mode,
        "granularity": args.granularity,
        "group_size": args.group_size,
        "source_bytes": package_bytes(source),
        "output_bytes": package_bytes(output),
        "compression_seconds": time.perf_counter() - started,
        "coremltools": ct.__version__,
    }
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
