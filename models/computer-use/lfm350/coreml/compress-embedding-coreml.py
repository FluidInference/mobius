"""Quantize only the tied token-embedding/output weight of the LFM Core ML scorer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    get_weights_metadata,
    linear_quantize_weights,
)


def package_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-mode", choices=("linear_symmetric", "linear"), default="linear_symmetric")
    args = parser.parse_args()
    started = time.perf_counter()
    model = ct.models.MLModel(str(args.source), skip_model_load=True)
    metadata = get_weights_metadata(model, weight_threshold=2048)
    names = [
        name for name, weight in metadata.items()
        if "model_embed_tokens_weight" in name
        and len(weight.val.shape) == 2
        and any(op.op_type == "gather" for op in weight.child_ops)
    ]
    if len(names) != 1:
        raise ValueError(f"expected one tied embedding weight, found {names}")
    quantizer = OpLinearQuantizerConfig(
        mode=args.linear_mode, dtype="int8", granularity="per_channel", weight_threshold=2048,
    )
    compressed = linear_quantize_weights(model, config=OptimizationConfig(op_name_configs={names[0]: quantizer}))
    compressed.short_description = f"{model.short_description}; W8 tied embedding/output weight"
    compressed.author = model.author
    compressed.license = model.license
    compressed.user_defined_metadata.update(model.user_defined_metadata)
    compressed.user_defined_metadata["weight_compression"] = f"W8 {args.linear_mode} tied embedding/output"
    compressed.save(str(args.output))
    report = {
        "source": str(args.source), "output": str(args.output), "selected_weights": names,
        "linear_mode": args.linear_mode,
        "source_bytes": package_bytes(args.source),
        "output_bytes": package_bytes(args.output), "compression_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".compression.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
