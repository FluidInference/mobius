"""Create targeted int8 Jeff Core ML variants without modifying the FP16 source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto


def package_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def selected_constants(model, scheme: str) -> list[str]:
    metadata = cto.get_weights_metadata(model, weight_threshold=2048)
    selected = []
    for name, weight in metadata.items():
        if not weight.child_ops:
            continue
        consumer = weight.child_ops[0].op_type
        embedding = consumer == "gather" and len(weight.val.shape) == 2
        linear = consumer == "linear" and len(weight.val.shape) == 2
        if (scheme == "e8" and embedding) or (scheme == "w8" and (embedding or linear)):
            selected.append(name)
    if not selected:
        raise ValueError(f"no eligible constants found for {scheme}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("build/JeffDecision-L128-FP16.mlpackage"))
    parser.add_argument("--scheme", choices=("e8", "w8"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"build/JeffDecision-L128-{args.scheme.upper()}.mlpackage")
    if output.exists():
        parser.error(f"output already exists: {output}")
    model = ct.models.MLModel(str(args.source), compute_units=ct.ComputeUnit.CPU_ONLY)
    names = selected_constants(model, args.scheme)
    config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
    compressed = cto.linear_quantize_weights(
        model, cto.OptimizationConfig(op_name_configs={name: config for name in names})
    )
    compressed.user_defined_metadata["precision"] = args.scheme
    compressed.save(str(output))
    report = {
        "source": str(args.source.resolve()),
        "source_bytes": package_bytes(args.source),
        "output": str(output.resolve()),
        "output_bytes": package_bytes(output),
        "compressed_constants": len(names),
        "scheme": args.scheme,
    }
    Path(f"build/quantize-{args.scheme}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
