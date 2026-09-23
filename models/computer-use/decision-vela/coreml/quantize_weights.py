"""Create an experimental per-channel W8 variant of one validated typed package.

This is weight-only compression. The caller must separately verify native
decisions, Core ML placement, and full-request latency before publishing it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typed_coreml import package_sha256


def package_bytes(package: Path) -> int:
    return sum(path.stat().st_size for path in package.rglob("*") if path.is_file())


def quantize(source: Path, destination: Path, scope: str = "all", linear_mode: str = "linear_symmetric") -> dict:
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        get_weights_metadata,
        linear_quantize_weights,
    )

    source = source.resolve()
    destination = destination.resolve()
    if source == destination or destination.exists():
        raise ValueError("Destination must be a distinct, nonexistent package")
    if source.suffix != ".mlpackage" or destination.suffix != ".mlpackage":
        raise ValueError("Source and destination must be .mlpackage directories")
    if not source.is_dir():
        raise FileNotFoundError(source)
    source_report = source.with_suffix(".json")
    if not source_report.is_file():
        raise FileNotFoundError(source_report)
    metadata = json.loads(source_report.read_text())
    if metadata.get("coreml_choice_agreement") is not True:
        raise ValueError(
            "The source package must have successful conversion validation"
        )
    original = ct.models.MLModel(str(source), skip_model_load=True)
    quantizer = OpLinearQuantizerConfig(
        mode=linear_mode, dtype="int8", granularity="per_channel", weight_threshold=2048,
    )
    selected_weights = None
    if scope == "embedding":
        weights = get_weights_metadata(original, weight_threshold=2048)
        selected_weights = [
            name for name, weight in weights.items()
            if len(weight.val.shape) == 2
            and any(op.op_type == "gather" for op in weight.child_ops)
            and "embeddings_tok_embeddings" in name
        ]
        if len(selected_weights) != 1:
            raise ValueError(f"expected one token embedding, found {selected_weights}")
        config = OptimizationConfig(op_name_configs={selected_weights[0]: quantizer})
    elif scope == "all":
        config = OptimizationConfig(global_config=quantizer)
    else:
        raise ValueError(f"unsupported compression scope: {scope}")
    compressed = linear_quantize_weights(original, config=config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    compressed.save(str(destination))
    result = {
        "kind": metadata["kind"],
        "shape": metadata["shape"],
        "source_repo": metadata["source_repo"],
        "source_revision": metadata["source_revision"],
        "package_sha256": package_sha256(destination),
        "package_bytes": package_bytes(destination),
        "compressed_from": {
            "package": source.name,
            "sha256": package_sha256(source),
            "bytes": package_bytes(source),
        },
        "compression": {
            "type": "weight-only linear quantization",
            "mode": linear_mode,
            "dtype": "int8",
            "granularity": "per_channel",
            "weight_threshold": 2048,
            "scope": scope,
            "selected_weights": selected_weights,
        },
        "validated": False,
    }
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope", choices=("all", "embedding"), default="all")
    parser.add_argument("--linear-mode", choices=("linear_symmetric", "linear"), default="linear_symmetric")
    args = parser.parse_args()
    print(json.dumps(quantize(args.source, args.output, args.scope, args.linear_mode), indent=2))


if __name__ == "__main__":
    main()
