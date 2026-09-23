"""Compress only Verdict's token embedding, preserving encoder/head FP16 weights."""

from __future__ import annotations

import argparse
import json
import time

import coremltools as ct
import coremltools.optimize.coreml as cto

from assets import ROOT, sha256


def package_bytes(path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    args = parser.parse_args()
    source = ROOT / "build" / f"verdict_fp16_L{args.length}_candidates25.mlpackage"
    target = ROOT / "build" / f"verdict_e8_L{args.length}_candidates25.mlpackage"
    if target.exists():
        raise FileExistsError(target)
    started = time.perf_counter()
    model = ct.models.MLModel(str(source), skip_model_load=True)
    metadata = cto.get_weights_metadata(model, weight_threshold=2048)
    embedding_weights = [
        name
        for name, weight in metadata.items()
        if weight.child_ops and weight.child_ops[0].op_type == "gather" and len(weight.val.shape) == 2
    ]
    if len(embedding_weights) != 1:
        raise ValueError(f"expected exactly one token embedding gather, found {embedding_weights}")
    config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
    converted = cto.linear_quantize_weights(
        model, cto.OptimizationConfig(op_name_configs={name: config for name in embedding_weights})
    )
    converted.short_description = f"{model.short_description}; int8 token embedding, FP16 encoder/head"
    converted.author = model.author
    converted.license = model.license
    converted.user_defined_metadata.update(model.user_defined_metadata)
    converted.user_defined_metadata["weight_compression"] = "int8 token embedding only"
    converted.save(str(target))
    report = {
        "source": source.name,
        "package": target.name,
        "embedding_weights": embedding_weights,
        "source_bytes": package_bytes(source),
        "package_bytes": package_bytes(target),
        "seconds": time.perf_counter() - started,
        "coremltools": ct.__version__,
        "package_files_sha256": {
            str(path.relative_to(target)): sha256(path) for path in target.rglob("*") if path.is_file()
        },
        "parity_verified": False,
    }
    output = ROOT / "reports" / f"e8-L{args.length}-conversion.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "package_files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
