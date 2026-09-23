"""Quantize the trained GLiNER2.5 encoder embedding without changing other heads."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    get_weights_metadata,
    linear_quantize_weights,
)


def package_files(package: Path):
    """Return exact byte sizes and hashes for the files in a Core ML package."""
    files = []
    for file in sorted(item for item in package.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {"path": str(file.relative_to(package)), "bytes": file.stat().st_size, "sha256": digest.hexdigest()}
        )
    return files


def embedding_weight_name(model) -> tuple[str, tuple[int, ...], str]:
    """Locate the real checkpoint's word embedding constant in an ML Program."""
    metadata = get_weights_metadata(model)
    matches = [name for name in metadata if name.startswith("encoder_embeddings_word_embeddings_weight")]
    if len(matches) != 1:
        raise ValueError(f"Expected one trained word embedding constant, found {matches}")
    name = matches[0]
    weight = metadata[name].val
    if weight.ndim != 2 or weight.shape[0] < 65_536:
        raise ValueError(f"Unexpected word embedding shape: {weight.shape}")
    return name, tuple(weight.shape), str(weight.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["linear_symmetric", "linear"], default="linear_symmetric")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        raise ValueError("Quantized output must differ from the source package")
    started = time.perf_counter()
    model = ct.models.MLModel(str(source), skip_model_load=True)
    name, shape, dtype = embedding_weight_name(model)
    config = OptimizationConfig(
        op_name_configs={name: OpLinearQuantizerConfig(mode=args.mode, dtype="int8", granularity="per_channel")}
    )
    compressed = linear_quantize_weights(model, config=config)
    compressed.short_description = f"{model.short_description}; W8 {args.mode} per-channel word embedding"
    compressed.author = model.author
    compressed.license = model.license
    compressed.user_defined_metadata.update(model.user_defined_metadata)
    compressed.user_defined_metadata["weight_compression"] = f"W8 {args.mode} per-channel word embedding"
    compressed.user_defined_metadata["quantized_weight"] = name
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.rmtree(output)
    compressed.save(str(output))
    source_files, output_files = package_files(source), package_files(output)
    source_bytes = sum(file["bytes"] for file in source_files)
    output_bytes = sum(file["bytes"] for file in output_files)
    if output_bytes >= source_bytes:
        raise RuntimeError("Quantized package is not smaller than its source")
    report = {
        "source_package": source.name,
        "output_package": output.name,
        "selected_weight": name,
        "weight_shape": shape,
        "source_dtype": dtype,
        "method": f"{args.mode} int8 per-channel weight-only",
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "seconds": time.perf_counter() - started,
        "coremltools": ct.__version__,
        "output_files": output_files,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "output_files"}, indent=2))


if __name__ == "__main__":
    main()
