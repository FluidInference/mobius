"""Post-training weight compression of an exported FP16 bucket with coremltools.optimize.

Variants (precision tag → scheme):
  w8   int8 linear-symmetric per-channel weights everywhere, including the 256k×768 embedding table
  w8e  int8 encoder/head weights only; embedding table stays fp16
  e8   int8 embedding table only; encoder/head stay fp16
  w6   6-bit k-means palettization of encoder/head weights, fp16 embedding table
  w4   4-bit k-means palettization of encoder/head weights, fp16 embedding table
  w6e8 / w4e8  as w6 / w4 with an int8 embedding table

The result is saved next to the FP16 package as `laya_<variant>_<precision>_L<L>_options32.mlpackage`
and must pass `verify.py --precision <tag>` before it is published.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto

from assets import ROOT, sha256
from preprocessing import package_name

# tag -> (embedding, other weights); "fp16" leaves that group untouched, int n = n-bit k-means palette
SCHEMES = {
    "w8": ("int8", "int8"),
    "w8e": ("fp16", "int8"),
    "e8": ("int8", "fp16"),
    "w6": ("fp16", 6),
    "w4": ("fp16", 4),
    "w6e8": ("int8", 6),
    "w4e8": ("int8", 4),
    # isolation experiments
    "w8enc": ("fp16", "int8-encoder"),  # encoder layers only, decision head fp16
    "w8head": ("fp16", "int8-head"),  # decision head / scorer / action head only
    "w8b": ("fp16", "int8-block32"),  # per-block (32) int8 on every linear weight
}


def compress(model: ct.models.MLModel, precision: str) -> ct.models.MLModel:
    """Compress only the linear weight matrices (and optionally the embedding table).

    The default weight threshold would also sweep the additive attention masks, the RoPE cos/sin
    tables and the biases into compression, which wrecks the outputs on the Neural Engine.
    """
    embedding, others = SCHEMES[precision]
    int8 = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
    metadata = cto.get_weights_metadata(model, weight_threshold=2048)
    linear_weights = [
        name
        for name, weight in metadata.items()
        if name.endswith("weight_to_fp16")
        and len(weight.val.shape) == 2
        and weight.child_ops
        and weight.child_ops[0].op_type == "linear"
    ]
    embedding_weights = [
        name for name, weight in metadata.items() if weight.child_ops and weight.child_ops[0].op_type == "gather"
    ]
    if others == "int8-encoder":
        linear_weights = [n for n in linear_weights if "_encoder_" in n]
    elif others == "int8-head":
        linear_weights = [n for n in linear_weights if "_encoder_" not in n]
    if isinstance(others, str) and others.startswith("int8"):
        if others == "int8-block32":
            int8 = cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8", granularity="per_block", block_size=32
            )
        others = "int8"
    result = model
    if embedding == "int8":
        result = cto.linear_quantize_weights(
            result, cto.OptimizationConfig(op_name_configs={name: int8 for name in embedding_weights})
        )
    if others == "int8":
        result = cto.linear_quantize_weights(
            result, cto.OptimizationConfig(op_name_configs={name: int8 for name in linear_weights})
        )
    elif isinstance(others, int):
        palette = cto.OpPalettizerConfig(mode="kmeans", nbits=others, granularity="per_tensor", num_kmeans_workers=8)
        result = cto.palettize_weights(
            result, cto.OptimizationConfig(op_name_configs={name: palette for name in linear_weights})
        )
    print(
        f"compressed {len(linear_weights)} linear weights ({others}) and {len(embedding_weights) if embedding == 'int8' else 0} embedding tables"
    )
    return result


def package_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="multilingual")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--precision", choices=sorted(SCHEMES), required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    source = args.build_dir / f"{package_name(args.variant, args.length, args.max_options)}.mlpackage"
    target = args.build_dir / f"{package_name(args.variant, args.length, args.max_options, args.precision)}.mlpackage"
    started = time.perf_counter()
    model = ct.models.MLModel(str(source), compute_units=ct.ComputeUnit.CPU_ONLY)
    compressed = compress(model, args.precision)
    compressed.user_defined_metadata["precision"] = args.precision
    compressed.short_description = (model.short_description or "") + f" [{args.precision}]"
    compressed.save(str(target))
    manifest = {
        "model": target.name,
        "source": source.name,
        "precision": args.precision,
        "scheme": {
            "embedding": SCHEMES[args.precision][0],
            "other_weights": str(SCHEMES[args.precision][1]),
        },
        "source_bytes": package_bytes(source),
        "package_bytes": package_bytes(target),
        "seconds": round(time.perf_counter() - started, 1),
        "coremltools": ct.__version__,
        "package_files": {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob("*")) if p.is_file()},
    }
    (args.build_dir / f"{target.stem}.conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Saved {target.name}: {manifest['package_bytes'] / 1e6:.0f} MB from {manifest['source_bytes'] / 1e6:.0f} MB in {manifest['seconds']} s"
    )


if __name__ == "__main__":
    main()
