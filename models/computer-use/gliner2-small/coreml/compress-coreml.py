"""Palettize a verified GLiNER2 classification Core ML model."""
import argparse
import json
import time
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OpPalettizerConfig,
    OptimizationConfig,
    get_weights_metadata,
    linear_quantize_weights,
    palettize_weights,
)


def package_bytes(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", type=int, choices=[4, 6, 8], default=8)
    parser.add_argument("--method", choices=["lut", "linear", "embedding"], default="lut")
    parser.add_argument("--granularity", choices=["per_tensor", "per_grouped_channel"], default="per_tensor")
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args()
    start = time.perf_counter()
    model = ct.models.MLModel(args.source, skip_model_load=True)
    selected_weights = None
    if args.method in ("linear", "embedding"):
        if args.bits != 8:
            parser.error("linear quantization here supports only 8-bit weights")
        quantizer = OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=2048,
        )
        if args.method == "embedding":
            metadata = get_weights_metadata(model, weight_threshold=2048)
            names = [
                name for name, weight in metadata.items()
                if "encoder_embeddings_word_embeddings" in name
                and len(weight.val.shape) == 2
                and any(op.op_type == "gather" for op in weight.child_ops)
            ]
            if len(names) != 1:
                raise ValueError(f"expected one token embedding weight, found {names}")
            config = OptimizationConfig(op_name_configs={names[0]: quantizer})
            selected_weights = names
        else:
            config = OptimizationConfig(global_config=quantizer)
        compressed = linear_quantize_weights(model, config=config)
        variant = "W8 token embedding only" if args.method == "embedding" else "W8 per-channel"
    else:
        config = OptimizationConfig(global_config=OpPalettizerConfig(
            mode="kmeans", nbits=args.bits, granularity=args.granularity,
            group_size=args.group_size,
            enable_per_channel_scale=args.granularity == "per_grouped_channel",
            num_kmeans_workers=4,
        ))
        compressed = palettize_weights(model, config=config)
        variant = f"LUT{args.bits} {args.granularity}"
    compressed.short_description = f"{model.short_description}; {variant} weights"
    compressed.author = model.author
    compressed.license = model.license
    compressed.user_defined_metadata.update(model.user_defined_metadata)
    compressed.user_defined_metadata["weight_compression"] = variant
    compressed.save(args.output)
    report = {"source": args.source, "output": args.output, "method": args.method,
              "granularity": "per_channel" if args.method in ("linear", "embedding") else args.granularity,
              "selected_weights": selected_weights,
              "source_bytes": package_bytes(Path(args.source)),
              "output_bytes": package_bytes(Path(args.output)), "compression_seconds": time.perf_counter() - start}
    Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
