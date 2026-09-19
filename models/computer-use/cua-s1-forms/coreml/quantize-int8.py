"""Compress the verified baseline to INT8 weights while retaining FP16 computation."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import coremltools as ct

from assets import ROOT, sha256, verify_assets
from verify import check_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/int8-weights")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a fresh output directory to preserve existing artifacts")
    verify_assets()
    source, manifest = check_package(args.source)
    if manifest.get("optimization", "baseline") != "baseline" or "quantization" in manifest:
        raise ValueError("This trial requires the original unquantized baseline")
    if manifest["precision"] != "float16":
        raise ValueError("Expected the original FP16 baseline")
    model = ct.models.MLModel(str(source), compute_units=ct.ComputeUnit.CPU_ONLY)
    settings = {"mode": "linear_symmetric", "dtype": "int8", "granularity": "per_channel", "weight_threshold": 2048}
    config = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpLinearQuantizerConfig(**settings))
    started = time.perf_counter()
    quantized = ct.optimize.coreml.linear_quantize_weights(model, config)
    spec = quantized.get_spec()
    operations = Counter(
        op.type
        for function in spec.mlProgram.functions.values()
        for block in function.block_specializations.values()
        for op in block.operations
    )
    if not operations["constexpr_affine_dequantize"]:
        raise ValueError("No INT8 weight-dequantization operations were produced")
    quantized.short_description = "CUA-S1-FORMS: experimental INT8 weights, FP16 computation"
    quantized.user_defined_metadata["quantization"] = "int8-weights; FP16 computation; no activation quantization"
    args.output_dir.mkdir(parents=True)
    package = args.output_dir / "cua_s1_forms_int8_options32.mlpackage"
    quantized.save(str(package))
    source_bytes = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    package_bytes = sum(p.stat().st_size for p in package.rglob("*") if p.is_file())
    result = {
        **manifest,
        "model": package.name,
        "precision": "int8_weights_float16_compute",
        "coremltools": ct.__version__,
        "quantization": {
            "name": "int8-weights",
            "settings": settings,
            "activations": "float16",
            "calibration": "None; weight-only, data-free quantization",
            "source_conversion_sha256": sha256(args.source / "conversion.json"),
            "source_package_files": manifest["package_files"],
            "source_package_bytes": source_bytes,
            "package_bytes": package_bytes,
            "compression_seconds": time.perf_counter() - started,
            "operation_counts": dict(sorted(operations.items())),
            "script_sha256": sha256(Path(__file__)),
        },
        "package_files": {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*")) if p.is_file()},
    }
    (args.output_dir / "conversion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"INT8 weights: {source_bytes:,} -> {package_bytes:,} bytes; FP16 compute retained; {package}", flush=True)


if __name__ == "__main__":
    main()
