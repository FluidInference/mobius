"""Upgrade the verified FP16 graph to iOS18 without changing its attention lowering."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import coremltools as ct
from coremltools.converters.mil.frontend.milproto.load import load

from assets import ROOT, sha256, verify_assets
from verify import check_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/int4-source-fp16")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a fresh output directory to preserve existing artifacts")
    verify_assets()
    package, manifest = check_package(args.source)
    if (
        manifest.get("optimization", "baseline") != "baseline"
        or "quantization" in manifest
        or manifest["precision"] != "float16"
        or manifest["minimum_target"] != "iOS17/macOS14"
    ):
        raise ValueError("Expected the verified original iOS17 FP16 baseline")
    source = ct.models.MLModel(str(package), skip_model_load=True)
    program = load(source.get_spec(), specification_version=9, file_weights_dir=source.weights_dir)
    # The loaded graph already contains FP16 operations. FLOAT32 prevents a
    # second precision rewrite; it does not change the existing tensor dtypes.
    upgraded = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        pass_pipeline=ct.PassPipeline.EMPTY,
    )
    upgraded.short_description = "CUA-S1-FORMS: FP16 control for the INT4 trial, iOS18/macOS15"
    upgraded.author, upgraded.license = source.author, source.license
    upgraded.user_defined_metadata.update(source.user_defined_metadata)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / manifest["model"]
    upgraded.save(str(output))
    operations = Counter(op.op_type for op in program.functions["main"].operations)
    result = {
        **manifest,
        "minimum_target": "iOS18/macOS15",
        "target_upgrade": {
            "source_conversion_sha256": sha256(args.source / "conversion.json"),
            "source_package_files": manifest["package_files"],
            "method": "Load original MIL at specification 9; empty pass pipeline; retain decomposed attention",
            "operation_counts": dict(sorted(operations.items())),
            "script_sha256": sha256(Path(__file__)),
        },
        "package_files": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (args.output_dir / "conversion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved FP16 iOS18 control: {output}", flush=True)


if __name__ == "__main__":
    main()
