"""Compare a quantized extraction package with stored pinned native outputs."""

import argparse
import json
import runpy
from pathlib import Path

import coremltools as ct

from extraction_runtime import CoreMLBoundaryExtractor

MODEL_REVISION = "a221b77a8baf4a613b8f8652661d41fa10a5641e"
UNITS = {
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_and_neural_engine": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--feature-package", required=True)
    parser.add_argument("--precision", choices=["fp16", "fp32"], required=True)
    parser.add_argument("--units", choices=list(UNITS), default="cpu_and_neural_engine")
    parser.add_argument("--reference", default=str(Path(__file__).parent / "reports/extraction-verify-fp32.json"))
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text())
    if reference["source_revision"] != MODEL_REVISION or reference["matched"] != reference["total"]:
        raise ValueError("Reference must be the complete pinned native/FP32 parity manifest")
    helpers = runpy.run_path(str(Path(__file__).parent / "verify-full-extraction.py"))
    fixtures = helpers["fixtures"]()
    without_confidence = helpers["without_confidence"]
    confidence_errors = helpers["confidence_errors"]
    reference_cases = {case["name"]: case for case in reference["cases"]}
    if {name for name, _, _ in fixtures} != set(reference_cases):
        raise ValueError("Fixture names differ from the pinned native reference")
    runtime = CoreMLBoundaryExtractor(
        args.model_dir,
        precision=args.precision,
        compute_units=UNITS[args.units],
        feature_package=args.feature_package,
    )
    cases = []
    for name, text, schema in fixtures:
        baseline = reference_cases[name]
        if text != baseline["text"]:
            raise ValueError(f"Fixture text changed for {name}")
        expected = baseline["native"]
        actual = runtime.extract(text, schema, include_confidence=True, include_spans=True)
        errors = confidence_errors(expected, actual)
        cases.append(
            {
                "name": name,
                "structure_match": without_confidence(expected) == without_confidence(actual),
                "maximum_confidence_error": max(errors, default=None),
                "coreml": actual,
            }
        )
        print(f"{name}: {cases[-1]['structure_match']}", flush=True)
    report = {
        "source_revision": MODEL_REVISION,
        "selected_manifest": "pinned real-text schema fixtures, not a Decision Index score",
        "feature_package": args.feature_package,
        "precision": args.precision,
        "compute_units": args.units,
        "matched": sum(case["structure_match"] for case in cases),
        "total": len(cases),
        "cases": cases,
    }
    output = Path(args.model_dir) / f"verify-quantized-{args.precision}-{args.units}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"matched": report["matched"], "total": report["total"]}, indent=2))
    if report["matched"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
