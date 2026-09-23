"""Check the adaptive W8 runtime against stored pinned native outputs."""

import argparse
import json
import runpy
from pathlib import Path

from extraction_runtime import CoreMLAdaptiveBoundaryExtractor

MODEL_REVISION = "a221b77a8baf4a613b8f8652661d41fa10a5641e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
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
    runtime = CoreMLAdaptiveBoundaryExtractor(args.model_dir)
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
                "route": runtime.selected_precision(schema),
                "structure_match": without_confidence(expected) == without_confidence(actual),
                "maximum_confidence_error": max(errors, default=None),
                "coreml": actual,
            }
        )
        print(f"{name}: {cases[-1]['route']} {cases[-1]['structure_match']}", flush=True)
    report = {
        "source_revision": MODEL_REVISION,
        "selected_manifest": "fifteen pinned real-text schema fixtures, not a Decision Index score",
        "variant": "adaptive-w8-embedding",
        "matched": sum(case["structure_match"] for case in cases),
        "total": len(cases),
        "cases": cases,
    }
    output = Path(args.model_dir) / "verify-adaptive-w8.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"matched": report["matched"], "total": report["total"]}, indent=2))
    if report["matched"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
