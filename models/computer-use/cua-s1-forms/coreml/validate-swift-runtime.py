"""Validate stable Swift probabilities on the pinned full split without changing raw benchmarks."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from assets import ROOT, sha256
from preprocessing import InputLimits
from synthetic_test import inspect_rows, load_test
from verify import check_package


def audit_trace(trace: Path, rows: list[dict], expected: dict[str, list[int]]) -> dict:
    """Require complete coverage and compare SDK outputs to saved raw model decisions."""
    results = {
        name: {
            "rows": 0,
            "correct": 0,
            "changed_choices": [],
            "raw_choice_changes": [],
            "raw_sum_failures": [],
            "runtime_sum_failures": [],
            "max_probability_change": 0.0,
            "max_softmax_reference_error": 0.0,
        }
        for name in expected
    }
    with trace.open() as stream:
        for line in stream:
            record = json.loads(line)
            name, index = record["model"], record["row"]
            result = results[name]
            if index != result["rows"] or record["label"] != rows[index]["label"]:
                raise ValueError("Runtime trace reordered, duplicated, or changed a test row")
            raw = np.asarray(record["rawProbabilities"], dtype=np.float32)
            actual = np.asarray(record["probabilities"], dtype=np.float32)
            logits = np.asarray(record["logits"], dtype=np.float64)
            count = len(rows[index]["options"])
            if raw.shape != (count,) or actual.shape != (count,) or logits.shape != (count,):
                raise ValueError("Runtime trace has inconsistent tensor sizes")
            if not all(np.isfinite(values).all() for values in (raw, actual, logits)):
                raise ValueError("Runtime trace contains nonfinite output")
            if np.any(actual < 0) or np.any(actual > 1):
                raise ValueError("Runtime trace has invalid probabilities")
            if record["selected"] != int(actual.argmax()) or record["rawSelected"] != int(raw.argmax()):
                raise ValueError("Reported selected option disagrees with emitted scores")
            reference = np.exp(logits - logits.max())
            reference /= reference.sum()
            result["max_softmax_reference_error"] = max(
                result["max_softmax_reference_error"], float(np.max(np.abs(actual - reference)))
            )
            result["max_probability_change"] = max(
                result["max_probability_change"], float(np.max(np.abs(actual - raw)))
            )
            result["rows"] += 1
            result["correct"] += record["selected"] == record["label"]
            if record["selected"] != record["rawSelected"]:
                result["changed_choices"].append(index)
            if record["rawSelected"] != expected[name][index]:
                result["raw_choice_changes"].append(index)
            if abs(float(raw.astype(np.float64).sum()) - 1) > 0.001:
                result["raw_sum_failures"].append(index)
            if abs(float(actual.astype(np.float64).sum()) - 1) > 0.001:
                result["runtime_sum_failures"].append(index)
    for result in results.values():
        if result["rows"] != len(rows):
            raise ValueError("Runtime trace does not cover every row for every model")
        result["accuracy"] = result["correct"] / result["rows"]
        result["passed"] = not (
            result["changed_choices"]
            or result["raw_choice_changes"]
            or result["runtime_sum_failures"]
            or result["max_softmax_reference_error"] > 1e-7
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fluidaudio", type=Path, required=True)
    parser.add_argument(
        "--models", type=Path, required=True, help="Local HF repository download with reports and traces"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/swift-runtime-validation")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a fresh output directory to preserve recorded validation")
    rows, dataset = load_test()
    model_root = args.models.resolve()
    variants = [
        ("fp16", "", "synthetic-test", "baseline"),
        ("int8", "int8-weights", "int8-synthetic-test", "int8-weights"),
        ("int4", "int4-weights", "int4-synthetic-test", "int4-weights"),
    ]
    models, conversions, references, expected = [], {}, {}, {}
    for name, folder, stem, key in variants:
        package, conversion = check_package(model_root / folder)
        inspect_rows(rows, InputLimits(**conversion["limits"]))
        models.append({"name": name, "path": str(package)})
        conversions[name] = conversion
        report = json.loads((model_root / "reports" / f"{stem}.json").read_text())
        if report["conversions"][key] != conversion or report["dataset"]["sha256"] != dataset["sha256"]:
            raise ValueError("Saved benchmark does not match the selected model and test split")
        trace = model_root / "reports" / report["trace"]["file"]
        if sha256(trace) != report["trace"]["sha256"]:
            raise ValueError("Saved benchmark trace checksum mismatch")
        with gzip.open(trace, "rt") as stream:
            entries = [json.loads(line) for line in stream]
        if [entry["row"] for entry in entries] != list(range(len(rows))):
            raise ValueError("Saved benchmark trace coverage mismatch")
        expected[name] = [entry["outputs"][key]["selected"] for entry in entries]
        references[name] = {
            "report_sha256": sha256(model_root / "reports" / f"{stem}.json"),
            "trace_sha256": sha256(trace),
        }
    output = args.output_dir.resolve()
    project = output / "package"
    source = project / "Sources/Runtime"
    source.mkdir(parents=True)
    shutil.copy2(ROOT / "swift-runtime-check.swift", source / "main.swift")
    project.joinpath("Package.swift").write_text(
        "// swift-tools-version: 6.0\nimport PackageDescription\n"
        'let package = Package(name: "CuaRuntimeValidation", platforms: [.macOS(.v14)],\n'
        f'    dependencies: [.package(name: "FluidAudio", path: {json.dumps(str(args.fluidaudio.resolve()))})],\n'
        '    targets: [.executableTarget(name: "Runtime", dependencies: [.product(name: "FluidAudio", '
        'package: "FluidAudio")])])\n'
    )
    trace = output / "decisions.jsonl"
    config = {
        "dataset": str(ROOT / dataset["path"]),
        "datasetSHA256": dataset["sha256"],
        "rows": len(rows),
        "models": models,
        "trace": str(trace),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run(
        ["swift", "run", "--package-path", str(project), "-c", "release", "Runtime", str(output / "config.json")],
        check=True,
    )
    results = audit_trace(trace, rows, expected)
    packed = output / "decisions.jsonl.gz"
    with trace.open("rb") as reader, gzip.open(packed, "wb") as writer:
        shutil.copyfileobj(reader, writer)
    swift_root = args.fluidaudio / "Sources/FluidAudio/Decision/CuaS1Forms"
    report = {
        "purpose": "Stable Swift softmax validation; no model changes, training or new timing claims",
        "dataset": dataset,
        "conversions": conversions,
        "references": references,
        "results": results,
        "passed": all(result["passed"] for result in results.values()),
        "swift_source_sha256": {p.name: sha256(p) for p in sorted(swift_root.glob("*.swift"))},
        "harness_sha256": {p.name: sha256(p) for p in (Path(__file__), ROOT / "swift-runtime-check.swift")},
        "trace": {"file": packed.name, "sha256": sha256(packed), "records": len(rows) * len(models)},
        "limitations": [
            "Original raw conversion-parity failures remain in the unchanged benchmarks.",
            "INT4 still has its previously measured accuracy loss.",
            "CPU+ANE on the local Mac; no iPhone or older-OS validation.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
