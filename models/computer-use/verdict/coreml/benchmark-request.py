"""Small fixed request-level Verdict comparison, including rendering and calibration."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

from assets import ROOT, sha256, verify_assets
from decision_index_engine import adapt_question, as_text
from native_reference import build_request, decode, load_calibrator
from preprocessing import prepare


def select_rows(path: Path, tokenizer, limit: int = 12, per_family: int = 2) -> list[tuple[str, str, dict]]:
    """Select before model calls, using only input fields and L128 capacity."""
    counts = Counter()
    selected = []
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            family = row["family"]
            if counts[family] >= per_family or len(row["questions"]) != 1:
                continue
            question = next(iter(row["questions"].values()))
            if question["type"] not in ("choice", "noul"):
                continue
            try:
                adapted = adapt_question(question)
                request = build_request(as_text(row["state"]), adapted)
            except ValueError:
                continue
            if len(tokenizer(request.text, truncation=False)["input_ids"]) > 128:
                continue
            selected.append((row["id"], family, {"state": as_text(row["state"]), "question": adapted}))
            counts[family] += 1
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise ValueError(f"only {len(selected)} eligible rows, expected {limit}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=ROOT / "build" / "verdict_e8_L128_candidates25.mlpackage")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    source = verify_assets(required=("config.json", "tokenizer.json", "tokenizer_config.json", "calibrator.json"))
    tokenizer = AutoTokenizer.from_pretrained(source)
    config = json.loads((source / "config.json").read_text())
    class_token_index = int(config["class_token_index"])
    calibrator = load_calibrator(source)
    selected = select_rows(args.rows, tokenizer)
    packages = {
        "fp16": ROOT / "build" / "verdict_fp16_L128_candidates25.mlpackage",
        "candidate": args.candidate,
    }
    if args.candidate.name != "verdict_e8_L128_candidates25.mlpackage":
        raise ValueError("this fixed comparison is for the Verdict embedding-only e8 candidate")
    package_hashes = {
        "fp16": json.loads((ROOT / "reports" / "conversion-L128.json").read_text())["package_files_sha256"],
        "candidate": json.loads((ROOT / "reports" / "e8-L128-conversion.json").read_text())["package_files_sha256"],
    }
    for name, package in packages.items():
        for relative, expected in package_hashes[name].items():
            if sha256(package / relative) != expected:
                raise ValueError(f"{name} package changed since conversion: {relative}")
    models = {name: ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL) for name, path in packages.items()}
    first = selected[0][2]
    request = build_request(first["state"], first["question"])
    warmup = prepare(tokenizer, class_token_index, request.text, 128, 25)
    for model in models.values():
        for _ in range(2):
            model.predict(warmup)

    results = {}
    for name, model in models.items():
        rows, times = [], []
        for row_id, family, inputs in selected:
            row_times = []
            prediction = None
            for _ in range(args.repeats):
                started = time.perf_counter()
                request = build_request(inputs["state"], inputs["question"])
                arrays = prepare(tokenizer, class_token_index, request.text, 128, 25)
                output = model.predict(arrays)
                prediction = decode(output["logits"], request, calibrator)
                row_times.append((time.perf_counter() - started) * 1000)
            assert prediction is not None
            rows.append(
                {
                    "id": row_id,
                    "family": family,
                    "prediction": prediction,
                    "median_request_ms": statistics.median(row_times),
                }
            )
            times.extend(row_times)
        results[name] = {
            "p50_request_ms": statistics.median(times),
            "p95_request_ms": float(np.percentile(times, 95)),
            "rows": rows,
        }
    comparison = []
    for reference, candidate in zip(results["fp16"]["rows"], results["candidate"]["rows"]):
        fp16, alternative = reference["prediction"], candidate["prediction"]
        comparison.append(
            {
                "id": reference["id"],
                "family": reference["family"],
                "selection_agrees": fp16["selected_id"] == alternative["selected_id"],
                "abstention_agrees": fp16["is_abstention"] == alternative["is_abstention"],
                "max_calibrated_probability_error": max(
                    abs(fp16["probabilities"][key] - alternative["probabilities"][key]) for key in fp16["probabilities"]
                ),
            }
        )
    report = {
        "suite_file": args.rows.name,
        "suite_sha256": sha256(args.rows),
        "selection_protocol": "First two eligible L128 requests per family in suite order; no gold labels used",
        "chip": platform.processor(),
        "macos": platform.mac_ver()[0],
        "packages": {name: path.name for name, path in packages.items()},
        "package_files_sha256": package_hashes,
        "package_hashes_verified": True,
        "selected_manifest": [{"id": row_id, "family": family} for row_id, family, _ in selected],
        "power_audit_after_run": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.strip(),
        "requests": len(comparison),
        "selection_agreements": sum(row["selection_agrees"] for row in comparison),
        "abstention_agreements": sum(row["abstention_agrees"] for row in comparison),
        "worst_calibrated_probability_error": max(row["max_calibrated_probability_error"] for row in comparison),
        "timing": {
            name: {key: value for key, value in result.items() if key != "rows"} for name, result in results.items()
        },
        "rows": comparison,
    }
    report["passed"] = (
        report["selection_agreements"] == len(comparison)
        and report["abstention_agreements"] == len(comparison)
        and report["worst_calibrated_probability_error"] <= 0.02
    )
    path = ROOT / "reports" / f"{args.candidate.stem}-request-comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
