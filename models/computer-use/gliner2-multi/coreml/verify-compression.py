"""Compare two GLiNER2 Core ML packages on selected real classification requests."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np

from preprocessing import load_processor, prepare_with_processor


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, min(len(values) - 1, math.ceil(p * len(values)) - 1))]


def median(cases: list[dict], field: str) -> float | None:
    return statistics.median(row[field] for row in cases) if cases else None


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compute-units", choices=["all", "cpu-ane"], default="all")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many eligible requests")
    parser.add_argument("--max-probability-error", type=float, default=0.02)
    args = parser.parse_args()

    units = ct.ComputeUnit.ALL if args.compute_units == "all" else ct.ComputeUnit.CPU_AND_NE
    processor = load_processor(str(args.tokenizer))
    load_start = time.perf_counter()
    reference = ct.models.MLModel(str(args.reference), compute_units=units)
    reference_load_ms = (time.perf_counter() - load_start) * 1000
    load_start = time.perf_counter()
    candidate = ct.models.MLModel(str(args.candidate), compute_units=units)
    candidate_load_ms = (time.perf_counter() - load_start) * 1000
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    prepared = []
    skipped = []
    for row in rows:
        if args.limit is not None and len(prepared) >= args.limit:
            break
        labels = [(description or key).strip() for key, description in row["options"]]
        prepare_start = time.perf_counter()
        try:
            arrays = prepare_with_processor(processor, row["state"], "decision", labels, args.length, args.max_options)
        except ValueError as error:
            skipped.append({"suite": row["suite"], "index": row["index"], "reason": str(error)})
            continue
        prepare_ms = (time.perf_counter() - prepare_start) * 1000

        prepared.append((row, labels, arrays, prepare_ms))

    if prepared:
        for model in (reference, candidate):
            model.predict(prepared[0][2])

    checked = []
    for index, (row, labels, arrays, prepare_ms) in enumerate(prepared):
        outputs = {}
        order = (("reference", reference), ("candidate", candidate))
        if index % 2:
            order = tuple(reversed(order))
        for name, model in order:
            start = time.perf_counter()
            probabilities = np.asarray(model.predict(arrays)["probabilities"])[0, : len(labels)]
            outputs[name] = (probabilities, (time.perf_counter() - start) * 1000)
        expected, reference_ms = outputs["reference"]
        actual, candidate_ms = outputs["candidate"]
        checked.append({
            "suite": row["suite"],
            "index": row["index"],
            "choice_agrees": int(expected.argmax()) == int(actual.argmax()),
            "max_probability_error": float(np.max(np.abs(expected - actual))),
            "prepare_ms": prepare_ms,
            "reference_ms": reference_ms,
            "candidate_ms": candidate_ms,
            "reference_total_ms": prepare_ms + reference_ms,
            "candidate_total_ms": prepare_ms + candidate_ms,
        })

    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "manifest": str(args.manifest),
        "compute_units": args.compute_units,
        "processor": platform.processor(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "power_audit_after_run": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.strip(),
        "reference_load_ms": reference_load_ms,
        "candidate_load_ms": candidate_load_ms,
        "warmup_calls_per_model": 1 if prepared else 0,
        "timed_order": "alternating reference-first/candidate-first by request",
        "checked": len(checked),
        "skipped": skipped,
        "choice_agreement": sum(row["choice_agrees"] for row in checked),
        "max_probability_error": max((row["max_probability_error"] for row in checked), default=None),
        "probability_error_limit": args.max_probability_error,
        "reference_p50_ms": median(checked, "reference_ms"),
        "candidate_p50_ms": median(checked, "candidate_ms"),
        "reference_p95_ms": percentile([row["reference_ms"] for row in checked], 0.95),
        "candidate_p95_ms": percentile([row["candidate_ms"] for row in checked], 0.95),
        "reference_total_p50_ms": median(checked, "reference_total_ms"),
        "candidate_total_p50_ms": median(checked, "candidate_total_ms"),
        "cases": checked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key not in ("cases", "skipped")}
    summary["skipped_count"] = len(skipped)
    print(json.dumps(summary, indent=2))
    if not checked or report["choice_agreement"] != len(checked):
        raise SystemExit("candidate changed a checked decision")
    if report["max_probability_error"] > args.max_probability_error:
        raise SystemExit("candidate exceeded the selected-request probability error limit")


if __name__ == "__main__":
    run()
