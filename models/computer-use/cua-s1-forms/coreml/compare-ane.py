"""Compare baseline and ANE-gather latency in one process with a bounded ABBA schedule."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import coremltools as ct

from assets import ROOT, load_demo, sha256, verify_assets
from preprocessing import InputLimits, prepare_inputs
from verify import check_package, latency_summary, validate_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "build")
    parser.add_argument("--candidate", type=Path, default=ROOT / "build/ane-gather")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/ane-comparison.json")
    args = parser.parse_args()
    lock = verify_assets()
    all_rows = load_demo()
    row_ids = [0, 68, 130]
    rows = [all_rows[index] for index in row_ids]
    models, conversions = {}, {}
    for name, directory in [("baseline", args.baseline), ("candidate", args.candidate)]:
        package, conversion = check_package(directory)
        conversions[name] = conversion
        models[name] = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    if conversions["baseline"]["limits"] != conversions["candidate"]["limits"]:
        raise ValueError("Compare models with the same tensor interface")
    limits = InputLimits(**conversions["baseline"]["limits"])
    inputs = [prepare_inputs(row["context"], row["options"], limits) for row in rows]
    schedule = ["baseline", "candidate", "candidate", "baseline"]
    calls = []
    for block, name in enumerate(schedule):
        # Rewarm both policy transitions and repeated blocks equally.
        for iteration in range(12):
            for row_id, row, arrays in zip(row_ids, rows, inputs):
                started = time.perf_counter()
                output = models[name].predict(arrays)
                milliseconds = 1000 * (time.perf_counter() - started)
                _, probabilities = validate_prediction(output, len(row["options"]), limits.max_options)
                if int(probabilities.argmax()) != row["label"]:
                    raise ValueError(f"Wrong {name} prediction for row {row_id}")
                if iteration >= 2:
                    calls.append({"model": name, "block": block, "row": row_id, "milliseconds": milliseconds})
    summary = {
        name: latency_summary([call["milliseconds"] for call in calls if call["model"] == name])
        for name in models
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Matched warm model-call comparison, not a power or utilization measurement",
        "environment": {
            "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "macos": platform.mac_ver()[0],
            "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
            "python": platform.python_version(),
            "coremltools": ct.__version__,
        },
        "protocol": {
            "compute_units": "CPU_AND_NE",
            "rows": row_ids,
            "schedule": schedule,
            "warmup_passes_per_block": 2,
            "timed_passes_per_block": 10,
            "timing_scope": "Synchronous Python predict with pre-encoded inputs; validation outside timer",
            "model_loading": "Both models resident in one process; loads excluded and system caches retained",
        },
        "dataset_revision": lock["dataset_revision"],
        "dataset_sha256": sha256(ROOT / lock["evaluation_file"]),
        "package_files": {name: conversion["package_files"] for name, conversion in conversions.items()},
        "summary": summary,
        "candidate_over_baseline_median": summary["candidate"]["median_ms"] / summary["baseline"]["median_ms"],
        "correct_timed_predictions": len(calls),
        "calls": calls,
        "script_sha256": sha256(Path(__file__)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
