"""Measure full Laya requests on the fixed 16-question parity fixture."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

from assets import ROOT, checkpoint_dir, sha256
from calibration import calibrated_probabilities, temperature_for
from preprocessing import Shape, encode, package_name, prepare_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["english", "multilingual"], required=True)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--candidate", default="e8")
    parser.add_argument("--units", choices=["all", "ane"], default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cases", type=Path, default=ROOT / "fixtures" / "cases.json")
    args = parser.parse_args()
    source = checkpoint_dir(args.variant)
    tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer")
    config = json.loads((source / "rl_agent_config.json").read_text())
    shape = Shape(args.length, 32)
    fixture = json.loads(args.cases.read_text())
    selected = []
    for case in fixture:
        for question_id, question in case["questions"].items():
            try:
                encode(tokenizer, case["state"], question, shape, config["head_max_len"])
            except ValueError:
                continue
            selected.append((case["name"], question_id, case["state"], question))
    if not selected:
        raise ValueError("no fixture requests fit the selected length")
    units = ct.ComputeUnit.ALL if args.units == "all" else ct.ComputeUnit.CPU_AND_NE
    packages = {
        "fp16": ROOT / "build" / f"{package_name(args.variant, args.length, 32)}.mlpackage",
        "candidate": ROOT / "build" / f"{package_name(args.variant, args.length, 32, args.candidate)}.mlpackage",
    }
    models = {}
    load_seconds = {}
    for name, path in packages.items():
        start = time.perf_counter()
        models[name] = ct.models.MLModel(str(path), compute_units=units)
        load_seconds[name] = time.perf_counter() - start
    first = selected[0]
    ids, markers, qtype = encode(tokenizer, first[2], first[3], shape, config["head_max_len"])
    warmup = prepare_arrays(ids, markers, qtype, shape, tokenizer.pad_token_id)
    for model in models.values():
        for _ in range(2):
            model.predict(warmup)

    results = {}
    for name, model in models.items():
        rows = []
        durations = []
        for case_name, question_id, state, question in selected:
            samples = []
            prediction = None
            for _ in range(args.repeats):
                started = time.perf_counter()
                ids, markers, qtype = encode(tokenizer, state, question, shape, config["head_max_len"])
                arrays = prepare_arrays(ids, markers, qtype, shape, tokenizer.pad_token_id)
                output = model.predict(arrays)
                probabilities = calibrated_probabilities(
                    output["logits"][0, : len(markers)], temperature_for(config, qtype, len(markers))
                )
                prediction = {
                    "selected_index": int(np.argmax(probabilities)),
                    "probabilities": probabilities.astype(float).tolist(),
                    "action_probabilities": output["action_probabilities"][0].astype(float).tolist(),
                }
                samples.append((time.perf_counter() - started) * 1000)
            assert prediction is not None
            rows.append(
                {
                    "case": case_name,
                    "question": question_id,
                    "type": question["type"],
                    "tokens": len(ids),
                    "options": len(markers),
                    "prediction": prediction,
                    "median_request_ms": statistics.median(samples),
                }
            )
            durations.extend(samples)
        results[name] = {
            "p50_request_ms": statistics.median(durations),
            "p95_request_ms": float(np.percentile(durations, 95)),
            "rows": rows,
        }

    comparisons = []
    for fp16, candidate in zip(results["fp16"]["rows"], results["candidate"]["rows"]):
        p, q = fp16["prediction"], candidate["prediction"]
        comparisons.append(
            {
                "case": fp16["case"],
                "question": fp16["question"],
                "selection_agrees": p["selected_index"] == q["selected_index"],
                "max_calibrated_probability_error": float(
                    np.max(np.abs(np.asarray(p["probabilities"]) - np.asarray(q["probabilities"])))
                ),
                "max_action_probability_error": float(
                    np.max(np.abs(np.asarray(p["action_probabilities"]) - np.asarray(q["action_probabilities"])))
                ),
            }
        )
    report = {
        "variant": args.variant,
        "length": args.length,
        "units": args.units,
        "packages": {name: path.name for name, path in packages.items()},
        "cases_file": args.cases.name,
        "cases_sha256": sha256(args.cases),
        "selected_manifest": [{"case": name, "question": question_id} for name, question_id, _, _ in selected],
        "power_audit_after_run": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.strip(),
        "chip": platform.processor(),
        "macos": platform.mac_ver()[0],
        "questions": len(comparisons),
        "load_seconds": load_seconds,
        "timing": {
            name: {key: value for key, value in result.items() if key != "rows"} for name, result in results.items()
        },
        "selection_agreements": sum(row["selection_agrees"] for row in comparisons),
        "worst_calibrated_probability_error": max(row["max_calibrated_probability_error"] for row in comparisons),
        "worst_action_probability_error": max(row["max_action_probability_error"] for row in comparisons),
        "rows": comparisons,
    }
    report["passed_against_fp16"] = (
        report["selection_agreements"] == len(comparisons)
        and report["worst_calibrated_probability_error"] <= 0.02
        and report["worst_action_probability_error"] <= 0.02
    )
    target = ROOT / "reports" / f"request-{args.variant}-L{args.length}-{args.candidate}-{args.units}.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not report["passed_against_fp16"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
