"""Measure Verdict LUT8 against FP16 on fixed Decision Index requests without using gold labels."""

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


def selected_requests(rows_path: Path, tokenizer, limit: int, length: int):
    """First ten eligible rows per family, before inspecting either model's output."""
    family_counts = Counter()
    skipped = Counter()
    selected = []
    with gzip.open(rows_path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            family = row["family"]
            if family_counts[family] >= 10:
                continue
            if len(row["questions"]) != 1:
                skipped["multiple_questions"] += 1
                continue
            question = next(iter(row["questions"].values()))
            if question["type"] not in ("choice", "noul"):
                skipped["unsupported_question_type"] += 1
                continue
            try:
                request = build_request(as_text(row["state"]), adapt_question(question))
            except ValueError:
                skipped["invalid_or_overcapacity"] += 1
                continue
            tokens = len(tokenizer(request.text, truncation=False)["input_ids"])
            if tokens > length:
                skipped["overlength"] += 1
                continue
            selected.append((row["id"], family, request, tokens))
            family_counts[family] += 1
            if len(selected) >= limit:
                break
    return selected, family_counts, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="pinned public Decision Index selected-rows.jsonl.gz")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.length != 128:
        raise ValueError("the predeclared LUT8 validation protocol is L128 only")
    if args.limit != 100:
        raise ValueError("the predeclared validation manifest uses exactly 100 requests")
    source = verify_assets(required=("config.json", "tokenizer.json", "tokenizer_config.json", "calibrator.json"))
    tokenizer = AutoTokenizer.from_pretrained(source)
    config = json.loads((source / "config.json").read_text())
    calibrator = load_calibrator(source)
    selected, families, skipped = selected_requests(args.rows, tokenizer, args.limit, args.length)
    if len(selected) != args.limit:
        raise ValueError(f"only {len(selected)} eligible fixed requests; expected {args.limit}")

    packages = {
        "fp16": ROOT / "build" / "verdict_fp16_L128_candidates25.mlpackage",
        "lut8": ROOT / "build" / "verdict_lut8_kmeans_per_tensor_L128_candidates25.mlpackage",
    }
    models = {name: ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL) for name, path in packages.items()}
    for _, _, request, _ in selected[:2]:
        arrays = prepare(tokenizer, config["class_token_index"], request.text, args.length, 25)
        for model in models.values():
            model.predict(arrays)

    rows = []
    timings = {name: [] for name in models}
    for index, (row_id, family, request, tokens) in enumerate(selected):
        arrays = prepare(tokenizer, config["class_token_index"], request.text, args.length, 25)
        results = {}
        for name, model in models.items():
            output = model.predict(arrays)
            results[name] = decode(output["logits"], request, calibrator)
            if index < 20:
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    model.predict(arrays)
                    timings[name].append((time.perf_counter() - start) * 1000)
        reference = results["fp16"]
        compressed = results["lut8"]
        differences = [abs(reference["probabilities"][key] - compressed["probabilities"][key]) for key in request.ids]
        rows.append(
            {
                "id": row_id,
                "family": family,
                "tokens": tokens,
                "candidates": len(request.ids),
                "fp16_selected_id": reference["selected_id"],
                "lut8_selected_id": compressed["selected_id"],
                "selection_agrees": reference["selected_id"] == compressed["selected_id"],
                "abstention_agrees": reference["is_abstention"] == compressed["is_abstention"],
                "max_probability_error": max(differences),
            }
        )
    errors = np.array([row["max_probability_error"] for row in rows])
    selection_agreement = sum(row["selection_agrees"] for row in rows) / len(rows)
    abstention_agreement = sum(row["abstention_agrees"] for row in rows) / len(rows)
    gates = {
        "min_selection_agreement": 0.99,
        "min_abstention_agreement": 0.99,
        "max_p95_probability_error": 0.02,
        "max_worst_probability_error": 0.10,
    }
    report = {
        "suite_file": args.rows.name,
        "suite_sha256": sha256(args.rows),
        "selection_protocol": "First 10 eligible rows per family in suite order, 100 total; no gold labels used",
        "selected_row_ids": [row["id"] for row in rows],
        "families": dict(families),
        "skipped_before_limit": dict(skipped),
        "packages": {name: path.name for name, path in packages.items()},
        "hardware": {
            "chip": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip(),
            "macos": platform.mac_ver()[0],
        },
        "questions": len(rows),
        "selection_agreement": selection_agreement,
        "abstention_agreement": abstention_agreement,
        "p95_probability_error": float(np.percentile(errors, 95)),
        "worst_probability_error": float(errors.max()),
        "median_model_call_ms": {name: statistics.median(values) for name, values in timings.items()},
        "gates": gates,
        "rows": rows,
    }
    report["passed"] = (
        selection_agreement >= gates["min_selection_agreement"]
        and abstention_agreement >= gates["min_abstention_agreement"]
        and report["p95_probability_error"] <= gates["max_p95_probability_error"]
        and report["worst_probability_error"] <= gates["max_worst_probability_error"]
    )
    target = ROOT / "reports" / "lut8-L128-suite-parity.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key not in ("rows", "selected_row_ids")}, indent=2)
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
