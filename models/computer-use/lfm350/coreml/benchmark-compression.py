"""Pair LFM FP16 and compressed full-value scoring on three pinned real upstream requests."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

from decision import source_path
from preprocessing import Shape, batch_arrays, prepare_candidates


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(source_path()))
    from rlcd.tasks import CASES  # noqa: E402

    selected = CASES[:3]
    shape = Shape()
    tokenizer = AutoTokenizer.from_pretrained(source_path())
    units = ct.ComputeUnit.ALL
    models = {
        "reference": ct.models.MLModel(str(args.reference), compute_units=units),
        "candidate": ct.models.MLModel(str(args.candidate), compute_units=units),
    }

    def infer(model, context, schema):
        started = time.perf_counter()
        candidates = prepare_candidates(tokenizer, context, schema, shape)
        groups = [
            batch_arrays(tokenizer, candidates[offset : offset + shape.candidates], shape)
            for offset in range(0, len(candidates), shape.candidates)
        ]
        prepare_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        scores = []
        for group in groups:
            scores.extend(np.asarray(model.predict(group)["scores"]).reshape(-1).tolist())
        model_ms = (time.perf_counter() - started) * 1000
        return scores[: len(candidates)], prepare_ms, model_ms

    for model in models.values():
        infer(model, selected[0][2], selected[0][1])
    timed = {name: [] for name in models}
    differences = []
    for repeat in range(5):
        for case_id, schema, context, _ in selected:
            order = ("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference")
            outputs = {}
            for name in order:
                scores, prepare_ms, model_ms = infer(models[name], context, schema)
                timed[name].append({"case_id": case_id, "prepare_ms": prepare_ms, "model_ms": model_ms,
                                    "total_ms": prepare_ms + model_ms})
                outputs[name] = scores
            differences.append(max(abs(a - b) for a, b in zip(outputs["reference"], outputs["candidate"])))
    report = {
        "reference": str(args.reference), "candidate": str(args.candidate),
        "selected_case_ids": [case[0] for case in selected],
        "protocol": "first three upstream cases, five repetitions, alternating model order, warmup once per model",
        "macos": platform.mac_ver()[0],
        "power": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.strip(),
        "max_paired_score_difference": max(differences),
        "timing": {
            name: {metric: {"p50_ms": statistics.median(row[metric] for row in rows),
                            "p95_ms": percentile([row[metric] for row in rows], 0.95)}
                   for metric in ("prepare_ms", "model_ms", "total_ms")}
            for name, rows in timed.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
