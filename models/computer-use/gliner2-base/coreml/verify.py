"""Compare the pinned GLiNER2 native classifier with its Core ML export."""
import argparse
import json
import statistics
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliner2 import AutoExtractor

from preprocessing import prepare_classification

MODEL_ID = "fastino/gliner2.5-base-v1"
MODEL_REVISION = "1a8bc24e00dc7300b9017c81d63e3dcdabb26596"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="build/gliner2_base_classification_fp16_L128_K8.mlpackage")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument("--out", default="build/verify.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = Path.home() / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1/snapshots" / MODEL_REVISION
    native = AutoExtractor.from_pretrained(str(source), map_location="cpu").eval()
    model = ct.models.MLModel(args.package, compute_units=ct.ComputeUnit.ALL)
    counts = {"checked": 0, "too_many_options": 0, "too_long": 0, "duplicate_labels": 0, "mismatches": 0}
    errors = []
    latencies = []
    failures = []
    for line in Path(args.suite).open():
        row = json.loads(line)
        options = row.get("options") or []
        if len(options) > args.max_options or not options:
            counts["too_many_options"] += 1
            continue
        labels = [(description or key).strip() for key, description in options]
        if len(set(labels)) != len(labels):
            counts["duplicate_labels"] += 1
            continue
        text = row["state"]
        task = "decision"
        try:
            arrays = prepare_classification(native, text, task, labels, args.length, args.max_options)
        except ValueError:
            counts["too_long"] += 1
            continue
        expected = native.classify_text(text, {task: labels}, include_confidence=True, max_len=args.length)[task]
        start = time.perf_counter()
        prediction = model.predict(arrays)
        latencies.append((time.perf_counter() - start) * 1000)
        probabilities = np.asarray(prediction["probabilities"])[0, : len(labels)]
        chosen = labels[int(probabilities.argmax())]
        error = abs(float(probabilities.max()) - float(expected["confidence"]))
        errors.append(error)
        counts["checked"] += 1
        if chosen != expected["label"]:
            counts["mismatches"] += 1
            failures.append({"suite": row["suite"], "index": row["index"], "native": expected, "coreml": chosen})
        if counts["checked"] >= args.limit:
            break
    result = {
        "model": MODEL_ID, "revision": MODEL_REVISION, "package": args.package,
        "selected_manifest": "first eligible rows in source suite order; no gold labels used",
        "counts": counts, "maximum_confidence_error": max(errors, default=None),
        "mean_confidence_error": statistics.mean(errors) if errors else None,
        "model_call_p50_ms": statistics.median(latencies[1:]) if len(latencies) > 1 else None,
        "model_call_p95_ms": sorted(latencies[1:])[int(0.95 * (len(latencies) - 1))] if len(latencies) > 1 else None,
        "failures": failures[:20],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if counts["checked"] == 0 or counts["mismatches"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
