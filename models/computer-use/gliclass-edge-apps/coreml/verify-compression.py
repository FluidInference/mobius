"""Compare a compressed GLiClass Core ML bucket with its FP16 source."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from collections import defaultdict
from pathlib import Path

import coremltools as ct
import numpy as np
from gliclass import GLiClassModel
from gliclass.pipeline import UniEncoderZeroShotClassificationPipeline
from transformers import AutoTokenizer

from suite_mapping import labels_for
from verify import make_arrays

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="build/checkpoint-v2")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--suites", default="../../laya/coreml/benchmark/suites.jsonl")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = GLiClassModel.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    formatter = UniEncoderZeroShotClassificationPipeline(
        source,
        tokenizer,
        max_classes=args.max_options,
        max_length=args.length,
        classification_type="single-label",
        device="cpu",
        progress_bar=False,
    )
    reference = ct.models.MLModel(args.reference, compute_units=ct.ComputeUnit.CPU_AND_NE)
    candidate = ct.models.MLModel(args.candidate, compute_units=ct.ComputeUnit.CPU_AND_NE)

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "reference_correct": 0, "candidate_correct": 0, "agreement": 0}
    )
    absolute_errors: list[float] = []
    candidate_times: list[float] = []
    unsupported = 0
    rows = [json.loads(line) for line in Path(args.suites).read_text().splitlines()]
    for index, row in enumerate(rows):
        choices = labels_for(row, "description")
        labels = [label for label, _ in choices]
        gold = [value for _, value in choices]
        rendered = formatter.prepare_input(row["state"], labels, prompt=row["instructions"])
        try:
            arrays = make_arrays(tokenizer, source, rendered, len(labels), args.length, args.max_options)
        except ValueError:
            unsupported += 1
            continue
        reference_logits = np.asarray(reference.predict(arrays)["logits"])[0, : len(labels)]
        started = time.perf_counter()
        candidate_logits = np.asarray(candidate.predict(arrays)["logits"])[0, : len(labels)]
        candidate_times.append((time.perf_counter() - started) * 1000)
        reference_choice = int(reference_logits.argmax())
        candidate_choice = int(candidate_logits.argmax())
        result = counts[row["suite"]]
        result["rows"] += 1
        result["reference_correct"] += int(gold[reference_choice] == row["gold"])
        result["candidate_correct"] += int(gold[candidate_choice] == row["gold"])
        result["agreement"] += int(candidate_choice == reference_choice)
        absolute_errors.extend(np.abs(reference_logits - candidate_logits).tolist())
        if (index + 1) % 500 == 0:
            print(f"{index + 1}/{len(rows)}", flush=True)

    supported = sum(value["rows"] for value in counts.values())
    output = {
        "reference": args.reference,
        "candidate": args.candidate,
        "total_rows": len(rows),
        "supported_rows": supported,
        "unsupported_rows": unsupported,
        "argmax_agreement": sum(value["agreement"] for value in counts.values()) / supported,
        "reference_micro_accuracy": sum(value["reference_correct"] for value in counts.values()) / supported,
        "candidate_micro_accuracy": sum(value["candidate_correct"] for value in counts.values()) / supported,
        "mean_logit_absolute_error": statistics.mean(absolute_errors),
        "max_logit_absolute_error": max(absolute_errors),
        "candidate_median_ms_python": statistics.median(candidate_times),
        "candidate_p95_ms_python": sorted(candidate_times)[int(0.95 * len(candidate_times)) - 1],
        "suites": {
            suite: {
                **value,
                "reference_accuracy": value["reference_correct"] / value["rows"],
                "candidate_accuracy": value["candidate_correct"] / value["rows"],
            }
            for suite, value in counts.items()
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
