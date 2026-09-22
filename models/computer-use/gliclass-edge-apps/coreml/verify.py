"""Verify bucketed Core ML inference against PyTorch on the application suite."""

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
import torch
from gliclass import GLiClassModel
from gliclass.pipeline import UniEncoderZeroShotClassificationPipeline
from transformers import AutoTokenizer

from suite_mapping import ORDER, labels_for

warnings.filterwarnings("ignore")


def select_bucket(token_count: int, bucket_lengths: list[int]) -> int:
    """Return the smallest bucket that fits, truncating to the largest if necessary."""
    for length in bucket_lengths:
        if token_count <= length:
            return length
    return bucket_lengths[-1]


def make_arrays(
    tokenizer,
    model,
    rendered: str,
    option_count: int,
    length: int,
    max_options: int,
) -> dict[str, np.ndarray]:
    encoded = tokenizer(
        rendered,
        truncation=True,
        max_length=length,
        padding="max_length",
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int32)
    attention_mask = encoded["attention_mask"].astype(np.int32)
    positions = np.flatnonzero(input_ids[0] == model.config.class_token_index)
    if len(positions) != option_count:
        raise ValueError(f"options do not fit L{length}: expected {option_count} markers, found {len(positions)}")
    marker_map = np.zeros((1, max_options, length), dtype=np.float32)
    for index, position in enumerate(positions):
        marker_map[0, index, position] = 1.0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "class_marker_map": marker_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="build/checkpoint-v2")
    parser.add_argument("--packages", default="build/coreml")
    parser.add_argument("--suites", default="../../laya/coreml/benchmark/suites.jsonl")
    parser.add_argument("--lengths", default="128,256,512")
    parser.add_argument("--max-options", type=int, default=25)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--out", default="reports/coreml-verify.json")
    args = parser.parse_args()

    lengths = sorted({int(value) for value in args.lengths.split(",")})
    model = GLiClassModel.from_pretrained(args.model).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    formatter = UniEncoderZeroShotClassificationPipeline(
        model,
        tokenizer,
        max_classes=args.max_options,
        max_length=lengths[-1],
        classification_type="single-label",
        device=args.device,
        progress_bar=False,
    )
    packages = {
        length: ct.models.MLModel(
            str(Path(args.packages) / f"gliclass_edge_apps_fp16_L{length}_options{args.max_options}.mlpackage"),
            compute_units=ct.ComputeUnit.ALL,
        )
        for length in lengths
    }

    rows = [json.loads(line) for line in Path(args.suites).read_text().splitlines()]
    per_suite: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    bucket_counts: dict[int, int] = defaultdict(int)
    disagreements = []
    absolute_errors = []
    coreml_times = []

    for index, row in enumerate(rows):
        choices = labels_for(row, "description")
        labels = [label for label, _ in choices]
        label_to_gold = [gold for _, gold in choices]
        rendered = formatter.prepare_input(row["state"], labels, prompt=row["instructions"])
        token_count = len(tokenizer(rendered, truncation=False)["input_ids"])
        length = select_bucket(token_count, lengths)
        arrays = make_arrays(tokenizer, model, rendered, len(labels), length, args.max_options)

        tick = time.perf_counter()
        coreml_output = packages[length].predict(arrays)
        coreml_times.append((time.perf_counter() - tick) * 1000)
        coreml_logits = np.asarray(coreml_output["logits"])[0, : len(labels)]
        coreml_choice = int(coreml_logits.argmax())

        tensors = {
            "input_ids": torch.from_numpy(arrays["input_ids"]).long().to(args.device),
            "attention_mask": torch.from_numpy(arrays["attention_mask"]).long().to(args.device),
        }
        with torch.no_grad():
            reference_logits = model(**tensors, max_num_classes=len(labels)).logits.float().cpu().numpy()[0]
        reference_choice = int(reference_logits.argmax())
        absolute_errors.extend(np.abs(coreml_logits - reference_logits).tolist())
        if coreml_choice != reference_choice and len(disagreements) < 20:
            disagreements.append(
                {
                    "row": index,
                    "suite": row["suite"],
                    "bucket": length,
                    "coreml": coreml_choice,
                    "pytorch": reference_choice,
                }
            )

        result = per_suite[row["suite"]]
        result["n"] += 1
        result["ok"] += int(label_to_gold[coreml_choice] == row["gold"])
        bucket_counts[length] += 1
        if (index + 1) % 250 == 0:
            print(f"{index + 1}/{len(rows)}", flush=True)

    accuracies = {suite: values["ok"] / values["n"] for suite, values in per_suite.items()}
    sorted_times = sorted(coreml_times)
    output = {
        "model": args.model,
        "rows": len(rows),
        "compute_units": "all",
        "bucket_counts": {str(key): value for key, value in sorted(bucket_counts.items())},
        "macro": statistics.mean(accuracies.values()),
        "micro": sum(values["ok"] for values in per_suite.values()) / len(rows),
        "coreml_p50_ms_python": statistics.median(sorted_times),
        "coreml_p95_ms_python": sorted_times[int(0.95 * len(sorted_times)) - 1],
        "argmax_disagreements": len(disagreements),
        "disagreement_examples": disagreements,
        "max_logit_absolute_error": max(absolute_errors),
        "mean_logit_absolute_error": statistics.mean(absolute_errors),
        "suites": {key: {**per_suite[key], "accuracy": accuracies[key]} for key in ORDER},
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
