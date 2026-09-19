"""Pinned published synthetic test data and provider-independent result summaries."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter

import numpy as np

from assets import ROOT, sha256
from preprocessing import InputLimits

MANIFEST = ROOT / "synthetic-test.lock.json"


def load_test(*, download: bool = False) -> tuple[list[dict], dict]:
    """Read the exact public test file; never generate, filter, or resample examples."""
    manifest = json.loads(MANIFEST.read_text())
    path = ROOT / manifest["path"]
    if not path.exists() and download:
        with urllib.request.urlopen(manifest["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
            raise ValueError("Downloaded synthetic test checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(path)
    if sha256(path) != manifest["sha256"] or path.stat().st_size != manifest["bytes"]:
        raise ValueError("Published synthetic test checksum/size mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != manifest["rows"]:
        raise ValueError("Published synthetic test row count mismatch")
    return rows, manifest


def action(option: str) -> str:
    if option.startswith("fill "):
        return "fill"
    if option in {"check", "click", "skip"}:
        return option
    raise ValueError("Unknown action candidate")


def inspect_rows(rows: list[dict], limits: InputLimits) -> dict:
    """Require every decision to fit unchanged; detect schema and annotation errors."""
    if not rows:
        raise ValueError("The test manifest must not be empty")
    context_lengths, option_lengths, option_counts, actions, seeds = [], [], [], [], set()
    for index, row in enumerate(rows):
        context, options, label = row["context"], row["options"], row["label"]
        if not isinstance(context, str) or not context:
            raise ValueError(f"Invalid context in row {index}")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"Invalid choices in row {index}")
        if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(options):
            raise ValueError(f"Invalid label in row {index}")
        if any(not isinstance(option, str) or not option for option in options):
            raise ValueError(f"Invalid option in row {index}")
        if options[-3:] != ["check", "click", "skip"]:
            raise ValueError(f"Missing fixed actions in row {index}")
        context_lengths.append(len(context.encode("utf-8")))
        option_lengths.extend(len(option.encode("utf-8")) for option in options)
        option_counts.append(len(options))
        expected_action = action(options[label])
        if row["meta"]["action"] != expected_action:
            raise ValueError(f"Action annotation disagrees with the label in row {index}")
        actions.append(expected_action)
        seeds.add(row["meta"]["seed"])
    maxima = {
        "context_bytes": max(context_lengths),
        "option_bytes": max(option_lengths),
        "max_options": max(option_counts),
    }
    if any(value > getattr(limits, key) for key, value in maxima.items()):
        raise ValueError("The full test split does not fit unchanged; do not filter or truncate it")
    return {
        "rows": len(rows),
        "unique_episode_seeds": len(seeds),
        "maxima": maxima,
        "per_action": dict(sorted(Counter(actions).items())),
        "option_count_histogram": dict(sorted(Counter(option_counts).items())),
        "excluded_rows": 0,
        "truncated_contexts": 0,
        "truncated_options": 0,
    }


def calibration(confidence: list[float], gold_probability: list[float], correct: list[bool], bins: int = 15) -> dict:
    """Report NLL and equal-width ECE; this measures scores without calibrating them."""
    conf, gold, hits = np.asarray(confidence), np.asarray(gold_probability), np.asarray(correct)
    if not len(conf) or len(conf) != len(gold) or len(conf) != len(hits):
        raise ValueError("Calibration arrays must cover the same nonempty set of rows")
    if not np.isfinite(conf).all() or not np.isfinite(gold).all():
        raise ValueError("Calibration probabilities must be finite")
    if np.any(conf < 0) or np.any(conf > 1) or np.any(gold < 0) or np.any(gold > 1) or bins < 1:
        raise ValueError("Invalid probabilities or bin count")
    assignments = np.minimum((conf * bins).astype(int), bins - 1)
    details, ece = [], 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if not count:
            continue
        accuracy, mean_confidence = float(hits[mask].mean()), float(conf[mask].mean())
        ece += count / len(conf) * abs(accuracy - mean_confidence)
        details.append({"bin": index, "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    return {
        "nll": float(-np.log(np.maximum(gold, np.finfo(np.float32).tiny)).mean()),
        "ece": ece,
        "equal_width_bins": bins,
        "bins": details,
        "nll_probability_floor": float(np.finfo(np.float32).tiny),
    }


def paired_outcomes(labels: list[int], reference: list[int], candidate: list[int]) -> dict:
    """Separate harmful flips, corrected mistakes, and different wrong predictions."""
    if not labels or len(labels) != len(reference) or len(labels) != len(candidate):
        raise ValueError("Paired predictions must cover every row")
    gold, base, converted = np.asarray(labels), np.asarray(reference), np.asarray(candidate)
    reference_right, candidate_right = base == gold, converted == gold
    return {
        "rows": len(labels),
        "argmax_agreement": int((base == converted).sum()),
        "agreement_rate": float((base == converted).mean()),
        "reference_correct_candidate_wrong": int((reference_right & ~candidate_right).sum()),
        "reference_wrong_candidate_correct": int((~reference_right & candidate_right).sum()),
        "both_wrong": int((~reference_right & ~candidate_right).sum()),
        "disagreement_rows": np.flatnonzero(base != converted).tolist(),
    }
