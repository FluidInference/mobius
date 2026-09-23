"""Verdict's pinned formatting, calibration and typed output contract.

The rendering and decoding match Verdict-open-jev/core/{formatting,engine_encoder}.py
at 30f15564821626ca5c1ad5b2638c4eb7078787dd. The encoder is loaded from
heman10x/rlcd-modernbert-151m at the revision in assets.lock.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ABSTAIN_ID = "__insufficient_evidence__"
ABSTAIN_DESCRIPTION = "insufficient evidence"
MAX_SUBSTANTIVE_OPTIONS = 24
MAX_CANDIDATES = 25


@dataclass(frozen=True)
class Request:
    kind: str
    text: str
    labels: tuple[str, ...]
    ids: tuple[str, ...]
    values: tuple[float, ...] = ()


def build_request(context: str, question: dict[str, Any]) -> Request:
    """Render one native typed query without dropping trained abstention."""
    kind = question["type"]
    if kind == "choice":
        options = question["options"]
        if not 1 <= len(options) <= MAX_SUBSTANTIVE_OPTIONS:
            raise ValueError("Verdict supports 1 to 24 substantive choice options")
        labels = tuple(f"It is {option['description']}" for option in options)
        ids = tuple(option["id"] for option in options)
        query = question["question"]
        text = f"Question: {query}\n\nContext:\n{context}"
        values: tuple[float, ...] = ()
    elif kind == "score":
        levels = question["levels"]
        if not 1 <= len(levels) <= MAX_SUBSTANTIVE_OPTIONS:
            raise ValueError("Verdict supports 1 to 24 substantive score levels")
        labels = tuple(f"{level['description']} (Value: {level['value']})" for level in levels)
        ids = tuple(level["id"] for level in levels)
        values = tuple(float(level["value"]) for level in levels)
        query = question["question"]
        text = f"Question: {query}\n\nContext:\n{context}"
    elif kind == "noul":
        proposition = question["proposition"]
        labels = (f"true: {proposition}", f"false: not {proposition}")
        ids = ("true", "false")
        values = ()
        text = f"Context:\n{context}\n\nEvaluate proposition: {proposition}"
        query = ""
    else:
        raise ValueError(f"unknown Verdict question type: {kind}")
    labels += (ABSTAIN_DESCRIPTION,)
    ids += (ABSTAIN_ID,)
    if len(ids) != len(set(ids)):
        raise ValueError("Verdict candidate IDs must be unique and cannot use the abstention ID")
    prefix = "".join(f"<<LABEL>>{label}" for label in labels)
    return Request(kind, f"{prefix}<<SEP>>{text}", labels, ids, values)


def temperature(calibrator: dict[str, Any], count: int) -> float:
    per_k = calibrator.get("per_k", {})
    value = float(per_k.get(str(count), calibrator["temperature"]))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid trained temperature")
    return value


def decode(logits: np.ndarray, request: Request, calibrator: dict[str, Any]) -> dict[str, Any]:
    """Apply the released per-K calibration and preserve native abstention."""
    k = len(request.ids)
    z = np.asarray(logits, dtype=np.float64).reshape(-1)[:k]
    if len(z) != k or not np.isfinite(z).all():
        raise ValueError("invalid Verdict logits")
    z = z / temperature(calibrator, k)
    p = np.exp(z - z.max())
    p /= p.sum()
    distribution = dict(zip(request.ids, map(float, p)))
    selected_id = request.ids[int(p.argmax())]
    abstained = selected_id == ABSTAIN_ID
    entropy = -sum(float(value) * math.log(float(value)) for value in p if value > 1e-12)
    concentration = max(0.0, min(1.0, 1.0 - entropy / math.log(k))) if k > 1 else 1.0
    result: dict[str, Any] = {
        "type": request.kind,
        "selected_id": selected_id,
        "selected_probability": distribution[selected_id],
        "probabilities": distribution,
        "concentration": concentration,
        "is_abstention": abstained,
        "p_abstain": distribution[ABSTAIN_ID],
        "calibration_status": "calibrated_for_scope",
    }
    if request.kind == "choice":
        result["choice"] = selected_id
    elif request.kind == "noul":
        substantive = distribution["true"] + distribution["false"]
        result["noul"] = distribution["true"] / substantive if substantive and not abstained else None
        result["selected_outcome"] = selected_id
    else:
        substantive = sum(distribution[key] for key in request.ids[:-1])
        result["score"] = (
            sum(value * distribution[key] / substantive for key, value in zip(request.ids[:-1], request.values))
            if substantive and not abstained
            else None
        )
        result["selected_level_id"] = selected_id
        result["selected_value"] = (
            request.values[request.ids.index(selected_id)] if not abstained else None
        )
    return result


def load_calibrator(source: Path) -> dict[str, Any]:
    return json.loads((source / "calibrator.json").read_text())
