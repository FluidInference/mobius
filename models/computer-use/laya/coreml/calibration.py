"""Standalone Laya host calibration for raw Core ML option logits."""

from __future__ import annotations

import numpy as np

QUESTION_TYPES = ("choice", "score", "noul")


def temperature_for(config: dict, question_type: int | str, option_count: int) -> float:
    """Match the released Laya Agent's temperature bucket selection."""
    if isinstance(question_type, str):
        index = QUESTION_TYPES.index(question_type)
    else:
        index = int(question_type)
    if index not in range(len(QUESTION_TYPES)):
        raise ValueError("invalid Laya question type")
    size = "2" if option_count <= 2 else "3-5" if option_count <= 5 else "6-10" if option_count <= 10 else "11+"
    bucket = f"{QUESTION_TYPES[index]}:{size}"
    base = config.get("temperature", [1.0, 1.0, 1.0])[index]
    value = config.get("temperature_by_options", {}).get(bucket, base)
    return max(1e-3, float(value))


def calibrated_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply the selected temperature to only the provided option logits."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max()
    probabilities = np.exp(z)
    return probabilities / probabilities.sum()
