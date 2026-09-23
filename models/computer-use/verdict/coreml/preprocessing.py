"""Tensorize Verdict's native rendered text for one fixed Core ML bucket."""

from __future__ import annotations

import numpy as np


def prepare(
    tokenizer, class_token_index: int, rendered: str, length: int, max_candidates: int
) -> dict[str, np.ndarray]:
    full = tokenizer(rendered, truncation=False)
    if len(full["input_ids"]) > length:
        raise ValueError(f"Verdict prompt needs {len(full['input_ids'])} tokens; L{length} has no room")
    encoded = tokenizer(rendered, truncation=False, padding="max_length", max_length=length, return_tensors="np")
    ids = encoded["input_ids"].astype(np.int32)
    positions = np.flatnonzero(ids[0] == class_token_index)
    if len(positions) > max_candidates:
        raise ValueError("candidate markers exceed exported head capacity")
    markers = np.zeros((1, max_candidates, length), dtype=np.float32)
    for row, position in enumerate(positions):
        markers[0, row, position] = 1.0
    return {
        "input_ids": ids,
        "attention_mask": encoded["attention_mask"].astype(np.int32),
        "class_marker_map": markers,
    }
