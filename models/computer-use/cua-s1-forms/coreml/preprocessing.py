"""Host-side UTF-8 byte encoding for the fixed-shape Core ML interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputLimits:
    context_bytes: int = 224
    option_bytes: int = 96
    max_options: int = 32

    def __post_init__(self):
        if self.context_bytes < 1 or self.option_bytes < 1 or self.max_options < 2:
            raise ValueError("Positive byte limits and at least two option slots are required")


def byte_ids(text: str, length: int) -> np.ndarray:
    """Match upstream: truncate UTF-8 bytes, offset byte values by one, zero-pad."""
    data = text.encode("utf-8", errors="replace")[:length]
    return np.frombuffer(data, dtype=np.uint8).astype(np.int32) + 1


def prepare_inputs(context: str, options: list[str] | tuple[str, ...], limits: InputLimits) -> dict[str, np.ndarray]:
    """Encode one decision; options are never silently removed to fit the model."""
    if not isinstance(context, str) or not context:
        raise ValueError("context must be a nonempty string")
    if not isinstance(options, (list, tuple)) or len(options) < 2:
        raise ValueError("At least two options are required")
    if len(options) > limits.max_options:
        raise ValueError(f"{len(options)} options exceed {limits.max_options}; re-export with a larger --max-options")
    if any(not isinstance(option, str) or not option for option in options):
        raise ValueError("Options must be nonempty strings")
    context_ids = np.zeros((1, limits.context_bytes), dtype=np.int32)
    option_ids = np.zeros((1, limits.max_options, limits.option_bytes), dtype=np.int32)
    option_mask = np.zeros((1, limits.max_options), dtype=np.int32)
    tokens = byte_ids(context, limits.context_bytes)
    context_ids[0, : len(tokens)] = tokens
    for index, option in enumerate(options):
        tokens = byte_ids(option, limits.option_bytes)
        option_ids[0, index, : len(tokens)] = tokens
    option_mask[0, : len(options)] = 1
    return {"context_ids": context_ids, "option_ids": option_ids, "option_mask": option_mask}
