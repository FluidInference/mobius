"""The released app.py choice-scorer rendering, padding, and calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence

MAX_LENGTH = 256
TEMPERATURE = 2.35


def encode(tokenizer, state: str, question: str, option: str, max_length: int = MAX_LENGTH) -> list[int]:
    """Keep the question/option tail; truncate the state from its end, as upstream does."""
    tail = tokenizer("\n\nQuestion:\n" + question + "\n\nOption:\n" + option, add_special_tokens=False)["input_ids"]
    if len(tail) >= max_length:
        return tail[-max_length:]
    head = tokenizer("State:\n" + state, add_special_tokens=False)["input_ids"]
    return head[: max_length - len(tail)] + tail


def encode_options(tokenizer, state: str, question: str, options: Sequence[str]) -> list[list[int]]:
    if len(options) < 2:
        raise ValueError("the native app requires at least two options")
    if any(not option.strip() for option in options):
        raise ValueError("options must be nonempty")
    return [encode(tokenizer, state, question, option) for option in options]


def pad_batch(
    sequences: Sequence[Sequence[int]], pad_id: int, length: int | None = None
) -> tuple[list[list[int]], list[list[int]]]:
    """Right-pad to the longest candidate (or fixed export length), with true token masks."""
    if not sequences:
        raise ValueError("empty candidate batch")
    width = min(max(map(len, sequences)), MAX_LENGTH) if length is None else length
    if width > MAX_LENGTH or width < max(map(len, sequences)):
        raise ValueError("invalid padding length")
    ids = [list(row) + [pad_id] * (width - len(row)) for row in sequences]
    masks = [[1] * len(row) + [0] * (width - len(row)) for row in sequences]
    return ids, masks


def softmax(logits: Sequence[float], temperature: float = TEMPERATURE) -> list[float]:
    if not logits or temperature <= 0:
        raise ValueError("need logits and a positive temperature")
    scaled = [float(value) / temperature for value in logits]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def choose(options: Sequence[str], logits: Sequence[float]) -> dict:
    if len(options) != len(logits):
        raise ValueError("one scalar logit is required per option")
    probabilities = softmax(logits)
    index = max(range(len(options)), key=probabilities.__getitem__)
    return {"selected_index": index, "selected_option": options[index], "probabilities": probabilities}
