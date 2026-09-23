"""Kev request encoding for fixed Core ML token and option buckets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from kev.api import SystemOneRequest, to_record
from kev.data import materialize
from kev.model import encode


@dataclass(frozen=True)
class Shape:
    length: int = 128
    max_options: int = 32


def _pack(encode_record, tokenizer, record: dict, shape: Shape) -> tuple[dict[str, np.ndarray], dict]:
    """Pack one upstream Kev record into the fixed Core ML input tensors."""
    if len(record["questions"]) != 1:
        raise ValueError("Core ML export accepts exactly one question per call")
    full = encode_record(tokenizer, record, max_state=8192, max_branch=16384)
    state_tokens = full["seg"].count(0)
    branch_tokens = len(full["ids"]) - state_tokens
    available_state = shape.length - branch_tokens
    if available_state < 1:
        raise ValueError(f"question branch needs {branch_tokens} tokens and does not fit length {shape.length}")
    encoded = encode_record(tokenizer, record, max_state=available_state, max_branch=shape.length * 2)
    if len(encoded["ids"]) > shape.length:
        raise ValueError(f"encoded request has {len(encoded['ids'])} tokens for length {shape.length}")
    option_positions = encoded["opt_idx"][0]
    if len(option_positions) > shape.max_options:
        raise ValueError(f"request has {len(option_positions)} options; capacity is {shape.max_options}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = np.full((1, shape.length), pad_id, dtype=np.int32)
    attention_mask = np.zeros((1, shape.length), dtype=np.int32)
    decide_map = np.zeros((1, 1, shape.length), dtype=np.float32)
    option_map = np.zeros((1, shape.max_options, shape.length), dtype=np.float32)
    used = len(encoded["ids"])
    input_ids[0, :used] = encoded["ids"]
    attention_mask[0, :used] = 1
    decide_map[0, 0, encoded["decide_idx"][0]] = 1
    for option, position in enumerate(option_positions):
        option_map[0, option, position] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decide_map": decide_map,
        "option_map": option_map,
    }, encoded


def prepare_inputs(model, tokenizer, request: dict, shape: Shape) -> tuple[dict[str, np.ndarray], dict]:
    """Keep the labelled native-model verification path unchanged."""
    return _pack(model.encode, tokenizer, materialize(request), shape)


def prepare_runtime_inputs(
    tokenizer, request: dict, shape: Shape
) -> tuple[dict[str, np.ndarray], dict, list[dict], SystemOneRequest]:
    """Encode an unlabelled request using upstream's weight-free serving renderer."""
    parsed = SystemOneRequest.model_validate(request)
    if len(parsed.questions) != 1:
        raise ValueError("This Core ML package accepts exactly one question per call")
    record, metadata = to_record(parsed)

    def encode_record(tok, rec, **kwargs):
        return encode(tok, rec, option_isolation=False, **kwargs)

    arrays, encoded = _pack(encode_record, tokenizer, record, shape)
    return arrays, encoded, metadata, parsed
