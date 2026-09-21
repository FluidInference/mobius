"""Fixed-shape input preparation that reuses the upstream laya sequence builder."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from laya.common import QTYPES, build_sequence, render_options

MAX_OPTIONS = 32


def package_name(variant: str, length: int, max_options: int, precision: str = "fp16") -> str:
    return f"laya_{variant}_{precision}_L{length}_options{max_options}"


@dataclass(frozen=True)
class Shape:
    length: int
    max_options: int = MAX_OPTIONS


def to_internal(question: dict) -> dict:
    """Mirror laya.agent.Agent._to_internal."""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice" and isinstance(criteria, list):
        criteria = {c: None for c in criteria}
    instructions = question["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions)
    return {"t": kind, "ins": instructions, "crit": criteria}


def encode(tok, state, question: dict, shape: Shape, head_max_len: int) -> tuple[list[int], list[int], int]:
    """Token ids, marker positions and question type for one question at the bucket length."""
    q = to_internal(question)
    ids, markers = build_sequence(tok, state, q, shape.length, head_max_len)
    if len(markers) != len(render_options(q)):
        raise ValueError(f"options exceed head_max_len={head_max_len}")
    if len(markers) > shape.max_options:
        raise ValueError(f"{len(markers)} options exceed the exported capacity {shape.max_options}")
    return ids, markers, QTYPES[q["t"]]


def prepare_arrays(ids: list[int], markers: list[int], qtype: int, shape: Shape, pad_id: int) -> dict[str, np.ndarray]:
    length, k = shape.length, shape.max_options
    if len(ids) > length:
        raise ValueError(f"sequence of {len(ids)} tokens exceeds bucket length {length}")
    input_ids = np.full((1, length), pad_id, dtype=np.int32)
    input_ids[0, : len(ids)] = ids
    attention_mask = np.zeros((1, length), dtype=np.int32)
    attention_mask[0, : len(ids)] = 1
    marker_map = np.zeros((1, k, length), dtype=np.float32)
    for row, position in enumerate(markers):
        marker_map[0, row, position] = 1.0
    question_type = np.zeros((1, 3), dtype=np.float32)
    question_type[0, qtype] = 1.0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_map": marker_map,
        "question_type": question_type,
    }


def prepare_inputs(tok, state, question: dict, shape: Shape, head_max_len: int) -> tuple[dict[str, np.ndarray], int]:
    ids, markers, qtype = encode(tok, state, question, shape, head_max_len)
    return prepare_arrays(ids, markers, qtype, shape, tok.pad_token_id), len(markers)
