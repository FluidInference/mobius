"""Pinned RLCD schema rendering and fixed-shape candidate preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass

import jsonschema
import numpy as np


@dataclass(frozen=True)
class Shape:
    length: int = 256
    candidates: int = 8
    max_value_tokens: int = 16


def validate_schema(schema: dict) -> None:
    """Enforce the pinned RLCD engine's flat, closed schema contract."""
    jsonschema.Draft202012Validator.check_schema(schema)
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("Only closed, flat object schemas are supported")
    fields = schema.get("properties", {})
    if not fields or set(schema.get("required", [])) != set(fields):
        raise ValueError("All fields must be required")
    if set(schema) - {"type", "properties", "required", "additionalProperties"}:
        raise ValueError("Unsupported object constraints")
    for spec in fields.values():
        if set(spec) - {"type", "enum", "description"}:
            raise ValueError("Unsupported field constraints")
        if spec.get("type") == "boolean" and "enum" not in spec:
            continue
        values = spec.get("enum", [])
        if spec.get("type") != "string" or not values or any(type(value) is not str for value in values):
            raise ValueError("Fields must be booleans or nonempty string enums")
        if len(set(values)) != len(values):
            raise ValueError("Duplicate candidates")


def prompt(tokenizer, context: str, schema: dict) -> str:
    """Build the pinned RLCD prompt exactly."""
    validate_schema(schema)
    messages = [
        {
            "role": "system",
            "content": "Extract the attributes from the text. Return only a JSON object matching this schema. "
            "Use the exact allowed values. No explanation or markdown.\n" + json.dumps(schema, ensure_ascii=False),
        },
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "{\n"


@dataclass(frozen=True)
class Candidate:
    field: str
    value: str | bool
    ids: list[int]
    positions: list[int]
    targets: list[int]


def prepare_candidates(tokenizer, context: str, schema: dict, shape: Shape) -> list[Candidate]:
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prefix = encode(prompt(tokenizer, context, schema))
    candidates = []
    for name, spec in schema["properties"].items():
        values = [True, False] if spec["type"] == "boolean" else spec["enum"]
        suffix = encode("  " + json.dumps(name, ensure_ascii=False) + ": ")
        for value in values:
            value_ids = encode(json.dumps(value, ensure_ascii=False) + "\n")
            ids = prefix + suffix + value_ids
            if len(ids) > shape.length:
                raise ValueError(f"candidate exceeds L{shape.length}: {name}={value!r}, {len(ids)} tokens")
            if len(value_ids) > shape.max_value_tokens:
                raise ValueError(f"value exceeds {shape.max_value_tokens} tokens: {name}={value!r}")
            first = len(prefix) + len(suffix) - 1
            candidates.append(Candidate(name, value, ids, list(range(first, first + len(value_ids))), value_ids))
    return candidates


def batch_arrays(tokenizer, candidates: list[Candidate], shape: Shape) -> dict[str, np.ndarray]:
    if len(candidates) > shape.candidates:
        raise ValueError(f"batch has {len(candidates)} candidates, max is {shape.candidates}")
    if not candidates:
        raise ValueError("empty candidate batch")
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("tokenizer has no pad token")
    ids = np.full((shape.candidates, shape.length), pad, dtype=np.int32)
    attention = np.zeros_like(ids)
    positions = np.zeros((shape.candidates, shape.max_value_tokens), dtype=np.int32)
    targets = np.zeros_like(positions)
    value_mask = np.zeros((shape.candidates, shape.max_value_tokens), dtype=np.float32)
    for row, candidate in enumerate(candidates):
        length = len(candidate.ids)
        count = len(candidate.targets)
        ids[row, :length] = candidate.ids
        attention[row, :length] = 1
        positions[row, :count] = candidate.positions
        targets[row, :count] = candidate.targets
        value_mask[row, :count] = 1.0
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "value_positions": positions,
        "value_targets": targets,
        "value_mask": value_mask,
    }


def select_values(candidates: list[Candidate], scores: list[float]) -> dict:
    if len(candidates) != len(scores):
        raise ValueError("one score required per candidate")
    selected = {}
    for candidate, score in zip(candidates, scores):
        previous = selected.get(candidate.field)
        if previous is None or score > previous[1]:
            selected[candidate.field] = (candidate.value, score)
    return {field: value for field, (value, _) in selected.items()}
