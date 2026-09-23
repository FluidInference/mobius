"""Use NanoJev's pinned native renderer for decision requests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from assets import upstream_module


def prepare_request(root: Path, tokenizer, request: dict, length: int, max_candidates: int):
    native = upstream_module(root, "predict_toy_decisions")
    examples = native.prepare_examples(request, tokenizer, length)
    if len(examples) != 1:
        raise ValueError("This Core ML bucket accepts one question per call")
    example = examples[0]
    paths = example["leaf_tokens"]
    if len(paths) > max_candidates:
        raise ValueError(f"{len(paths)} candidates exceed bucket capacity {max_candidates}")
    pad = tokenizer.pad_token_id
    ids = np.full((max_candidates, length), pad, dtype=np.int32)
    attention = np.zeros((max_candidates, length), dtype=np.int32)
    eos_map = np.zeros((max_candidates, 1, length), dtype=np.float32)
    candidate_mask = np.zeros((1, max_candidates), dtype=np.float32)
    for index, tokens in enumerate(paths):
        ids[index, : len(tokens)] = tokens
        attention[index, : len(tokens)] = 1
        eos_map[index, 0, len(tokens) - 1] = 1
        candidate_mask[0, index] = 1
    return {"input_ids": ids, "attention_mask": attention, "eos_map": eos_map}, candidate_mask, example
