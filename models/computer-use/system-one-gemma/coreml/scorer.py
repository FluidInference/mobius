"""Run the converted trained scalar scorer over complete option sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from assets import ROOT
from native_reference import MAX_LENGTH, choose, encode_options, pad_batch


def score_coreml(model, tokenizer, state: str, question: str, options: list[str]) -> dict:
    """Pad each independent K16 scorer call, then softmax all real logits jointly."""
    sequences = encode_options(tokenizer, state, question, options)
    logits = []
    for offset in range(0, len(sequences), 16):
        chunk = sequences[offset : offset + 16]
        padded = chunk + [chunk[0]] * (16 - len(chunk))
        ids, masks = pad_batch(padded, tokenizer.pad_token_id, MAX_LENGTH)
        result = model.predict(
            {"input_ids": np.asarray(ids, dtype=np.int32), "attention_mask": np.asarray(masks, dtype=np.int32)}
        )
        logits.extend(np.asarray(result["logits"]).reshape(-1)[: len(chunk)].astype(float).tolist())
    return {**choose(options, logits), "logits": logits}


def load_runtime(package: Path = ROOT / "build" / "system_one_gemma_fp16_L256_K16.mlpackage"):
    """Load only the package and public base tokenizer after terms acceptance."""
    import coremltools as ct
    from transformers import AutoTokenizer

    from assets import fetch_base

    base = fetch_base()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL), tokenizer
