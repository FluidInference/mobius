#!/usr/bin/env python3
"""Greedy RNN-T decoding for Zipformer2 transducer models.

Works with both PyTorch modules and CoreML MLModel objects so the same
algorithm can validate either backend.

Usage:
    from rnnt_decode import greedy_decode_pytorch, greedy_decode_coreml

    tokens = greedy_decode_pytorch(encoder_out, enc_export, dec_export, join_export, blank_id, context_size)
    tokens = greedy_decode_coreml(encoder_out_np, enc_ml, dec_ml, join_ml, blank_id, context_size)
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def greedy_decode_pytorch(
    encoder_out: torch.Tensor,
    dec_model: torch.nn.Module,
    join_model: torch.nn.Module,
    blank_id: int,
    context_size: int,
    joiner_dim: int,
) -> List[int]:
    """Greedy RNNT decode using PyTorch modules.

    Args:
        encoder_out: (1, T, joiner_dim) encoder output after projection.
        dec_model: DecoderForExport module (takes (1, context_size) int64 -> (1, joiner_dim)).
        join_model: JoinerForExport module (takes (1, joiner_dim), (1, joiner_dim) -> (1, vocab_size)).
        blank_id: Blank token ID.
        context_size: Decoder context window size.
        joiner_dim: Joiner hidden dimension.

    Returns:
        List of emitted token IDs (excluding blanks).
    """
    T = encoder_out.shape[1]
    hyp = [blank_id] * context_size
    tokens: List[int] = []

    with torch.no_grad():
        for t in range(T):
            enc_frame = encoder_out[:, t, :]  # (1, joiner_dim)

            y = torch.tensor([hyp[-context_size:]], dtype=torch.int64)
            dec_out = dec_model(y)  # (1, joiner_dim)

            logit = join_model(enc_frame, dec_out)  # (1, vocab_size)
            token_id = logit.argmax(dim=-1).item()

            if token_id != blank_id:
                tokens.append(token_id)
                hyp.append(token_id)

    return tokens


def greedy_decode_coreml(
    encoder_out: np.ndarray,
    dec_ml,
    join_ml,
    blank_id: int,
    context_size: int,
) -> List[int]:
    """Greedy RNNT decode using CoreML models.

    Args:
        encoder_out: (1, T, joiner_dim) numpy array from CoreML encoder.
        dec_ml: CoreML MLModel for decoder.
        join_ml: CoreML MLModel for joiner.
        blank_id: Blank token ID.
        context_size: Decoder context window size.

    Returns:
        List of emitted token IDs (excluding blanks).
    """
    T = encoder_out.shape[1]
    hyp = [blank_id] * context_size
    tokens: List[int] = []

    for t in range(T):
        enc_frame = encoder_out[:, t, :]  # (1, joiner_dim)

        y = np.array([hyp[-context_size:]], dtype=np.int32)
        dec_pred = dec_ml.predict({"y": y})
        dec_out = dec_pred["decoder_out"]  # (1, joiner_dim)

        join_pred = join_ml.predict({
            "encoder_out": enc_frame,
            "decoder_out": dec_out,
        })
        logit = join_pred["logit"]  # (1, vocab_size)
        token_id = int(np.argmax(logit, axis=-1).item())

        if token_id != blank_id:
            tokens.append(token_id)
            hyp.append(token_id)

    return tokens


def tokens_to_text(token_ids: List[int], vocab: List[str]) -> str:
    """Convert token IDs to text using a vocabulary list.

    Handles SentencePiece BPE tokens with word boundary marker (U+2581).
    """
    pieces = [vocab[tid] for tid in token_ids if 0 <= tid < len(vocab)]
    text = "".join(pieces).replace("\u2581", " ").strip()
    return text
