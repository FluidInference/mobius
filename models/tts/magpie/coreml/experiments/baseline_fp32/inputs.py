"""Per-stage representative-input builders for the fp32 baseline harness.

Each stage's `build_inputs(...)` returns a dict matching the input
contract of both the fp32 mlpackage and the production fp16 mlmodelc.
The numeric values are deterministic (seeded numpy) so re-runs produce
identical comparisons; the goal is parity, not statistical coverage.

Shapes mirror the trace-time shapes used in the corresponding
`convert_<stage>.py` exporters.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

_SEED = 42


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def build_inputs(stage: str) -> Dict[str, np.ndarray]:
    if stage == "text_encoder":
        return _text_encoder()
    if stage == "decoder_prefill":
        return _decoder_prefill()
    if stage == "decoder_step":
        return _decoder_step()
    if stage == "local_transformer":
        return _local_transformer()
    if stage == "nanocodec_decoder_v3":
        return _nanocodec_v3()
    raise ValueError(f"unknown stage: {stage}")


def _text_encoder(T: int = 256) -> Dict[str, np.ndarray]:
    rng = _rng()
    tokens = rng.integers(0, 100, size=(1, T)).astype(np.int32)
    mask = np.ones((1, T), dtype=np.float32)
    mask[0, T // 2:] = 0.0
    return {"text_tokens": tokens, "text_mask": mask}


def _decoder_prefill(T_ctx: int = 110, max_text_len: int = 256,
                     d_model: int = 768) -> Dict[str, np.ndarray]:
    """Speaker-context prefill — see convert_decoder_prefill.py inputs."""
    rng = _rng()
    return {
        "audio_embed": rng.standard_normal((1, T_ctx, d_model)).astype(np.float32) * 0.05,
        "encoder_output": rng.standard_normal((1, max_text_len, d_model)).astype(np.float32) * 0.05,
        # CoreML's bool input is materialised as fp32 0/1 at the I/O boundary.
        "encoder_mask": np.ones((1, max_text_len), dtype=np.float32),
    }


def _decoder_step(max_text_len: int = 256, max_seq_len: int = 512,
                  d_model: int = 768, n_layers: int = 12,
                  sa_n_heads: int = 12, position: int = 110) -> Dict[str, np.ndarray]:
    """Per-step decoder feed.

    Mirrors the trace-time shapes in `convert_decoder_step.py`:
    audio_embed (1, 1, d_model), encoder_output (1, T_enc, d_model),
    encoder_mask (1, T_enc), and 12 × {cache_k, cache_v, position}.
    `position=110` puts the AR loop just past the speaker-context
    prefill window — typical mid-loop scheduler state.
    """
    rng = _rng()
    d_head = d_model // sa_n_heads
    feed: Dict[str, np.ndarray] = {
        "audio_embed": rng.standard_normal((1, 1, d_model)).astype(np.float32) * 0.05,
        "encoder_output": rng.standard_normal(
            (1, max_text_len, d_model)).astype(np.float32) * 0.05,
        "encoder_mask": np.ones((1, max_text_len), dtype=np.float32),
    }
    cache_shape = (1, max_seq_len, sa_n_heads, d_head)
    for i in range(n_layers):
        feed[f"cache_k{i}"] = rng.standard_normal(cache_shape).astype(np.float32) * 0.05
        feed[f"cache_v{i}"] = rng.standard_normal(cache_shape).astype(np.float32) * 0.05
        feed[f"position{i}"] = np.array([position], dtype=np.float32)
    return feed


def _local_transformer(d_model: int = 768) -> Dict[str, np.ndarray]:
    """Per-step LT feed. See `convert_local_transformer.py` line 480 area:
    `decoder_hidden (1, d_model)`, `uniforms (8,)`, `forbid_eos (1,)`,
    `temperature (1,)`. All Float32 at the I/O boundary."""
    rng = _rng()
    return {
        "decoder_hidden": rng.standard_normal((1, d_model)).astype(np.float32) * 0.5,
        # Uniforms are random samples in [0, 1) — one per codebook for top-k CDF sampling.
        "uniforms": rng.random(8).astype(np.float32),
        "forbid_eos": np.array([1.0], dtype=np.float32),
        "temperature": np.array([0.6], dtype=np.float32),
    }


def _nanocodec_v3(T: int = 24, n_codebooks: int = 8,
                  codebook_size: int = 2024) -> Dict[str, np.ndarray]:
    """Per-call NanoCodec input. T_in=24 is the production chunk shape."""
    rng = _rng()
    tokens = rng.integers(0, codebook_size, size=(1, n_codebooks, T)).astype(np.int32)
    return {"tokens": tokens}
