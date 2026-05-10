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


def _decoder_step() -> Dict[str, np.ndarray]:
    raise NotImplementedError(
        "decoder_step has 36 KV-cache inputs + position vars; build "
        "from the converter's example feed once we tackle it."
    )


def _local_transformer() -> Dict[str, np.ndarray]:
    raise NotImplementedError("local_transformer feed not built yet")


def _nanocodec_v3(T: int = 24, n_codebooks: int = 8) -> Dict[str, np.ndarray]:
    raise NotImplementedError("nanocodec feed not built yet")
