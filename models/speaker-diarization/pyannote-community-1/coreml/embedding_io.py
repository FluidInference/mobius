"""Shared helpers for packing audio and weights into Core ML-friendly tensors."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

EMBEDDING_SAMPLES = 160_000  # 10 s @ 16 kHz
SEGMENTATION_FRAMES = 589
EMBEDDING_PACKED_WIDTH = EMBEDDING_SAMPLES + SEGMENTATION_FRAMES


def pack_audio_weights_torch(
    waveforms: torch.Tensor,
    weights: torch.Tensor,
    *,
    weight_frames: int = SEGMENTATION_FRAMES,
) -> torch.Tensor:
    """Pack waveforms and weights into a single 4D tensor for Core ML tracing."""
    if waveforms.dim() != 3:
        raise ValueError(f"Expected waveforms with 3 dims, got {tuple(waveforms.shape)}")
    if weights.dim() != 2:
        raise ValueError(f"Expected weights with 2 dims, got {tuple(weights.shape)}")

    batch, channels, samples = waveforms.shape
    if channels != 1 or samples != EMBEDDING_SAMPLES:
        raise ValueError(
            f"Waveforms must have shape (batch, 1, {EMBEDDING_SAMPLES}), got {tuple(waveforms.shape)}"
        )
    if weight_frames <= 0:
        raise ValueError(f"weight_frames must be positive, got {weight_frames}")
    if weights.shape[0] != batch or weights.shape[1] != weight_frames:
        raise ValueError(
            f"Weights must have shape (batch, {weight_frames}), got {tuple(weights.shape)}"
        )

    packed_width = EMBEDDING_SAMPLES + weight_frames
    packed = torch.empty(
        batch,
        1,
        1,
        packed_width,
        dtype=waveforms.dtype,
        device=waveforms.device,
    )
    packed[..., :EMBEDDING_SAMPLES] = waveforms.view(batch, 1, 1, EMBEDDING_SAMPLES)
    packed[..., EMBEDDING_SAMPLES:] = weights.view(batch, 1, 1, weight_frames)
    return packed


def unpack_audio_weights_torch(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack the combined Core ML tensor into waveforms and weights."""
    if packed.dim() != 4:
        raise ValueError(f"Expected packed input with 4 dims, got {tuple(packed.shape)}")

    batch, channels, height, width = packed.shape
    if channels != 1 or height != 1 or width <= EMBEDDING_SAMPLES:
        raise ValueError(
            "Packed input must have shape "
            f"(batch, 1, 1, >= {EMBEDDING_PACKED_WIDTH}), got {tuple(packed.shape)}"
        )

    weight_frames = width - EMBEDDING_SAMPLES
    waveforms = packed[..., :EMBEDDING_SAMPLES].reshape(batch, 1, EMBEDDING_SAMPLES)
    weights = packed[..., EMBEDDING_SAMPLES:].reshape(batch, weight_frames)
    return waveforms, weights


def pack_audio_weights_numpy(
    audio: np.ndarray,
    weights: np.ndarray,
    *,
    weight_frames: int = SEGMENTATION_FRAMES,
) -> np.ndarray:
    """Pack numpy arrays into the combined Core ML input tensor."""
    if audio.ndim != 3:
        raise ValueError(f"Expected audio with 3 dims, got {audio.shape}")
    if weights.ndim != 2:
        raise ValueError(f"Expected weights with 2 dims, got {weights.shape}")

    batch, channels, samples = audio.shape
    if channels != 1 or samples != EMBEDDING_SAMPLES:
        raise ValueError(
            f"Audio must have shape (batch, 1, {EMBEDDING_SAMPLES}), got {audio.shape}"
        )
    if weight_frames <= 0:
        raise ValueError(f"weight_frames must be positive, got {weight_frames}")
    if weights.shape[0] != batch or weights.shape[1] != weight_frames:
        raise ValueError(
            f"Weights must have shape (batch, {weight_frames}), got {weights.shape}"
        )

    packed_width = EMBEDDING_SAMPLES + weight_frames
    packed = np.empty((batch, 1, 1, packed_width), dtype=audio.dtype)
    packed[:, 0, 0, :EMBEDDING_SAMPLES] = audio[:, 0, :]
    packed[:, 0, 0, EMBEDDING_SAMPLES:] = weights
    return packed


def unpack_audio_weights_numpy(packed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack numpy arrays from the combined Core ML tensor."""
    if packed.ndim != 4:
        raise ValueError(f"Expected packed input with 4 dims, got {packed.shape}")

    batch, channels, height, width = packed.shape
    if channels != 1 or height != 1 or width <= EMBEDDING_SAMPLES:
        raise ValueError(
            "Packed input must have shape "
            f"(batch, 1, 1, >= {EMBEDDING_PACKED_WIDTH}), got {packed.shape}"
        )

    weight_frames = width - EMBEDDING_SAMPLES
    audio = packed[:, :, :, :EMBEDDING_SAMPLES].reshape(batch, 1, EMBEDDING_SAMPLES)
    weights = packed[:, 0, 0, EMBEDDING_SAMPLES:].reshape(batch, weight_frames)
    return audio, weights


__all__ = [
    "EMBEDDING_SAMPLES",
    "SEGMENTATION_FRAMES",
    "EMBEDDING_PACKED_WIDTH",
    "pack_audio_weights_numpy",
    "unpack_audio_weights_numpy",
    "pack_audio_weights_torch",
    "unpack_audio_weights_torch",
]
