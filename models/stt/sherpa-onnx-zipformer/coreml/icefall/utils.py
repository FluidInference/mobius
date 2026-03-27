"""Minimal icefall.utils stub for inference-only usage."""

from contextlib import contextmanager

import torch
from torch import Tensor


@contextmanager
def torch_autocast(enabled: bool = True, cache_enabled: bool = True):
    """No-op autocast context for CPU inference."""
    yield


def make_pad_mask(lengths: Tensor, max_len: int = 0) -> Tensor:
    """Create a boolean padding mask (True = padded position).

    Args:
      lengths: (batch,) actual lengths.
      max_len: if > 0, use this as the sequence dimension; otherwise infer
               from the max of *lengths*.
    Returns:
      (batch, max_len) bool tensor.
    """
    if max_len <= 0:
        max_len = int(lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device)
    return idx.unsqueeze(0) >= lengths.unsqueeze(1)


def add_sos(y, sos_id: int = 0):
    raise NotImplementedError("add_sos is training-only")


def num_tokens(symbol_table) -> int:
    raise NotImplementedError("use vocab_size from checkpoint directly")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError(f"Cannot interpret {v!r} as bool")


def time_warp(*args, **kwargs):
    raise NotImplementedError("time_warp is training-only")
