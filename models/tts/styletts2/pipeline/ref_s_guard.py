"""Defensive wrapper for the StyleTTS2 voice reference vector `ref_s`.

History
-------
Past CoreML conversion attempts repeatedly silently mutated `ref_s`:
  * `ref_s[:, :128]` and `ref_s[:, 128:]` are *views*, not copies. In-place
    ops on either half mutate the original tensor.
  * The diffusion sampler's `features=ref_s` argument has historically been
    written in-place by certain sampler implementations.
  * Looping over multiple texts while reusing the same `ref_s` tensor can
    permanently taint it after the first synthesis.

This module makes mutation impossible-by-construction at the boundary,
and detectable everywhere else via snapshot assertions.

Public API
----------
    freeze_ref_s(ref_s)   -> torch.Tensor
        Returns a clone+detach+contiguous copy of `ref_s`. Cheap (~256 floats).

    RefSGuard(ref_s)
        Context manager / explicit snapshotter. Asserts the input tensor's
        bytes are unchanged on exit (or on `.assert_unchanged()`).
"""

from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from typing import Optional

import torch


def freeze_ref_s(ref_s: torch.Tensor) -> torch.Tensor:
    """Return a fully-independent, contiguous, detached copy of `ref_s`.

    Use this at the boundary of any function that consumes `ref_s` so the
    caller's tensor cannot be mutated (e.g. by an in-place sampler op).
    """
    if not isinstance(ref_s, torch.Tensor):
        raise TypeError(f"ref_s must be a torch.Tensor, got {type(ref_s).__name__}")
    return ref_s.detach().clone().contiguous()


def _fingerprint(t: torch.Tensor) -> str:
    """Stable byte fingerprint of a tensor's storage."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class RefSGuard(AbstractContextManager):
    """Snapshot a `ref_s` tensor and verify it hasn't changed.

    Usage:
        with RefSGuard(ref_s):
            wav = inference(text, ref_s)   # raises if ref_s mutated
    """

    def __init__(self, ref_s: torch.Tensor, name: str = "ref_s") -> None:
        if not isinstance(ref_s, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        self._tensor = ref_s
        self._name = name
        self._snapshot: Optional[torch.Tensor] = ref_s.detach().clone()
        self._fp_before: str = _fingerprint(ref_s)

    def assert_unchanged(self) -> None:
        if self._snapshot is None:
            raise RuntimeError("RefSGuard already finalized")
        fp_after = _fingerprint(self._tensor)
        if fp_after != self._fp_before:
            diff = (self._tensor - self._snapshot).abs()
            raise AssertionError(
                f"{self._name} was mutated during inference: "
                f"max_abs_delta={diff.max().item():.6e}, "
                f"mean_abs_delta={diff.mean().item():.6e}, "
                f"fingerprint {self._fp_before[:12]} -> {fp_after[:12]}"
            )

    def __exit__(self, exc_type, exc, tb) -> None:
        # Only assert when leaving cleanly; otherwise propagate the original error.
        if exc_type is None:
            self.assert_unchanged()
        self._snapshot = None
