"""Unfused LayerNorm for the Flow DiT — Stage 0 of the Flow → ANE port.

Context
-------
coremltools emits a fused `layer_norm` MIL op whose internal scalar subgraph
is opaque to op-type based `FP16ComputePrecision` selection. The DiT graph
has 45 `nn.LayerNorm` instances × 22 blocks × 2 CFG × 10 Euler steps and at
fp16 the fused op produces NaN on real inputs (TRIALS_AND_ERRORS.md Phase 3).

This module provides a drop-in replacement that computes LayerNorm via
primitive ops (`reduce_mean`, sub, mul-square, rsqrt, mul) so coremltools
sees each primitive separately. In combination with removing the
`common::fuse_layernorm_or_instancenorm` pass from the conversion pipeline
(see `convert-flow.py`), the MIL graph retains primitives and does not
re-fuse them into `layer_norm`.

Usage
-----
Before tracing::

    from src.ane_layernorm import patch_dit_norms
    patch_dit_norms(flow.decoder.estimator)   # in-place swap

All 45 LN instances (AdaLayerNormZero, AdaLayerNormZero_Final, DiTBlock
`ff_norm`, ConvNeXtV2Block `norm`) are replaced; weight/bias are preserved
where `elementwise_affine=True`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ANEUnfusedLayerNorm(nn.Module):
    """Primitive-op LayerNorm (axis=-1) drop-in for `nn.LayerNorm`.

    Computes:
        diff = x - mean(x, dim=-1, keepdim=True)
        var  = mean(diff * diff, dim=-1, keepdim=True)
        y    = diff * rsqrt(var + eps)
        y    = y * weight + bias   # only if elementwise_affine=True

    Matches `nn.LayerNorm`'s output for 3D (B, S, C) input with
    `normalized_shape=(C,)`.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        if len(self.normalized_shape) != 1:
            raise NotImplementedError(
                "ANEUnfusedLayerNorm only supports 1D normalized_shape "
                f"(got {self.normalized_shape}). Flow DiT only uses 1D LN."
            )
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        dim = self.normalized_shape[0]
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel axis is always the last dim for DiT's (B, S, C) activations.
        mean = x.mean(dim=-1, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + self.eps)
        y = diff * inv
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return y


def patch_dit_norms(root: nn.Module) -> int:
    """Replace every `nn.LayerNorm` inside `root` with `ANEUnfusedLayerNorm`.

    Weights and biases are copied across when `elementwise_affine=True`.
    Returns the number of replacements applied.
    """
    replaced = 0
    for module in root.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.LayerNorm):
                if len(child.normalized_shape) != 1:
                    # Defensive: DiT only uses 1D LN; refuse to silently wrap
                    # anything else.
                    raise NotImplementedError(
                        "patch_dit_norms found a LayerNorm with "
                        f"normalized_shape={child.normalized_shape}; "
                        "only 1D is supported."
                    )
                replacement = ANEUnfusedLayerNorm(
                    normalized_shape=child.normalized_shape[0],
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                )
                if child.elementwise_affine:
                    replacement.weight.data.copy_(child.weight.data)
                    replacement.bias.data.copy_(child.bias.data)
                setattr(module, name, replacement)
                replaced += 1
    return replaced
