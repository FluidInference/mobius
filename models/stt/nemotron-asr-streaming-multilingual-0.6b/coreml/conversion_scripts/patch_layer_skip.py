"""Pre-trace patch: replace specific encoder.layers[i] with no-op pass-through.

The Nemotron encoder has 24 ConformerLayers. Each contributes ~25M params
of compute per chunk. Skipping middle layers at inference time is a
speculative speed lever — relies on residual+LayerNorm absorbing the
skipped contribution. The model was trained with stochastic_depth_drop_prob=0,
so this is purely an inference-time bet (no training-time skip exposure).

Usage:
    from patch_layer_skip import patch_skip_layers
    patch_skip_layers(encoder, skip_indices=[8, 12, 16])

The SkipLayer matches ConformerLayer.forward signature exactly:
    (x, att_mask, pos_emb, pad_mask, cache_last_channel, cache_last_time)
    -> (x, cache_last_channel, cache_last_time)

Returns input UNCHANGED + caches UNCHANGED (so the cache slots stay at
their previous chunk's values for this layer — slight drift expected).
"""
from __future__ import annotations

import sys
from typing import List

import torch
import torch.nn as nn


class SkipLayer(nn.Module):
    """No-op ConformerLayer drop-in: returns input unchanged, caches unchanged."""

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None,
                cache_last_channel=None, cache_last_time=None):
        # Pass through. Caches are returned as-is so the encoder's cache
        # tensor for this layer's slot stays at whatever it was.
        return x, cache_last_channel, cache_last_time


def patch_skip_layers(encoder: nn.Module, skip_indices: List[int]) -> int:
    """Replace encoder.layers[i] with SkipLayer for each i in skip_indices.

    Returns count of layers patched.
    """
    if not hasattr(encoder, "layers"):
        raise AttributeError("encoder has no .layers attribute")
    n_total = len(encoder.layers)
    count = 0
    for i in sorted(set(skip_indices)):
        if i < 0 or i >= n_total:
            print(f"  [patch_layer_skip] skipping invalid layer index {i}",
                  file=sys.stderr, flush=True)
            continue
        encoder.layers[i] = SkipLayer()
        count += 1
    print(
        f"  [patch_layer_skip] replaced {count} layers with SkipLayer "
        f"(indices: {sorted(set(skip_indices))}, total layers: {n_total})",
        file=sys.stderr, flush=True,
    )
    return count
