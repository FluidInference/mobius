"""Fold weight_norm parametrizations so tracing produces plain conv weights.

Handles two cases:
  1. Legacy `torch.nn.utils.weight_norm` (weight_g, weight_v hooks): use
     `torch.nn.utils.remove_weight_norm`.
  2. New `torch.nn.utils.parametrizations.weight_norm` (Parametrized modules on
     torch >= 2.1): use `torch.nn.utils.parametrize.remove_parametrizations`.

Safe to call on a model where only some submodules are weight-normed.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.parametrize as P
from torch.nn.utils import remove_weight_norm


def fold_weight_norm(module: nn.Module) -> nn.Module:
    """Recursively fold weight_norm parametrizations in-place, return module."""
    for m in module.modules():
        # New-style parametrized weight_norm
        if P.is_parametrized(m):
            for tensor_name in list(m.parametrizations.keys()):
                P.remove_parametrizations(m, tensor_name, leave_parametrized=True)
            continue
        # Legacy weight_norm hooks
        if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            try:
                remove_weight_norm(m)
            except ValueError:
                pass
    return module
