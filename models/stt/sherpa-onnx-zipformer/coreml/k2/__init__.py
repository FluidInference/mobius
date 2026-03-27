"""Minimal k2 stub — only the functions needed by scaling.py at import time.

After convert_scaled_to_non_scaled() replaces SwooshR/SwooshL with their
Onnx variants, none of these are actually called.
"""

import torch
from torch import Tensor


def swoosh_l_forward(x: Tensor) -> Tensor:
    return torch.log1p(torch.exp(x - 4.0)) - 0.08 * x - 0.035


def swoosh_r_forward(x: Tensor) -> Tensor:
    return torch.log1p(torch.exp(x - 1.0)) - 0.08 * x - 0.313261687


def swoosh_l(x: Tensor) -> Tensor:
    return swoosh_l_forward(x)


def swoosh_r(x: Tensor) -> Tensor:
    return swoosh_r_forward(x)


def swoosh_l_forward_and_deriv(x: Tensor):
    y = swoosh_l_forward(x)
    deriv = torch.sigmoid(x - 4.0) - 0.08
    return y, deriv


def swoosh_r_forward_and_deriv(x: Tensor):
    y = swoosh_r_forward(x)
    deriv = torch.sigmoid(x - 1.0) - 0.08
    return y, deriv
