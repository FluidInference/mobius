"""Shared PyTorch building blocks for Supertonic-3.

Naming conventions match ONNX initializer prefixes so weight loading is
trivial (no string remapping required).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Norm wrappers — ONNX names are `<scope>.norm.norm.*` because the upstream
# repo wraps the standard module inside a thin transposer.
# ---------------------------------------------------------------------------
class LayerNorm1d(nn.Module):
    """LayerNorm over the channel dim of a `[B, C, T]` tensor.

    Inner LN exposed as `self.norm` so ONNX names `<scope>.norm.norm.weight`
    load directly.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class BatchNorm1dWrap(nn.Module):
    """BatchNorm1d wrapper matching `<scope>.norm.{weight,bias,running_*}`."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# ---------------------------------------------------------------------------
# CausalConv1d
#
# All convs in the Supertonic-3 ONNX graphs use a separate `Pad` op with
# mode `edge` (replicate) and amounts `[0, 0, (k-1)*d, 0, 0, 0]` — i.e. pad
# only the temporal dim, only on the left side. The Conv itself then uses
# `pads=[0, 0]`.
#
# Some modules wrap the Conv1d as `<scope>.net` (vocoder), others have it
# directly at `<scope>` (text_encoder convnext.dwconv). Switch with
# `wrap_inner`.
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        *,
        groups: int = 1,
        dilation: int = 1,
        wrap_inner: bool = True,
        bias: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        total_pad = (kernel_size - 1) * dilation
        self.causal = causal
        if causal:
            self.pad_left = total_pad
            self.pad_right = 0
        else:
            # Symmetric (non-causal). ONNX text_encoder convnext uses (k-1)/2 each.
            self.pad_left = total_pad // 2
            self.pad_right = total_pad - self.pad_left
        conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            groups=groups,
            padding=0,
            dilation=dilation,
            bias=bias,
        )
        self.wrap_inner = wrap_inner
        if wrap_inner:
            self.net = conv
        else:
            # Inline conv: weight name is `<scope>.weight` instead of `<scope>.net.weight`
            self._conv = conv  # private to avoid the `.conv` namespace

    @property
    def conv(self) -> nn.Conv1d:
        return self.net if self.wrap_inner else self._conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, [self.pad_left, self.pad_right], mode="replicate")
        return self.conv(x)


# ---------------------------------------------------------------------------
# ConvNeXt-1D block.
# ---------------------------------------------------------------------------
class ConvNeXt1DBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        intermediate_dim: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        wrap_dwconv: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        self.dwconv = CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            dilation=dilation,
            wrap_inner=wrap_dwconv,
            causal=causal,
        )
        self.norm = LayerNorm1d(channels)
        self.pwconv1 = nn.Conv1d(channels, intermediate_dim, kernel_size=1)
        self.pwconv2 = nn.Conv1d(intermediate_dim, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """If `mask` is provided (shape [B, 1, T], values in {0, 1}), it is
        applied (a) after dwconv and (b) on the block output. The residual
        path assumes the caller has already masked `x` (matching the ONNX
        graph topology in text_encoder.convnext).
        """
        h = self.dwconv(x)
        if mask is not None:
            h = h * mask
        h = self.norm(h)
        h = self.pwconv1(h)
        h = F.gelu(h)
        h = self.pwconv2(h)
        out = x + self.gamma * h
        if mask is not None:
            out = out * mask
        return out


# ---------------------------------------------------------------------------
# Relative-position multi-head self-attention (VITS-style).
#
# Per-layer params (ONNX names):
#   conv_q.weight/bias  Conv1d(C, C, 1)
#   conv_k.weight/bias  Conv1d(C, C, 1)
#   conv_v.weight/bias  Conv1d(C, C, 1)
#   conv_o.weight/bias  Conv1d(C, C, 1)
#   emb_rel_k           [1, 2*W+1, dk]
#   emb_rel_v           [1, 2*W+1, dk]
#
# where W = window_size, dk = C // n_heads.
# ---------------------------------------------------------------------------
class RelPosSelfAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int, window_size: int):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.dk = channels // n_heads
        self.window_size = window_size
        self.conv_q = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_o = nn.Conv1d(channels, channels, kernel_size=1)
        self.emb_rel_k = nn.Parameter(torch.zeros(1, 2 * window_size + 1, self.dk))
        self.emb_rel_v = nn.Parameter(torch.zeros(1, 2 * window_size + 1, self.dk))

    def _get_relative_embeddings(self, emb: torch.Tensor, length: int) -> torch.Tensor:
        """Slice or pad emb (1, 2W+1, dk) so it has 2*length-1 positions along dim 1."""
        max_len = 2 * self.window_size + 1
        target_len = 2 * length - 1
        pad_total = max(target_len - max_len, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_total > 0:
            emb = F.pad(emb, [0, 0, pad_left, pad_right])
        # If max_len > target_len, slice from the middle.
        start = max((max_len - target_len) // 2, 0)
        end = start + target_len
        return emb[:, start:end, :]

    @staticmethod
    def _relative_to_absolute(x: torch.Tensor) -> torch.Tensor:
        """Convert relative-position logits [B, h, T, 2T-1] to absolute [B, h, T, T]."""
        B, H, T, _ = x.shape
        # Pad an extra column at the end.
        x = F.pad(x, [0, 1])  # [B, h, T, 2T]
        x = x.reshape(B, H, T * 2 * T)
        x = F.pad(x, [0, T - 1])  # [B, h, T*2*T + T-1]
        x = x.reshape(B, H, T + 1, 2 * T - 1)
        x = x[:, :, :T, T - 1 :]
        return x

    @staticmethod
    def _absolute_to_relative(x: torch.Tensor) -> torch.Tensor:
        """Inverse of `_relative_to_absolute`: [B, h, T, T] → [B, h, T, 2T-1]."""
        B, H, T, _ = x.shape
        x = F.pad(x, [0, T - 1])
        x = x.reshape(B, H, T * (2 * T - 1))
        x = F.pad(x, [T, 0])
        x = x.reshape(B, H, T, 2 * T)
        return x[:, :, :, 1:]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, C, T]; attn_mask: [B, 1, T] (1 for valid, 0 for pad)."""
        B, _, T = x.shape
        H, dk = self.n_heads, self.dk

        q = self.conv_q(x).reshape(B, H, dk, T).transpose(2, 3)  # [B, H, T, dk]
        k = self.conv_k(x).reshape(B, H, dk, T).transpose(2, 3)
        v = self.conv_v(x).reshape(B, H, dk, T).transpose(2, 3)

        scale = 1.0 / math.sqrt(dk)
        scores = torch.matmul(q * scale, k.transpose(-2, -1))  # [B, H, T, T]

        # Relative position contribution to scores.
        rel_k = self._get_relative_embeddings(self.emb_rel_k, T)  # [1, 2T-1, dk]
        rel_logits = torch.matmul(q * scale, rel_k.transpose(-2, -1).unsqueeze(0))  # [B,H,T,2T-1]
        scores = scores + self._relative_to_absolute(rel_logits)

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(2)  # [B, 1, 1, T]
            # ANE-friendly additive mask (float, no bool/select).
            scores = scores - (1.0 - mask) * 1e4

        p_attn = F.softmax(scores, dim=-1)
        out = torch.matmul(p_attn, v)  # [B, H, T, dk]

        # Relative position contribution to output.
        rel_v = self._get_relative_embeddings(self.emb_rel_v, T)
        rel_weights = self._absolute_to_relative(p_attn)  # [B, H, T, 2T-1]
        out = out + torch.matmul(rel_weights, rel_v.unsqueeze(0))

        out = out.transpose(2, 3).reshape(B, self.channels, T)
        return self.conv_o(out)


class TransformerFFN(nn.Module):
    """Conv1d(C, hdim) + activation + Conv1d(hdim, C). Both ksz=1.

    Per the text_encoder.onnx graph, the activation is ReLU (not GELU) and
    mask multiplications wrap each step: `conv_2(act(conv_1(x*m)) * m) * m`.
    The caller passes `x` already masked; this module re-masks inside and
    after to mirror the graph.
    """

    def __init__(self, channels: int, hdim: int, activation: str = "relu"):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, hdim, kernel_size=1)
        self.conv_2 = nn.Conv1d(hdim, channels, kernel_size=1)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(activation)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        h = self.conv_1(x)
        h = self.act(h)
        if mask is not None:
            h = h * mask
        h = self.conv_2(h)
        if mask is not None:
            h = h * mask
        return h


# ---------------------------------------------------------------------------
# ONNX → PyTorch weight loading helpers.
# ---------------------------------------------------------------------------
def onnx_initializers(onnx_path) -> Dict[str, np.ndarray]:
    """Return a dict of {initializer_name: numpy ndarray}."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    out: Dict[str, np.ndarray] = {}
    for t in model.graph.initializer:
        out[t.name] = numpy_helper.to_array(t)
    return out


def assign_param(module_param: nn.Parameter, arr: np.ndarray, *, name: str) -> None:
    t = torch.from_numpy(np.asarray(arr).copy()).to(module_param.dtype)
    if tuple(t.shape) != tuple(module_param.shape):
        raise ValueError(
            f"Shape mismatch for {name}: expected {tuple(module_param.shape)}, got {tuple(t.shape)}"
        )
    with torch.no_grad():
        module_param.copy_(t)


def load_convnext_block(
    block: ConvNeXt1DBlock,
    inits: Dict[str, np.ndarray],
    prefix: str,
    *,
    wrap_dwconv: bool = True,
) -> None:
    if wrap_dwconv:
        dw_w_key = f"{prefix}.dwconv.net.weight"
        dw_b_key = f"{prefix}.dwconv.net.bias"
    else:
        dw_w_key = f"{prefix}.dwconv.weight"
        dw_b_key = f"{prefix}.dwconv.bias"
    assign_param(block.dwconv.conv.weight, inits[dw_w_key], name=dw_w_key)
    assign_param(block.dwconv.conv.bias, inits[dw_b_key], name=dw_b_key)
    assign_param(block.norm.norm.weight, inits[f"{prefix}.norm.norm.weight"], name=f"{prefix}.norm.norm.weight")
    assign_param(block.norm.norm.bias, inits[f"{prefix}.norm.norm.bias"], name=f"{prefix}.norm.norm.bias")
    assign_param(block.pwconv1.weight, inits[f"{prefix}.pwconv1.weight"], name=f"{prefix}.pwconv1.weight")
    assign_param(block.pwconv1.bias, inits[f"{prefix}.pwconv1.bias"], name=f"{prefix}.pwconv1.bias")
    assign_param(block.pwconv2.weight, inits[f"{prefix}.pwconv2.weight"], name=f"{prefix}.pwconv2.weight")
    assign_param(block.pwconv2.bias, inits[f"{prefix}.pwconv2.bias"], name=f"{prefix}.pwconv2.bias")
    assign_param(block.gamma, inits[f"{prefix}.gamma"], name=f"{prefix}.gamma")


def load_relpos_attn(layer: RelPosSelfAttention, inits: Dict[str, np.ndarray], prefix: str) -> None:
    assign_param(layer.conv_q.weight, inits[f"{prefix}.conv_q.weight"], name=f"{prefix}.conv_q.weight")
    assign_param(layer.conv_q.bias, inits[f"{prefix}.conv_q.bias"], name=f"{prefix}.conv_q.bias")
    assign_param(layer.conv_k.weight, inits[f"{prefix}.conv_k.weight"], name=f"{prefix}.conv_k.weight")
    assign_param(layer.conv_k.bias, inits[f"{prefix}.conv_k.bias"], name=f"{prefix}.conv_k.bias")
    assign_param(layer.conv_v.weight, inits[f"{prefix}.conv_v.weight"], name=f"{prefix}.conv_v.weight")
    assign_param(layer.conv_v.bias, inits[f"{prefix}.conv_v.bias"], name=f"{prefix}.conv_v.bias")
    assign_param(layer.conv_o.weight, inits[f"{prefix}.conv_o.weight"], name=f"{prefix}.conv_o.weight")
    assign_param(layer.conv_o.bias, inits[f"{prefix}.conv_o.bias"], name=f"{prefix}.conv_o.bias")
    assign_param(layer.emb_rel_k, inits[f"{prefix}.emb_rel_k"], name=f"{prefix}.emb_rel_k")
    assign_param(layer.emb_rel_v, inits[f"{prefix}.emb_rel_v"], name=f"{prefix}.emb_rel_v")


def find_orphan_by_shape(
    inits: Dict[str, np.ndarray],
    shape: tuple,
    *,
    consumed: Optional[set] = None,
) -> str:
    consumed = consumed if consumed is not None else set()
    candidates = [
        name
        for name, arr in inits.items()
        if name.startswith("onnx::") and tuple(arr.shape) == tuple(shape) and name not in consumed
    ]
    if not candidates:
        raise KeyError(f"No orphan initializer with shape {shape}")
    if len(candidates) > 1:
        raise KeyError(f"Ambiguous orphan for shape {shape}: {candidates}")
    consumed.add(candidates[0])
    return candidates[0]
