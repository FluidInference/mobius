"""ANE-compatible primitives for the Flow DiT BC1S port (Stage 2).

All modules operate on the ml-ane-transformers channels-first 4D layout
`(B, C, 1, S)` and replace their (B, S, C) counterparts from
`verify/CosyVoice/cosyvoice/flow/DiT/modules.py`.

Design principles (from Apple ml-ane-transformers):
  1. Channels on axis=1 (not axis=-1).
  2. Linear → `Conv2d(kernel_size=1)` so weights live as (out, in, 1, 1)
     — this matches the ANE layout for weight matmuls.
  3. LayerNorm reduces on axis=1 (the C axis), affine weight/bias broadcast
     over (1, 1, 1) → preserved by coremltools Pattern 5 LN fusion.
  4. GELU stays as a single MIL op (`gelu`); the DiT uses
     `approximate="tanh"` which also lowers to the same op.
  5. RoPE: pre-compute real-valued cos/sin tables shaped for broadcast.

Shapes through the DiT (B=2 CFG, C=1024, S=token_count):
  input_embed out : (B, 1024, 1, S)
  transformer in  : (B, 1024, 1, S)
  Q/K/V proj out  : (B, 1024, 1, S)
  per-head Q/K/V  : (B, heads=16, head_dim=64, S)
  attn out        : (B, 1024, 1, S)
  FFN out         : (B, 1024, 1, S)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANELinear(nn.Conv2d):
    """Linear as 1×1 Conv2d on BC1S input.

    Stores weight as `(out_features, in_features, 1, 1)` — ANE-friendly layout.
    Input:  `(B, in_features, 1, S)`
    Output: `(B, out_features, 1, S)`

    `state_dict_port` converts a source `nn.Linear` weight `(out, in)` to
    `(out, in, 1, 1)` via `.unsqueeze(-1).unsqueeze(-1)` and bias is preserved
    as-is.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, kernel_size=1, bias=bias)


class ANELayerNormBC1S(nn.Module):
    """LayerNorm on axis=1 (the channel axis) of a `(B, C, 1, S)` tensor.

    Reduces mean/var along dim=1 (channels) with `keepdim=True`, applies
    `elementwise_affine` weight/bias if requested (stored as
    `(1, C, 1, 1)`). This is the ANE-safe LN layout — coremltools'
    `fuse_layernorm_or_instancenorm` Pattern 5 targets exactly this
    shape and the resulting `layer_norm` MIL op is ANE-compatible
    (unlike the default BSC-axis LN which gets fused but not ANE-placed).

    Matches `nn.LayerNorm(C, elementwise_affine=..., eps=...)` numerically
    when called on the transposed tensor.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 1, S)
        mean = x.mean(dim=1, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        y = diff * inv_std
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return y


class ANEGELU(nn.Module):
    """Thin wrapper around `nn.GELU(approximate="tanh")`.

    Kept as a distinct module so `state_dict_port` recognises it and so
    future ANE-specific tweaks (e.g. custom tanh approx) can live in one
    place.
    """

    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x)


class ANEFeedForward(nn.Module):
    """FFN on BC1S:
        Conv2d(C, mult*C, 1x1) → GELU(tanh) → Dropout → Conv2d(mult*C, C, 1x1)

    Mirrors the host `FeedForward(dim, mult, approximate="tanh")` from
    modules.py:271-282. `state_dict_port` maps
        ff.ff.0.0.{weight,bias} → fc1.{weight,bias}
        ff.ff.2.{weight,bias}   → fc2.{weight,bias}
    """

    def __init__(
        self,
        dim: int,
        mult: int = 2,
        dropout: float = 0.0,
        approximate: str = "tanh",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        self.fc1 = ANELinear(dim, inner_dim)
        self.act = ANEGELU(approximate=approximate)
        self.drop = nn.Dropout(dropout)
        self.fc2 = ANELinear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class ANERotaryEmbedding(nn.Module):
    """Precomputed real-valued cos/sin tables for x_transformers-style RoPE.

    The host `RotaryEmbedding` (x_transformers.py:747) generates freqs as:

        inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))     # (dim/2,)
        freqs    = t[:, None] * inv_freq[None, :]               # (S, dim/2)
        # even-odd interleaved: [θ₀, θ₀, θ₁, θ₁, ...] via
        # stack((freqs,freqs),-1).reshape(..., dim)
        freqs_interleaved = stack((freqs, freqs), -1).reshape(-1, dim)  # (S, dim)

    and `apply_rotary_pos_emb` computes:
        t_rot = t * cos(freqs) + rotate_half(t) * sin(freqs)
    where `rotate_half` maps pairs `(x0, x1) → (-x1, x0)` along even-odd.

    For ANE we pre-compute:
        cos_table: (1, dim, 1, S)   — broadcast against BC1S features
        sin_table: (1, dim, 1, S)
    Both are registered buffers (compile-time constants in CoreML so they
    don't trigger runtime cos/sin ops on ANE).

    `apply(x_bhds, cos, sin)` takes `(B, heads, head_dim, S)` and returns
    the RoPE-rotated tensor in the same shape.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"ANERotaryEmbedding requires even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )  # (head_dim/2,)
        t = torch.arange(max_seq_len, dtype=torch.float32)  # (S,)
        freqs = torch.outer(t, inv_freq)  # (S, head_dim/2)

        # Match x_transformers layout: [θ₀, θ₀, θ₁, θ₁, ...] per position.
        freqs_interleaved = torch.stack((freqs, freqs), dim=-1).reshape(
            max_seq_len, head_dim
        )  # (S, head_dim)

        cos = freqs_interleaved.cos()  # (S, head_dim)
        sin = freqs_interleaved.sin()  # (S, head_dim)

        # Reshape for broadcast against (B, heads, head_dim, S).
        # The rotation is per-(head_dim, S) and identical across heads & batch.
        cos = cos.transpose(0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, head_dim, S)
        sin = sin.transpose(0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, head_dim, S)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Even-odd interleaved `(x0, x1, x2, x3, ...) → (-x1, x0, -x3, x2, ...)`.

        `x` shape `(B, heads=1, head_dim, S)` — rotation is along head_dim.

        Partial-RoPE callers pass H=1 (only head 0 gets rotated). Squeezing
        the H axis here keeps every intermediate ≤4D, which is a hard
        requirement for ANEF tensor ops: the natural implementation
        `reshape(B, H, D/2, 2, S)` produces a rank-5 tensor that the ANE
        compiler cannot lower (ANECCompile error 11 observed in Stage 3).
        """
        B, H, D, S = x.shape
        assert H == 1, (
            f"ANERotaryEmbedding.rotate_half expects H=1 (partial RoPE) "
            f"to stay 4D; got H={H}"
        )
        x3 = x.squeeze(1)                              # (B, D, S)
        x4 = x3.reshape(B, D // 2, 2, S)               # (B, D/2, 2, S)
        x0 = x4[:, :, 0, :]                            # (B, D/2, S)
        x1 = x4[:, :, 1, :]
        rotated_4 = torch.stack((-x1, x0), dim=2)      # (B, D/2, 2, S)
        rotated_3 = rotated_4.reshape(B, D, S)         # (B, D, S)
        return rotated_3.unsqueeze(1)                  # (B, 1, D, S)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE to `(B, heads, head_dim, S)`. Slices cos/sin to seq_len."""
        cos = self.cos_table[..., :seq_len]  # (1, 1, D, S)
        sin = self.sin_table[..., :seq_len]
        return x * cos + self.rotate_half(x) * sin
