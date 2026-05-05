"""Standalone nn.Module wrappers for ANE diagnostic conversion.

Each class here is a self-contained subgraph that mirrors a specific Magpie
production pattern. Designed to be traced + converted to CoreML alone, then
analyzed via coreml-cli --fallback to determine ANE compatibility.

Categories:
    Snake*           — codec activation hypothesis (sin/pow blocks ANE)
    KVCacheWrite*    — decoder_step KV cache write hypothesis (rank-4 scatter
                       blocks ANE under runtime-varying position)
    WeightNormConv1d — codec weight_norm hypothesis (parametrization survives
                       trace and adds CPU ops)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Snake activation variants
# ---------------------------------------------------------------------------


class SnakeLearned(nn.Module):
    """Production Snake: x + (1/alpha) * sin^2(alpha * x), alpha learnable per-channel.

    Mirrors NeMo's Snake exactly (same alpha shape, same formula).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)


class SnakePolyTaylor(nn.Module):
    """3rd-order Taylor approximation of Snake, ANE-friendly.

    Snake(x) = x + (1/alpha) * sin^2(alpha * x)
    Using sin(y)^2 = (1 - cos(2y)) / 2 and cos(y) ≈ 1 - y^2/2 + y^4/24:
        sin^2(y) ≈ y^2 - y^4/3 + ...
    So:
        Snake(x) ≈ x + alpha * x^2 - (alpha^3 / 3) * x^4

    Valid range: |alpha * x| <= ~1.5 (otherwise polynomial diverges from sin^2).
    Codec activations are typically bounded by upstream LayerNorm/Conv1d so
    this should be safe for the operating range.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        ax = a * x
        # Snake(x) ≈ x + a*x^2 - (a^3/3)*x^4 = x + a*x^2 * (1 - (ax)^2 / 3)
        return x + a * x * x * (1.0 - ax * ax / 3.0)


class SnakeTaylor5(nn.Module):
    """5th-order Taylor expansion of Snake.

    sin²(y) = y² − y⁴/3 + 2y⁶/45 − ...
    Snake(x) ≈ x + α·x² − (α³/3)·x⁴ + (2α⁵/45)·x⁶

    Significantly more accurate over |α·x| ∈ [0, π/2] than the 3rd-order form.
    Codec post-LayerNorm activations land in roughly this range so Taylor5
    should give acceptable parity vs the original sin² Snake.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        a3 = a * a * a
        a5 = a3 * a * a
        return x + a * x2 - (a3 / 3.0) * x4 + (2.0 * a5 / 45.0) * x6


class SnakeTaylor7(nn.Module):
    """7th-order Taylor expansion of Snake.

    Snake(x) ≈ x + α·x² − (α³/3)·x⁴ + (2α⁵/45)·x⁶ − (α⁷/315)·x⁸

    Extra term over Taylor5; better near the edges of the operating range.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        a3 = a * a * a
        a5 = a3 * a * a
        a7 = a5 * a * a
        return (
            x
            + a * x2
            - (a3 / 3.0) * x4
            + (2.0 * a5 / 45.0) * x6
            - (a7 / 315.0) * x8
        )


class SnakeTaylor5Clipped(nn.Module):
    """Taylor5 with α·x clipped to [-π/2, π/2] before the polynomial.

    Without clipping, trained α values up to ~5 multiplied by codec
    activations up to ~3 produce α·x ≳ 10, where the 5th-order polynomial
    diverges by orders of magnitude from sin² (the polynomial is unbounded
    while sin² is bounded by 1). This propagates as NaN/inf through the 96
    Snake instances of nanocodec.

    Clamping the *input* to the polynomial to [-π/2, π/2] keeps the
    approximation in its accurate regime. Outside this range we lose the
    oscillating tail of sin², which is rarely entered by post-LayerNorm
    codec activations and is dominated by the linear `x` term anyway.

    Implementation:
        y = clamp(α·x, -π/2, π/2)         (always-bounded surrogate of α·x)
        sin²(y) ≈ y² − y⁴/3 + 2y⁶/45      (Taylor5)
        Snake(x) = x + (1/α) · sin²(y)    (preserves the trained α scaling)
    """

    HALF_PI = 1.5707963267948966  # π/2

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        # Clamp α·x to [-π/2, π/2]
        ax = torch.clamp(a * x, -self.HALF_PI, self.HALF_PI)
        y2 = ax * ax
        y4 = y2 * y2
        y6 = y4 * y2
        sin2 = y2 - y4 / 3.0 + 2.0 * y6 / 45.0
        # (1/α)·sin²(α·x) — protect against α=0 with a small epsilon
        return x + sin2 / (a + 1e-9)


class SnakeNoSinPow(nn.Module):
    """Polynomial Snake using only mul/add/sub — no sin, no pow, no division.

    Stronger ANE-friendliness check: drops both sin and pow (the two specific
    rejected ops in the nanocodec fallback analysis).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        x2 = x * x
        x4 = x2 * x2
        return x + a * x2 - (a * a * a / 3.0) * x4


# ---------------------------------------------------------------------------
# KV cache write patterns (decoder_step)
# ---------------------------------------------------------------------------


class KVCacheWriteRank4OneHot(nn.Module):
    """Production decoder_step KV write: rank-4 cache, one-hot blend.

    Mirrors traceable_decoder_step.py:60-67 exactly. The hypothesis from
    SWIFT_PORT_FINDINGS.md is that this pattern static-compiles to ANE but
    runtime-recompiles when ``position`` varies, causing the observed
    ANECCompile() failures during real synth.
    """

    def __init__(self, max_seq: int = 512):
        super().__init__()
        self.max_seq = max_seq

    def forward(
        self, kv_k: torch.Tensor, kv_v: torch.Tensor,
        k_new: torch.Tensor, v_new: torch.Tensor, position: torch.Tensor,
    ):
        # kv_k, kv_v: (B, max_seq, H, D)
        # k_new, v_new: (B, 1, H, D)
        # position: (1,) float
        positions_range = torch.arange(self.max_seq, dtype=kv_k.dtype, device=kv_k.device)
        mask = (positions_range == position).to(kv_k.dtype).view(1, self.max_seq, 1, 1)
        new_k = kv_k * (1.0 - mask) + k_new * mask
        new_v = kv_v * (1.0 - mask) + v_new * mask
        return new_k, new_v


class KVCacheWriteRank3OneHot(nn.Module):
    """Rank-3 variant: collapse H*D into a single C dim. Same one-hot blend.

    Tests whether the rank-4 dim count itself is the runtime ANE blocker. If
    rank-3 ANE-compiles cleanly under varying position but rank-4 doesn't, the
    fix is the layout change.
    """

    def __init__(self, max_seq: int = 512):
        super().__init__()
        self.max_seq = max_seq

    def forward(
        self, kv_k: torch.Tensor, kv_v: torch.Tensor,
        k_new: torch.Tensor, v_new: torch.Tensor, position: torch.Tensor,
    ):
        # kv_k, kv_v: (max_seq, C)  where C = H*D
        # k_new, v_new: (1, C)
        positions_range = torch.arange(self.max_seq, dtype=kv_k.dtype, device=kv_k.device)
        mask = (positions_range == position).to(kv_k.dtype).view(self.max_seq, 1)
        new_k = kv_k * (1.0 - mask) + k_new * mask
        new_v = kv_v * (1.0 - mask) + v_new * mask
        return new_k, new_v


class KVCacheWriteHostConcat(nn.Module):
    """Host-concat variant: emit the new K/V row only; host appends.

    No cache write inside the model. The model just produces (k_new, v_new)
    given the current input projection. Host-side Swift code maintains the
    cache buffer and concatenates new rows into it.

    This eliminates the entire scatter/blend pattern from the ANE graph.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (1, 1, d_model) — production shape, B=1 hardcoded for trace stability
        k_new = self.k_proj(x).view(1, 1, self.n_heads, self.d_head)
        v_new = self.v_proj(x).view(1, 1, self.n_heads, self.d_head)
        return k_new, v_new


# ---------------------------------------------------------------------------
# Attention with cache, full subgraph (the actual unit of work)
# ---------------------------------------------------------------------------


class CausalSelfAttnRank4Cache(nn.Module):
    """Production decoder_step self-attention: full subgraph with rank-4 cache.

    Tests the *real* ANE compatibility of the attention block at decoder_step's
    T=1 single-step shape. This is the unit Apple's compiler fuses on; isolated
    here without the surrounding 12-layer stack.
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12, max_seq: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq = max_seq
        self.scale = self.d_head ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_k, kv_v, position):
        # x: (1, 1, d_model) — B=1 hardcoded for trace stability
        qkv = self.qkv_proj(x).view(1, 1, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0]  # (1, 1, H, D)
        k_new = qkv[:, :, 1]
        v_new = qkv[:, :, 2]

        # One-hot cache write
        positions_range = torch.arange(self.max_seq, dtype=x.dtype, device=x.device)
        mask = (positions_range == position).to(x.dtype).view(1, self.max_seq, 1, 1)
        new_k = kv_k * (1.0 - mask) + k_new * mask
        new_v = kv_v * (1.0 - mask) + v_new * mask

        # Causal mask
        causal_mask = (positions_range <= position).to(x.dtype).view(1, 1, 1, self.max_seq)

        # (1, H, 1, D) x (1, H, D, max_seq) -> (1, H, 1, max_seq)
        q4 = q.permute(0, 2, 1, 3)
        k4 = new_k.permute(0, 2, 3, 1)
        v4 = new_v.permute(0, 2, 1, 3)

        attn = torch.matmul(q4, k4) * self.scale
        attn = attn + (1.0 - causal_mask) * (-3.0e4)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v4)
        out = out.permute(0, 2, 1, 3).reshape(1, 1, -1)
        out = self.o_proj(out)

        new_position = position + 1.0
        return out, new_k, new_v, new_position


# ---------------------------------------------------------------------------
# weight_norm Conv1d variants
# ---------------------------------------------------------------------------


class WeightNormConv1dUnfolded(nn.Module):
    """Conv1d with weight_norm parametrization left in place.

    PyTorch's weight_norm is a parametrization: the weight is reconstructed
    from g (scale) and v (direction) at every forward call. Without
    `remove_weight_norm()`, the trace captures the parametrization arithmetic
    inside the graph.

    Hypothesis: the 194 weight_normed Conv1d layers in nanocodec retain their
    parametrization in the trace, adding mul/normalize ops per conv that ANE
    can't absorb.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Wrapped variants: embed activations / KV-write inside larger blocks so the
# CoreML scheduler actually dispatches to ANE. A 4-op standalone graph is
# auto-rejected by the scheduler regardless of op-level support.
# ---------------------------------------------------------------------------


class SnakeBlock(nn.Module):
    """Conv1d -> Snake variant -> Conv1d, mirrors a HiFi-GAN ResBlock unit.

    Wrapping the activation between two convs gives the graph enough work
    (~30+ ops) that the scheduler will attempt ANE dispatch. Whatever blocks
    ANE here is the activation itself (sin/pow), not graph-size heuristics.
    """

    def __init__(self, snake_cls: type, channels: int = 256, kernel: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.act = snake_cls(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class KVAttnBlock(nn.Module):
    """Linear-q -> KV-write variant -> matmul(q, K^T) -> linear-out.

    Same idea: surround the KV-write pattern with weighted ops so the graph
    is big enough for ANE dispatch, then ask whether the cache-write itself
    falls back to CPU.
    """

    def __init__(
        self,
        kv_cls: type,
        d_model: int = 768,
        n_heads: int = 12,
        max_seq: int = 512,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq = max_seq
        self.scale = self.d_head ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        # Instantiate the wrapped KV-write
        self.kv_write = kv_cls(max_seq)

    def forward(self, x, kv_k, kv_v, position):
        # x: (1, 1, d_model) -> qkv: (1, 1, 3, H, D)
        qkv = self.qkv_proj(x).view(1, 1, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0]
        k_new = qkv[:, :, 1]
        v_new = qkv[:, :, 2]

        new_k, new_v = self.kv_write(kv_k, kv_v, k_new, v_new, position)

        positions_range = torch.arange(self.max_seq, dtype=x.dtype, device=x.device)
        causal_mask = (positions_range <= position).to(x.dtype).view(1, 1, 1, self.max_seq)

        q4 = q.permute(0, 2, 1, 3)
        k4 = new_k.permute(0, 2, 3, 1)
        v4 = new_v.permute(0, 2, 1, 3)

        attn = torch.matmul(q4, k4) * self.scale
        attn = attn + (1.0 - causal_mask) * (-3.0e4)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v4)
        out = out.permute(0, 2, 1, 3).reshape(1, 1, -1)
        out = self.o_proj(out)
        return out, new_k, new_v


class WeightNormConv1dFolded(nn.Module):
    """Conv1d with weight_norm folded into a regular weight at construction.

    Calls remove_weight_norm() so the resulting module is a plain nn.Conv1d
    with a single weight tensor. Compare ANE behaviour against unfolded.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256, kernel_size: int = 7):
        super().__init__()
        wn = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        )
        # Materialize weight, drop parametrization
        nn.utils.remove_weight_norm(wn)
        self.conv = wn

    def forward(self, x):
        return self.conv(x)
