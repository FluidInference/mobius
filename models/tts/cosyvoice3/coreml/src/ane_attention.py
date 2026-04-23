"""Targeted fp32-cast attention for the Flow DiT — Stage 2a of the Flow → ANE port.

Context
-------
Stage 1 NaN probe (`src/nan_probe.py`) localized the fp16 blowup to
`F.scaled_dot_product_attention`: the intermediate `QK^T * scale` tensor
exceeds fp16 max (65504) in 9 of 22 DiT blocks (peak ~1.6M at block 17).
coremltools lowers SDPA to `matmul → mul(scale) → add(mask) → softmax → matmul`
and the `QK^T` intermediate materializes in ambient (fp16) precision, so
even though `softmax` is pinned to fp32 via `FP32_OPS`, it receives already
saturated `+inf` inputs → NaN.

This module provides an `ANEAttnProcessor` that decomposes SDPA into
primitive ops with an explicit fp32 cast around the risky region
(matmul → scale → mask → softmax → matmul), then casts back to the ambient
dtype. CoreML honors explicit casts, so the attention subgraph runs in fp32
while the rest of the DiT (residual stream, FFN, AdaLN) stays fp16.

Usage
-----
Before tracing::

    from src.ane_attention import patch_dit_attention
    patch_dit_attention(flow.decoder.estimator)  # in-place swap

Only the processor is swapped; Attention module weights (to_q/to_k/to_v/to_out)
are untouched. RoPE application and post-attention output projection are
preserved verbatim from the original `AttnProcessor` (DiT/modules.py:349-407).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sdpa_fp32(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Scaled dot-product attention with the matmul/softmax core in fp32.

    Mirrors `F.scaled_dot_product_attention(q, k, v, attn_mask=m,
    dropout_p=0.0, is_causal=False)` semantics for bool-mask case:
    True = keep, False = mask out (becomes -inf in logits).

    All inputs are cast up to fp32 for the QK^T / softmax / PV sequence;
    output is cast back to the ambient dtype of `query`.
    """
    orig_dtype = query.dtype
    head_dim = query.shape[-1]
    scale = 1.0 / (head_dim ** 0.5)

    # Pre-scale Q in the ambient (fp16) dtype. Safe: q peak ~200, scale
    # 0.125 → q_scaled peak ~25, well within fp16 range. Doing the scale
    # BEFORE the fp32 cast means we don't need to pin `mul` to fp32 in
    # `FP16ComputePrecision`, preserving ANE placement of the many other
    # mul ops (gate_msa, AdaLN, gate_mlp, FFN) in the graph.
    q_scaled = query * scale

    q = q_scaled.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)

    # (B, H, S_q, D) @ (B, H, D, S_k) -> (B, H, S_q, S_k)
    # This matmul overflows fp16 (peak ~1.6M / sqrt(d) = still ~1.6M since we
    # absorbed scale into Q already → output ≈ pre-scaled logits).
    # Actually: matmul(q_scaled, k^T) = scale * matmul(q, k^T) = post-scale
    # logits ≈ 1.6M worst case, needs fp32 matmul + fp32 select + fp32 softmax.
    logits = torch.matmul(q, k.transpose(-1, -2))

    if attn_mask is not None:
        # Bool mask: True=keep, False=mask out (set to -inf before softmax).
        neg_inf = torch.full_like(logits, float("-inf"))
        logits = torch.where(attn_mask, logits, neg_inf)

    probs = F.softmax(logits, dim=-1)
    out = torch.matmul(probs, v)  # (B, H, S_q, D)
    return out.to(orig_dtype)


class ANEAttnProcessor:
    """Drop-in replacement for `DiT.modules.AttnProcessor` with fp32 SDPA core.

    Preserves:
      - attn.to_q / to_k / to_v weight layout (ambient precision)
      - RoPE application via `apply_rotary_pos_emb`
      - Head reshape (B, S, inner_dim) -> (B, H, S, D)
      - Attention mask broadcast to (B, H, S, S)
      - Post-attention to_out[0] (linear proj) + to_out[1] (dropout)
      - Optional output mask-fill

    Differs only in the SDPA call: instead of
    `F.scaled_dot_product_attention(q, k, v, attn_mask=m, ...)` (single op
    that lowers to an fp16-intermediate chain in CoreML), we manually
    decompose with an explicit fp32 cast around matmul/softmax/matmul.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        attn,  # Attention instance (has .to_q/.to_k/.to_v/.to_out/.heads)
        x,     # (B, S, inner_dim) ambient dtype
        mask=None,
        rope=None,
    ):
        # Lazy import to avoid coupling this module to the verify/ tree at
        # import time (matches the pattern in convert-flow.py).
        from x_transformers.x_transformers import apply_rotary_pos_emb

        batch_size = x.shape[0]

        # `sample` projections.
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # Apply RoPE (identical to original AttnProcessor).
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale ** -1.0)
                if xpos_scale is not None
                else (1.0, 1.0)
            )
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # Head reshape: (B, S, inner) -> (B, H, S, D)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Mask broadcast (identical to original).
        if mask is not None:
            attn_mask = mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # b n -> b 1 1 n
                attn_mask = attn_mask.expand(
                    batch_size, attn.heads, query.shape[-2], key.shape[-2]
                )
        else:
            attn_mask = None

        # === The targeted fix: fp32 SDPA core ===
        x = _sdpa_fp32(query, key, value, attn_mask=attn_mask)
        # ========================================

        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Post-attention projection.
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)  # dropout (inference: no-op)

        # Optional output mask.
        if mask is not None:
            if mask.dim() == 2:
                mask_out = mask.unsqueeze(-1)
            else:
                mask_out = mask[:, 0, -1].unsqueeze(-1)
            x = x.masked_fill(~mask_out, 0.0)

        return x


def patch_dit_attention(dit: nn.Module) -> int:
    """Replace AttnProcessor with ANEAttnProcessor on every DiTBlock attention.

    Walks `dit.transformer_blocks[i].attn.processor` and swaps in the fp32
    core. Weights are untouched. Returns the number of blocks patched.

    Must be called AFTER state_dict load and BEFORE tracing.
    """
    n = 0
    if not hasattr(dit, "transformer_blocks"):
        raise AttributeError(
            "patch_dit_attention expected `dit.transformer_blocks` — "
            "is this a DiT module from cosyvoice3?"
        )
    for block in dit.transformer_blocks:
        if not hasattr(block, "attn") or not hasattr(block.attn, "processor"):
            continue
        block.attn.processor = ANEAttnProcessor()
        n += 1
    return n
