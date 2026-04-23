"""ANE-compatible DiT for CosyVoice3 Flow.

Replaces `verify/CosyVoice/cosyvoice/flow/DiT/dit.py::DiT` and the
sub-modules in `modules.py` with ml-ane-transformers channels-first
(B, C, 1, S) 4D analogues.

Top-level interface matches the host DiT:
    forward(x, mask, mu, t, spks, cond, streaming=False) -> (B, mel_dim, S)

Internally:
    1. Host-side pieces that are cheap / not perf-critical stay in BSC:
       `time_embed` (TimestepEmbedding), `input_embed` (InputEmbedding
       including CausalConvPositionEmbedding's Conv1d position embed),
       and `rotary_embed` (pre-computed cos/sin buffers).
    2. After `input_embed`, permute (B, S, C) → (B, C, 1, S) once and run
       all 22 `ANEDiTBlock`s + `ANEAdaLayerNormZeroFinal` + `ANELinear`
       proj_out in BC1S.
    3. Final permute back to (B, mel_dim, S) matches the host DiT output
       signature (proj_out + transpose in dit.py:175).

Streaming (`streaming=True`) and `long_skip_connection` paths are NOT
supported here — this port targets the `streaming=False` inference path
used by the ODE Euler loop in `FlowCoreML`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ane_layers import (
    ANEFeedForward,
    ANEGELU,
    ANELayerNormBC1S,
    ANELinear,
    ANERotaryEmbedding,
)


class ANEAdaLayerNormZero(nn.Module):
    """AdaLN-Zero on BC1S.

    Host (`modules.py:230-244`):
        emb: (B, C)
        emb = linear(silu(emb))                       # (B, 6C)
        shift_msa, scale_msa, gate_msa,
        shift_mlp, scale_mlp, gate_mlp = chunk(6)      # each (B, C)
        x = norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]

    ANE version:
        emb: (B, C) — fed through `silu → ANELinear(C→6C)` as a 4D vector
             (B, C, 1, 1) → (B, 6C, 1, 1), then split along axis=1.
        Modulation params are all (B, C, 1, 1) → broadcast across S.
        `norm` is the affine-free `ANELayerNormBC1S`.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ANELinear(dim, dim * 6)
        self.norm = ANELayerNormBC1S(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        # x: (B, C, 1, S); emb: (B, C)
        e4 = emb.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        e4 = self.linear(self.silu(e4))       # (B, 6C, 1, 1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            e4, 6, dim=1
        )  # each (B, C, 1, 1)
        x_norm = self.norm(x) * (1.0 + scale_msa) + shift_msa
        return x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp


class ANEAdaLayerNormZeroFinal(nn.Module):
    """Final AdaLN-Zero (2-chunk) on BC1S.

    Host (`modules.py:251-265`):
        emb = linear(silu(emb))                       # (B, 2C)
        scale, shift = chunk(2)
        x = norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ANELinear(dim, dim * 2)
        self.norm = ANELayerNormBC1S(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        e4 = emb.unsqueeze(-1).unsqueeze(-1)       # (B, C, 1, 1)
        e4 = self.linear(self.silu(e4))            # (B, 2C, 1, 1)
        scale, shift = torch.chunk(e4, 2, dim=1)   # each (B, C, 1, 1)
        return self.norm(x) * (1.0 + scale) + shift


class ANEAttention(nn.Module):
    """Per-head attention on BC1S with x_transformers-compatible RoPE.

    Host (`modules.py:349-407`, `AttnProcessor.__call__` — non-joint path):
        q = to_q(x); k = to_k(x); v = to_v(x)       # (B, S, C)
        apply_rotary_pos_emb(q, freqs); apply_rotary_pos_emb(k, freqs)
        q,k,v reshape (B, H, S, D)
        attn_mask (B, 1, S, S) bool
        out = F.scaled_dot_product_attention(q, k, v, attn_mask)
        out reshape (B, S, C)
        out = to_out[0](out)

    ANE version:
        Input x:  (B, C, 1, S)  — BC1S
        to_q/k/v: ANELinear (1×1 Conv2d)
        Per-head reshape: (B, C, 1, S) → (B, heads, head_dim, S)
        RoPE applied on head_dim axis via `ANERotaryEmbedding` (even-odd
        interleaved rotate_half; cos/sin tables are compile-time buffers).
        Softmax SDPA:
            logits[b,h,i,j] = sum_c Q[b,h,c,i] * K[b,h,c,j]        (einsum bhci,bhcj->bhij)
            logits = logits * scale                                  # scale = 1/sqrt(D)
            logits = where(mask, logits, -inf)                       # mask (B, 1, S, S)
            probs  = softmax(logits, dim=-1)
            out[b,h,c,i] = sum_j probs[b,h,i,j] * V[b,h,c,j]         (einsum bhij,bhcj->bhci)
        Per-head merge: (B, heads, head_dim, S) → (B, C, 1, S)
        to_out[0] via ANELinear.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        assert self.inner_dim == dim, (
            f"ANEAttention expects heads*dim_head == dim; got {heads}*{dim_head} != {dim}"
        )

        self.to_q = ANELinear(dim, self.inner_dim)
        self.to_k = ANELinear(dim, self.inner_dim)
        self.to_v = ANELinear(dim, self.inner_dim)
        self.to_out = nn.ModuleList(
            [ANELinear(self.inner_dim, dim), nn.Dropout(dropout)]
        )

        self.rotary = ANERotaryEmbedding(dim_head, max_seq_len=max_seq_len)
        self.scale = 1.0 / (dim_head ** 0.5)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, C, 1, S) → (B, heads, head_dim, S)
        B, C, _, S = t.shape
        # squeeze the singleton, then view into (B, heads, head_dim, S).
        return t.view(B, self.heads, self.dim_head, S)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, heads, head_dim, S) → (B, C, 1, S)
        B, H, D, S = t.shape
        return t.reshape(B, H * D, 1, S)

    def forward(
        self,
        x: torch.Tensor,                       # (B, C, 1, S)
        mask: Optional[torch.Tensor] = None,   # (B, 1, S, S) bool, True = keep
    ) -> torch.Tensor:
        B, _, _, S = x.shape

        q = self._split_heads(self.to_q(x))  # (B, H, D, S)
        k = self._split_heads(self.to_k(x))
        v = self._split_heads(self.to_v(x))

        # Partial rotary embedding (Wang et al. GPT-J style) — the host
        # `AttnProcessor` applies `apply_rotary_pos_emb` to (B, S, C)
        # BEFORE per-head view. The vendored x_transformers routine takes
        # `rot_dim = freqs.shape[-1] = dim_head`, so only the first
        # `dim_head` features of the C=H*D axis get rotated. Because
        # `view(B, S, H, D)` assigns features [0:D] to head 0,
        # [D:2D] to head 1, etc., this means ONLY head 0 is rotated.
        # We replicate that here: rotate head 0, pass the rest through.
        q_h0 = self.rotary(q[:, :1, :, :], seq_len=S)
        k_h0 = self.rotary(k[:, :1, :, :], seq_len=S)
        q = torch.cat([q_h0, q[:, 1:, :, :]], dim=1)
        k = torch.cat([k_h0, k[:, 1:, :, :]], dim=1)

        # Match host SDPA order (math backend): logits = QK^T; scale after.
        # Ordering `matmul(Q,K) * scale` — not `matmul(Q*scale, K)` —
        # keeps fp32 accumulation inside the matmul identical to host's
        # F.scaled_dot_product_attention on the CPU math backend, so the
        # late-block isolated parity matches.
        logits = torch.einsum("bhci,bhcj->bhij", q, k) * self.scale  # (B, H, S, S)

        if mask is not None:
            # mask: (B, 1, S, S) bool — broadcast across head axis.
            neg_inf = torch.full_like(logits, float("-inf"))
            logits = torch.where(mask, logits, neg_inf)

        # ANE softmax on a 4D (B, H, S, S) tensor with S=250 can exceed ANEF
        # per-axis limits and causes `ANECCompile() FAILED (11)`. Fold the
        # head axis into the batch so softmax sees a 3D (B*H, S, S) tensor.
        # Numerically identical — softmax reduces only along the last axis.
        H = self.heads
        logits_3d = logits.reshape(B * H, S, S)
        probs_3d = F.softmax(logits_3d, dim=-1)
        probs = probs_3d.reshape(B, H, S, S)

        # out = einsum('bhij,bhcj->bhci', probs, v)
        out = torch.einsum("bhij,bhcj->bhci", probs, v)  # (B, H, D, S)

        out = self._merge_heads(out)           # (B, C, 1, S)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class ANEDiTBlock(nn.Module):
    """One DiT block on BC1S.

    Host (`modules.py:500-531`):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = attn_norm(x, t)
        x = x + gate_msa[:, None] * attn(norm, mask, rope)
        norm = ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * ff(norm)

    ANE version uses `ANEAdaLayerNormZero` + `ANEAttention` + an
    affine-free `ANELayerNormBC1S` for `ff_norm` + `ANEFeedForward`.
    Modulation params are (B, C, 1, 1) → broadcast over S implicitly.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        max_seq_len: int,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = ANEAdaLayerNormZero(dim)
        self.attn = ANEAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.ff_norm = ANELayerNormBC1S(dim, eps=1e-6, elementwise_affine=False)
        self.ff = ANEFeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(
        self,
        x: torch.Tensor,                       # (B, C, 1, S)
        t: torch.Tensor,                       # (B, C)
        mask: Optional[torch.Tensor] = None,   # (B, 1, S, S) bool
    ) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
        attn_out = self.attn(norm, mask=mask)
        x = x + gate_msa * attn_out

        norm = self.ff_norm(x) * (1.0 + scale_mlp) + shift_mlp
        ff_out = self.ff(norm)
        x = x + gate_mlp * ff_out
        return x


class ANEDiT(nn.Module):
    """Top-level DiT on BC1S.

    Wraps host `TimestepEmbedding` and `InputEmbedding` unchanged (they
    produce (B, C) and (B, S, C) respectively and are not perf bottlenecks)
    and BC1S-ports the 22-block transformer trunk + final AdaLN + proj_out.

    Input signature mirrors `DiT.forward` so `FlowCoreML` can swap
    estimator with minimal friction:
        x    : (B, mel_dim, S)
        mask : (B, S) bool
        mu   : (B, mel_dim, S)
        t    : scalar or (B,)
        spks : (B, spk_dim)
        cond : (B, mel_dim, S)
    Output:
        (B, mel_dim, S)
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        max_seq_len: int,
        dropout: float = 0.0,
        ff_mult: int = 2,
        mel_dim: int = 80,
        mu_dim: Optional[int] = None,
        spk_dim: Optional[int] = None,
        time_embed: Optional[nn.Module] = None,
        input_embed: Optional[nn.Module] = None,
    ):
        super().__init__()
        if mu_dim is None:
            mu_dim = mel_dim

        self.dim = dim
        self.depth = depth
        self.mel_dim = mel_dim
        self.max_seq_len = max_seq_len

        # Host-side modules kept as-is. Caller supplies pre-constructed
        # instances so we can reuse the trained weights without rewriting.
        assert time_embed is not None, "ANEDiT requires a host TimestepEmbedding instance"
        assert input_embed is not None, "ANEDiT requires a host InputEmbedding instance"
        self.time_embed = time_embed
        self.input_embed = input_embed

        self.transformer_blocks = nn.ModuleList(
            [
                ANEDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    max_seq_len=max_seq_len,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = ANEAdaLayerNormZeroFinal(dim)
        self.proj_out = ANELinear(dim, mel_dim)

    def forward(
        self,
        x: torch.Tensor,                     # (B, mel_dim, S)
        mask: Optional[torch.Tensor],        # None | (B, S) | (B, 1, S) | (B, 1, S, S)
        mu: torch.Tensor,                    # (B, mel_dim, S)
        t: torch.Tensor,                     # () or (B,)
        spks: torch.Tensor,                  # (B, spk_dim)
        cond: torch.Tensor,                  # (B, mel_dim, S)
    ) -> torch.Tensor:
        # Host DiT BSC path up to input_embed (dit.py:146-156).
        x_bsc = x.transpose(1, 2)          # (B, S, C=mel)
        mu_bsc = mu.transpose(1, 2)        # (B, S, mu)
        cond_bsc = cond.transpose(1, 2)    # (B, S, mel)
        batch = x_bsc.shape[0]
        if t.ndim == 0:
            t = t.repeat(batch)

        t_emb = self.time_embed(t)                                  # (B, C)
        x_bsc = self.input_embed(x_bsc, cond_bsc, mu_bsc, spks)     # (B, S, C)

        # BSC → BC1S once.
        x_bc1s = x_bsc.transpose(1, 2).unsqueeze(2).contiguous()    # (B, C, 1, S)

        # Build the 4D attention mask once at this boundary (if needed).
        # For densely-packed inference with no padding, pass mask=None.
        # Accepted input shapes:
        #   None                       — skip masking entirely
        #   (B, S)          bool       — padding mask per sequence
        #   (B, 1, S)       bool/float — host DiT form; squeezed to (B, S)
        #   (B, 1, S, S)    bool       — pre-built 4D mask (passed through)
        if mask is None:
            attn_mask = None
        else:
            m = mask.bool()
            if m.dim() == 4:
                attn_mask = m
            else:
                if m.dim() == 3:
                    m = m.squeeze(1)  # (B, S)
                # (B, S) → (B, 1, S, S)
                attn_mask = (m.unsqueeze(1) & m.unsqueeze(2)).unsqueeze(1)

        for block in self.transformer_blocks:
            x_bc1s = block(x_bc1s, t_emb, mask=attn_mask)

        x_bc1s = self.norm_out(x_bc1s, t_emb)
        x_bc1s = self.proj_out(x_bc1s)   # (B, mel_dim, 1, S)

        # BC1S → (B, mel_dim, S)
        return x_bc1s.squeeze(2)
