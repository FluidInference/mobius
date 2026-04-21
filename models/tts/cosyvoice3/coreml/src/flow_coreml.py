"""CoreML-traceable wrapper for CosyVoice3 CausalMaskedDiffWithDiT (finalize=True).

Strategy
--------
The upstream `flow.inference(streaming=False, finalize=True)` path is mostly
traceable already; this wrapper just locks the shapes and bypasses the bits
that depend on `.item()` / dynamic Python control flow:

  * `make_pad_mask(token_len)` becomes a constant ones mask (we always pass a
    densely-packed token sequence).
  * `add_optional_chunk_mask(..., static_chunk_size=0)` falls through to the
    trivial "use the input mask as-is" branch and we precompute the (1,1,L,L)
    full attention mask once.
  * `solve_euler` unrolls naturally during tracing (Python `for` over a fixed
    `range(1, n_timesteps+1)`).
  * Classifier-Free Guidance is the upstream 2x-batch strategy.

Shapes (single static bucket per mlpackage)
-------------------------------------------
Inputs:
    token_total      : (1, N_total) int64    -- prompt_token | new_token
    num_prompt_tokens: (1,)         int32   -- where prompt ends (≤ N_total)
    prompt_feat      : (1, N_total*2, 80) float32  -- prompt mel padded with 0s
    embedding        : (1, 192) float32

Output:
    mel              : (1, 80, N_total*2) float32  -- full mel (prompt+new)
    num_prompt_mel   : (1,) int32  -- = num_prompt_tokens * 2 (caller slices)

The caller is expected to:
    new_mel = mel[:, :, num_prompt_mel : num_prompt_mel + num_new_tokens*2]
which mirrors the upstream `feat[:, :, mel_len1:]` slicing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowCoreML(nn.Module):
    """Wraps a loaded CausalMaskedDiffWithDiT for CoreML tracing.

    The wrapper binds the upstream module directly (no weight copy); pass the
    already-loaded `flow` object (state_dict applied + .eval()) into the ctor.
    """

    def __init__(self, flow: nn.Module, n_total_tokens: int, n_timesteps: int = 10):
        super().__init__()
        self.flow = flow
        self.N = n_total_tokens
        self.M = n_total_tokens * flow.token_mel_ratio  # mel frames at output
        self.n_timesteps = n_timesteps
        self.token_mel_ratio = flow.token_mel_ratio
        self.cfg_rate = flow.decoder.inference_cfg_rate
        self.t_scheduler = flow.decoder.t_scheduler

        # Precompute t_span (constant)
        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        self.register_buffer("t_span", t_span, persistent=False)

        # Precompute initial noise z (deterministic — upstream uses
        # CausalConditionalCFM.rand_noise[:, :, :M])
        z0 = flow.decoder.rand_noise[:, :, : self.M].clone()
        self.register_buffer("z0", z0, persistent=False)

        # Precompute full-attention mask (1,1,M,M) since static_chunk_size=0
        # falls through to attn_mask = mask_in.repeat(1, M, 1).unsqueeze(1).
        # Build the densely-packed all-ones mask.
        attn_mask = torch.ones(1, 1, self.M, self.M, dtype=torch.bool)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(
        self,
        token_total: torch.Tensor,         # (1, N) int64
        num_prompt_tokens: torch.Tensor,   # (1,) int32
        prompt_feat: torch.Tensor,         # (1, M, 80) float32 (zero-padded after num_prompt_tokens*2)
        embedding: torch.Tensor,           # (1, 192) float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow = self.flow
        N, M = self.N, self.M

        # ---- speaker embedding ----
        emb = F.normalize(embedding, dim=1)
        emb = flow.spk_embed_affine_layer(emb)  # (1, 80)

        # ---- token embedding (mask is all-ones for our densely-packed input) ----
        tok = flow.input_embedding(torch.clamp(token_total, min=0))  # (1, N, 80)

        # ---- pre-lookahead causal conv stack (finalize=True path) ----
        h = flow.pre_lookahead_layer(tok)  # (1, N, 80)

        # ---- token → mel rate (repeat each token twice along time) ----
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)  # (1, M, 80)

        # ---- conditioning: prompt mel padded with zeros (caller responsibility) ----
        # `prompt_feat` is already shape (1, M, 80) with zeros after num_prompt_tokens*2.
        cond = prompt_feat.transpose(1, 2).contiguous()  # (1, 80, M)
        mu = h.transpose(1, 2).contiguous()              # (1, 80, M)

        # ---- diffusion with classifier-free guidance ----
        mel = self._solve_euler(mu=mu, cond=cond, spks=emb)  # (1, 80, M)

        num_prompt_mel = (num_prompt_tokens * self.token_mel_ratio).to(torch.int32)
        return mel, num_prompt_mel

    # --------------------------------------------------------------------- #
    # Euler ODE solver — mirrors CausalConditionalCFM.solve_euler() but
    # without the in-place writes / CFG buffers; uses pre-stacked CFG batch.
    # --------------------------------------------------------------------- #
    def _solve_euler(
        self,
        mu: torch.Tensor,    # (1, 80, M)
        cond: torch.Tensor,  # (1, 80, M)
        spks: torch.Tensor,  # (1, 80)
    ) -> torch.Tensor:
        x = self.z0  # (1, 80, M)
        # Constant zeros for the unconditional branch (CFG drops cond/spks/mu).
        zero_mu = torch.zeros_like(mu)
        zero_spks = torch.zeros_like(spks)
        zero_cond = torch.zeros_like(cond)

        cfg_rate = self.cfg_rate
        attn_mask_full = self.attn_mask  # (1,1,M,M) — but estimator wants (B,1,M)

        # The DiT.forward signature takes mask as (B, 1, M). add_optional_chunk_mask
        # then expands to (B, 1, M, M) internally. We feed all-ones mask.
        mask_in = torch.ones(2, 1, self.M, dtype=mu.dtype, device=mu.device)

        for step in range(1, self.n_timesteps + 1):
            t = self.t_span[step - 1]            # scalar
            dt = self.t_span[step] - t           # scalar

            # CFG batch: [cond, uncond] along dim 0
            x_in = torch.cat([x, x], dim=0)
            mu_in = torch.cat([mu, zero_mu], dim=0)
            spks_in = torch.cat([spks, zero_spks], dim=0)
            cond_in = torch.cat([cond, zero_cond], dim=0)
            t_in = t.repeat(2)

            dphi = self.flow.decoder.estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=False
            )  # (2, 80, M)

            d_cond, d_uncond = torch.split(dphi, [1, 1], dim=0)
            dphi_dt = (1.0 + cfg_rate) * d_cond - cfg_rate * d_uncond
            x = x + dt * dphi_dt

        return x.float()
