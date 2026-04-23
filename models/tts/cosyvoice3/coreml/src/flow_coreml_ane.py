"""ANE-port variant of `FlowCoreML`.

Identical schema to `flow_coreml.FlowCoreML` — same inputs/outputs, same
Euler loop, same CFG batching — but the DiT estimator is replaced with
`ANEDiT` running on BC1S 4D layout.

Usage:
    flow_ane = FlowCoreMLANE.build_from_flow(
        flow=loaded_flow,         # CausalMaskedDiffWithDiT, .eval()
        n_total_tokens=N_total,
        n_timesteps=10,
    )
    flow_ane.eval()
    traced = torch.jit.trace(flow_ane, example_inputs, strict=False)

`build_from_flow` does three things:
    1. Infers (dim, depth, heads, dim_head, ff_mult, mel_dim, mu_dim,
       max_seq_len) from the host DiT.
    2. Constructs `ANEDiT` reusing the host `time_embed` and `input_embed`
       instances (no weight copy — direct module references).
    3. Ports the transformer-block / norm_out / proj_out state dict via
       `convert_state_dict_to_ane` and loads it into the new ANEDiT.

The wrapper then behaves identically to `FlowCoreML` except for the
estimator call, which uses the ANE signature (no `streaming` kwarg).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dit_ane import ANEDiT
from src.state_dict_port import convert_state_dict_to_ane, strip_unused_rotary


def _infer_ff_mult(host_dit: nn.Module, dim: int) -> int:
    """Extract ff_mult from a host DiTBlock.

    Host FeedForward inner_dim = int(dim * mult). We read inner from the
    first block's `ff.ff[0][0]` (nn.Linear, project_in).
    """
    block0 = host_dit.transformer_blocks[0]
    project_in_linear = block0.ff.ff[0][0]  # Linear(dim, inner_dim)
    inner = project_in_linear.out_features
    mult = inner // dim
    assert mult * dim == inner, f"inner_dim {inner} not a clean multiple of dim {dim}"
    return mult


def build_ane_dit_from_host(host_dit: nn.Module, max_seq_len: int) -> ANEDiT:
    """Construct an `ANEDiT` mirroring `host_dit` with weights ported.

    Reuses host `time_embed` and `input_embed` by reference (not copy).
    """
    dim = host_dit.dim
    depth = host_dit.depth
    block0 = host_dit.transformer_blocks[0]
    heads = block0.attn.heads
    dim_head = block0.attn.inner_dim // heads
    ff_mult = _infer_ff_mult(host_dit, dim)
    mel_dim = host_dit.proj_out.out_features

    ane = ANEDiT(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        max_seq_len=max_seq_len,
        dropout=0.0,
        ff_mult=ff_mult,
        mel_dim=mel_dim,
        time_embed=host_dit.time_embed,
        input_embed=host_dit.input_embed,
    )

    # Port state dict — only entries corresponding to transformer_blocks /
    # norm_out / proj_out are reshaped; time_embed / input_embed come
    # through unchanged and land in the aliased host modules.
    host_sd = host_dit.state_dict()
    host_sd = strip_unused_rotary(host_sd)
    ane_sd = convert_state_dict_to_ane(host_sd)

    # Drop rotary cos/sin buffers from the target's expected keys — they
    # are registered non-persistent in ANERotaryEmbedding, so load_state_dict
    # with strict=False is safe. We also won't see host keys for them.
    #
    # Note: three attempts to rewrite `input_embed.conv_pos_embed` for ANE
    # all failed to eliminate the 77-op CPU island:
    #   - Option A (16× `groups=1` Conv1d split + Mish, math-equivalent):
    #     fp32 parity passed (cum MAE 2.907e-05); ANE compile FAILED (11).
    #   - Option C (16× split + Mish→SiLU, NOT math-equivalent): ANE
    #     compile also FAILED (11) — rules out Mish.
    #   - Option L (dense `groups=1` Conv1d with block-diagonal weight,
    #     math-equivalent, 16× FLOPs, +129 MB fp16 weight): ANE compile
    #     SUCCEEDED, but the dense conv still gets evicted to CPU by
    #     ANEF (77 CPU ops unchanged) and adds load time + FLOPs → p50
    #     regressed 904 ms → 1035 ms. The rejection isn't about
    #     `groups>1`; it's about `Conv1d(1024, 1024, k=31)` — kernel
    #     width × dim exceeds ANEF tile limits regardless of group count.
    # Host grouped conv stays on CPU (~77 ops); Flow p50 ~0.9 s warm,
    # ~1.6 s cold. `src/conv_pos_ane.py` is retained for reference.

    missing, unexpected = ane.load_state_dict(ane_sd, strict=False)
    # time_embed.* and input_embed.* land in aliased submodules — they'll
    # show up in `missing` only if the host dict was stripped of them,
    # which it isn't. Any other missing/unexpected is a bug.
    hard_missing = [
        k for k in missing
        if not (k.endswith("cos_table") or k.endswith("sin_table"))
    ]
    hard_unexpected = list(unexpected)
    if hard_missing or hard_unexpected:
        raise RuntimeError(
            f"ANEDiT state_dict port mismatch.\n"
            f"  missing (non-rotary): {hard_missing[:10]}{'...' if len(hard_missing) > 10 else ''}\n"
            f"  unexpected         : {hard_unexpected[:10]}{'...' if len(hard_unexpected) > 10 else ''}"
        )

    return ane


class FlowCoreMLANE(nn.Module):
    """Mirror of `FlowCoreML` with an ANE-ported DiT estimator.

    Same input / output schema as `FlowCoreML.forward` — the Swift-side
    caller sees no difference.
    """

    def __init__(self, flow: nn.Module, ane_dit: ANEDiT, n_total_tokens: int, n_timesteps: int = 10):
        super().__init__()
        self.flow = flow
        self.ane_dit = ane_dit
        self.N = n_total_tokens
        self.M = n_total_tokens * flow.token_mel_ratio
        self.n_timesteps = n_timesteps
        self.token_mel_ratio = flow.token_mel_ratio
        self.cfg_rate = flow.decoder.inference_cfg_rate
        self.t_scheduler = flow.decoder.t_scheduler

        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        self.register_buffer("t_span", t_span, persistent=False)

        z0 = flow.decoder.rand_noise[:, :, : self.M].clone()
        self.register_buffer("z0", z0, persistent=False)

    @classmethod
    def build_from_flow(cls, flow: nn.Module, n_total_tokens: int, n_timesteps: int = 10):
        max_seq_len = n_total_tokens * flow.token_mel_ratio
        ane_dit = build_ane_dit_from_host(flow.decoder.estimator, max_seq_len=max_seq_len)
        return cls(flow=flow, ane_dit=ane_dit, n_total_tokens=n_total_tokens, n_timesteps=n_timesteps)

    def forward(
        self,
        token_total: torch.Tensor,
        num_prompt_tokens: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow = self.flow
        N, M = self.N, self.M

        emb = F.normalize(embedding, dim=1)
        emb = flow.spk_embed_affine_layer(emb)

        tok = flow.input_embedding(torch.clamp(token_total, min=0))
        h = flow.pre_lookahead_layer(tok)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        cond = prompt_feat.transpose(1, 2).contiguous()
        mu = h.transpose(1, 2).contiguous()

        mel = self._solve_euler(mu=mu, cond=cond, spks=emb)

        num_prompt_mel = (num_prompt_tokens * self.token_mel_ratio).to(torch.int32)
        return mel, num_prompt_mel

    def _solve_euler(
        self,
        mu: torch.Tensor,
        cond: torch.Tensor,
        spks: torch.Tensor,
    ) -> torch.Tensor:
        x = self.z0
        zero_mu = torch.zeros_like(mu)
        zero_spks = torch.zeros_like(spks)
        zero_cond = torch.zeros_like(cond)

        cfg_rate = self.cfg_rate

        # Dense-pack inference: no effective padding. ANEDiT accepts None
        # mask and skips the softmax `where`. This avoids a (B,1,S,S) bool
        # tensor and a fp32-critical `where` op on the ANE graph.
        mask_in = None

        for step in range(1, self.n_timesteps + 1):
            t = self.t_span[step - 1]
            dt = self.t_span[step] - t

            x_in = torch.cat([x, x], dim=0)
            mu_in = torch.cat([mu, zero_mu], dim=0)
            spks_in = torch.cat([spks, zero_spks], dim=0)
            cond_in = torch.cat([cond, zero_cond], dim=0)
            t_in = t.repeat(2)

            dphi = self.ane_dit(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

            d_cond, d_uncond = torch.split(dphi, [1, 1], dim=0)
            dphi_dt = (1.0 + cfg_rate) * d_cond - cfg_rate * d_uncond
            x = x + dt * dphi_dt

        # Keep fp16 end-to-end; the mlpackage declares mel as np.float16.
        # Returning fp16 lets coremltools leave the final CFG-combine math
        # and proj_out conv in fp16 on ANE instead of trailing a fp32 cast
        # that cascades casts back up the graph and evicts the final 20
        # convs to CPU.
        return x
