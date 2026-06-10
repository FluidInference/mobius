"""Fused FlowLM step + flow decoder: ONE ANE dispatch per audio frame.

Trial 19 made `flowlm_step_ane` 100% ANE; Trial 16 made the fused flow
decoder 100% ANE. The host still pays TWO MLModel dispatches per frame
(flowlm -> flowdec) plus the fp32 boundary round-trip of the [1, 1, 1024]
`transformer_out` tensor between them — pure overhead, since the tensor is
produced and consumed on the same engine.

This wrapper composes the two verified modules as submodules and calls them
back-to-back inside one `forward`, so `transformer_out` never leaves the
graph (it stays an internal fp16 tensor) and the host makes ONE `predict()`
per frame. The math is the bit-identical concatenation of the two graphs:

    transformer_out, is_eos, caches' = flowlm_step_ane(sequence, ...)
    latent = latent_init
    for i in range(N):                         # N = 8, trace-time constants
        s, t = i/N, (i+1)/N
        latent += flow_net(transformer_out, s, t, latent) / N

I/O contract (per layer i in 0..5):
    inputs : sequence [1, 1, 32], latent_init [1, 32],
             k_cache{i} [1, L, H, D], v_cache{i} [1, L, H, D], position{i} [1]
    outputs: latent_final [1, 32], is_eos [1, 1, 1],
             then per layer (new_k_cache{i}, new_v_cache{i}, new_position{i})

BOS: Trial 22 removed the NaN-BOS protocol from `TraceableFlowLMStepANE`
(the ANE mangles NaN inputs before `isnan` evaluates); this fused wrapper
follows the same host contract — pass the BOS latent embedding itself as
`sequence` on the first generation step, never NaN, and there is no
`bos_emb` input.
"""
import os
import sys

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, _SCRIPT_DIR)  # for sibling traceable_* imports

from traceable_flow_decoder_fused import TraceableFlowDecoderFused
from traceable_flowlm_step_ane import TraceableFlowLMStepANE


class TraceableFlowLMFlowFused(nn.Module):
    """FlowLM step (T=1, rank-4, scatter-free) + N-step LSD Euler decode, fused."""

    def __init__(
        self,
        flowlm_step: TraceableFlowLMStepANE,
        flow_decoder: TraceableFlowDecoderFused,
    ):
        super().__init__()
        self.flowlm_step = flowlm_step
        self.flow_decoder = flow_decoder
        self.num_layers = flowlm_step.num_layers
        self.max_seq_len = flowlm_step.max_seq_len
        self.num_steps = flow_decoder.num_steps

    @classmethod
    def from_flowlm(
        cls, flow_lm, max_seq_len: int = 512, num_steps: int = 8
    ) -> "TraceableFlowLMFlowFused":
        """Build both verified submodules from the same FlowLM model."""
        flowlm_step = TraceableFlowLMStepANE.from_flowlm(flow_lm, max_seq_len=max_seq_len)
        flow_decoder = TraceableFlowDecoderFused.from_flowlm(flow_lm, num_steps=num_steps)
        return cls(flowlm_step, flow_decoder)

    def forward(
        self,
        sequence: torch.Tensor,  # [1, 1, 32] input latent (BOS latent on step 0, never NaN)
        latent_init: torch.Tensor,  # [1, 32] initial noise z_0 for the Euler loop
        *cache_and_positions: torch.Tensor,
    ):
        """One generation step + full LSD decode.

        `cache_and_positions` is grouped per layer:
        (k_cache0, v_cache0, position0, k_cache1, v_cache1, position1, ...).
        Length must be 3*num_layers.

        Returns:
            latent_final: [1, 32]
            is_eos: [1, 1, 1]
            then per layer (new_k_cache{i}, new_v_cache{i}, new_position{i}).
        """
        step_out = self.flowlm_step(sequence, *cache_and_positions)
        transformer_out = step_out[0]  # [1, 1, 1024]
        is_eos = step_out[1]
        cache_tail = step_out[2:]

        latent_final = self.flow_decoder(transformer_out.reshape(1, 1024), latent_init)

        return tuple([latent_final, is_eos] + list(cache_tail))


def test_parity_vs_separate():
    """Multi-step parity vs the two verified modules called back-to-back.

    Builds the fused module AND the separate pair from the same FlowLM
    weights, fills a random conditioning prefix (zeros beyond the valid
    length, matching the Swift host's zero-filled cache), then runs three
    autoregressive steps feeding each side its own updated caches. The fused
    graph is the literal concatenation of the two submodules, so every output
    must match exactly (0.0).
    """
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    max_seq_len = 512
    num_layers = 6
    num_steps = 8
    H, D = 16, 64
    prefix_len = 136  # simulated voice+text conditioning length

    print("Building fused + separate modules from the same weights...")
    fused = TraceableFlowLMFlowFused.from_flowlm(
        model.flow_lm, max_seq_len=max_seq_len, num_steps=num_steps
    )
    fused.eval()
    sep_step = TraceableFlowLMStepANE.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    sep_step.eval()
    sep_flow = TraceableFlowDecoderFused.from_flowlm(model.flow_lm, num_steps=num_steps)
    sep_flow.eval()

    torch.manual_seed(0)
    bos_emb = model.flow_lm.bos_emb.data

    # Shared conditioning prefix; unwritten slots are ZERO (host contract).
    sep_k, sep_v, sep_pos = [], [], []
    fused_k, fused_v, fused_pos = [], [], []
    for _ in range(num_layers):
        k = torch.zeros(1, max_seq_len, H, D)
        v = torch.zeros(1, max_seq_len, H, D)
        k[:, :prefix_len] = torch.randn(1, prefix_len, H, D) * 0.5
        v[:, :prefix_len] = torch.randn(1, prefix_len, H, D) * 0.5
        sep_k.append(k.clone())
        sep_v.append(v.clone())
        sep_pos.append(torch.tensor([float(prefix_len)]))
        fused_k.append(k.clone())
        fused_v.append(v.clone())
        fused_pos.append(torch.tensor([float(prefix_len)]))

    num_ar_steps = 3
    # No-NaN-BOS host contract (Trial 22): the BOS latent itself is the
    # first `sequence`, there is no bos_emb input.
    sequence = bos_emb.view(1, 1, 32).clone()
    worst = 0.0

    for step in range(num_ar_steps):
        latent_init = torch.randn(1, 32)

        sep_args = []
        for k, v, p in zip(sep_k, sep_v, sep_pos):
            sep_args.extend([k, v, p])
        fused_args = []
        for k, v, p in zip(fused_k, fused_v, fused_pos):
            fused_args.extend([k, v, p])

        with torch.no_grad():
            ref_step = sep_step(sequence, *sep_args)
            ref_latent = sep_flow(ref_step[0].reshape(1, 1024), latent_init.clone())
            got = fused(sequence, latent_init.clone(), *fused_args)

        d_latent = (ref_latent - got[0]).abs().max().item()
        d_eos = (ref_step[1] - got[1]).abs().max().item()
        worst = max(worst, d_latent, d_eos)

        # Compare caches + positions layer by layer.
        for i in range(num_layers):
            nk, nv, npos = ref_step[2 + 3 * i], ref_step[3 + 3 * i], ref_step[4 + 3 * i]
            gk, gv, gpos = got[2 + 3 * i], got[3 + 3 * i], got[4 + 3 * i]
            worst = max(
                worst,
                (nk - gk).abs().max().item(),
                (nv - gv).abs().max().item(),
                (npos - gpos).abs().max().item(),
            )
            sep_k[i], sep_v[i], sep_pos[i] = nk, nv, npos
            fused_k[i], fused_v[i], fused_pos[i] = gk, gv, gpos

        print(
            f"  step {step}: pos={fused_pos[0].item():.0f} "
            f"d_latent={d_latent:.3e} d_eos={d_eos:.3e}"
        )

        # Next step input: the real pipeline feeds latent_final back; for math
        # parity a shared random latent is equivalent (both sides get it).
        sequence = torch.randn(1, 1, 32)

    print(f"\nworst abs diff across {num_ar_steps} steps (latent + eos + caches): {worst:.3e}")
    assert worst < 1e-5, f"fused module diverged from separate pair: {worst}"
    print("Done! (parity vs TraceableFlowLMStepANE + TraceableFlowDecoderFused)")


if __name__ == "__main__":
    test_parity_vs_separate()
