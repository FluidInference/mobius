"""ANE-eligible FlowLM step: rank-4 KV cache, scatter-free write, T=1.

Phase 7 measured that `flowlm_step` cannot reach the ANE at ANY precision:
the rank-5 KV cache `(2, 1, 512, 16, 64)` makes `ANECCompile` fail outright
(see TRIALS.md "Phase 7 — MEASURED"). The graph also contains a second
rank-5 tensor (the RoPE interleaved-pair view `[B, T, H, 32, 2]`) and a
`scatter` cache write — both ANE-hostile.

This wrapper is the flow-decoder-fusion playbook applied to the step model
(Trial 16: the single-step kernel was always rejected; a reshaped,
scatter-free graph flipped 0% -> 100% ANE). Math is exactly equivalent to
`TraceableFlowLMStep` for the host's actual call pattern; what changes is
only the formulation:

1. **Rank-4 everywhere.** Each `cache{i} [2, 1, L, H, D]` input/output is
   split into `k_cache{i}` / `v_cache{i}` `[1, L, H, D]`. The cache slot
   layout (absolute position modulo L) is unchanged, so the caches written
   by `cond_step` / `cond_prefill` can be fed in directly after splitting
   the leading K/V axis — no cond-side changes.
2. **T=1 specialization.** Generation always runs one frame per call
   (43 calls/utt); cond/prefill own the multi-token path. With T=1 the
   RoPE pair view becomes a rank-4 `[1, H, D/2, 2]` reshape and the
   valid/causal masks collapse into one comparison (`slot <= position`).
3. **Scatter-free cache write.** The circular-buffer `scatter` becomes a
   one-hot multiply-add: `onehot = (arange(L) == position % L)`,
   `new_k = k_cache * (1 - onehot) + k * onehot`. Bit-identical slot
   semantics (including the modulo wrap), pure broadcast/elementwise ops.
4. **Additive attention mask.** `masked_fill(~mask, -inf)` becomes
   `scores + (mask - 1) * 1e4`. Masked logits underflow to exactly 0
   after softmax in fp32 and fp16, and there is never a fully-masked row
   (the current position is always valid), so this cannot NaN.
5. **No NaN scrubbing of the cache.** The FluidAudio host zero-fills
   fresh caches (`PocketTtsSynthesizer+KVCache.swift::emptyKVCacheState`),
   so the `isnan` scrub of K/V in the original trace is dead weight on
   the host path. REQUIREMENT: unwritten cache slots must be 0, not NaN.
6. **No NaN-BOS protocol.** The rank-5 packs signal BOS with a NaN-filled
   `sequence` that the graph replaces via `where(isnan(...), bos_emb, ...)`.
   The ANE mangles NaN inputs BEFORE `isnan` evaluates (measured: CPU-only
   matches torch at 3e-2 with NaN BOS; cpuAndNeuralEngine diverges at 2.9,
   BOS silently dropped). This graph therefore has NO `bos_emb` input and
   no isnan. HOST CONTRACT: pass the BOS latent embedding itself as
   `sequence` on the first generation step; never pass NaN.

I/O contract (per layer i in 0..5):
    inputs : sequence [1, 1, 32]  (BOS step: the BOS latent, NOT NaN),
             k_cache{i} [1, L, H, D], v_cache{i} [1, L, H, D],
             position{i} [1]
    outputs: transformer_out [1, 1, 1024], is_eos [1, 1, 1],
             then per layer (new_k_cache{i}, new_v_cache{i}, new_position{i})
"""
import math
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...


class TraceableFlowLMStepANE(nn.Module):
    """FlowLM step (T=1) with rank-4 tensors and a scatter-free KV write."""

    def __init__(self, num_layers: int = 6, max_seq_len: int = 512):
        super().__init__()

        self.num_layers = num_layers
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = 64
        self.max_seq_len = max_seq_len
        self.rope_max_period = 10000.0

        self.input_linear = nn.Linear(32, 1024, bias=False)

        for i in range(self.num_layers):
            setattr(self, f"attn{i}_in_proj", nn.Linear(1024, 3 * 1024, bias=False))
            setattr(self, f"attn{i}_out_proj", nn.Linear(1024, 1024, bias=False))
            setattr(self, f"norm{i}_1", nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f"norm{i}_2", nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f"linear{i}_1", nn.Linear(1024, 4096, bias=False))
            setattr(self, f"linear{i}_2", nn.Linear(4096, 1024, bias=False))

        self.out_norm = nn.LayerNorm(1024, eps=1e-5)
        self.out_eos = nn.Linear(1024, 1)

        # Constant buffers (fold to MIL consts at trace time).
        half_d = self.head_dim // 2
        ds = torch.arange(half_d, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(self.rope_max_period) * 2.0 / float(self.head_dim)))
        self.register_buffer("rope_freqs", freqs, persistent=False)  # [32]
        self.register_buffer(
            "slot_indices", torch.arange(max_seq_len, dtype=torch.float32), persistent=False
        )  # [L]

    @classmethod
    def from_flowlm(cls, flow_lm_model, max_seq_len: int = 512) -> "TraceableFlowLMStepANE":
        """Create the ANE step model from the original FlowLM model."""
        num_layers = len(flow_lm_model.transformer.layers)
        wrapper = cls(num_layers=num_layers, max_seq_len=max_seq_len)

        wrapper.input_linear.weight.data.copy_(flow_lm_model.input_linear.weight.data)

        for i, layer in enumerate(flow_lm_model.transformer.layers):
            getattr(wrapper, f"attn{i}_in_proj").weight.data.copy_(layer.self_attn.in_proj.weight.data)
            getattr(wrapper, f"attn{i}_out_proj").weight.data.copy_(layer.self_attn.out_proj.weight.data)
            getattr(wrapper, f"norm{i}_1").weight.data.copy_(layer.norm1.weight.data)
            getattr(wrapper, f"norm{i}_1").bias.data.copy_(layer.norm1.bias.data)
            getattr(wrapper, f"norm{i}_2").weight.data.copy_(layer.norm2.weight.data)
            getattr(wrapper, f"norm{i}_2").bias.data.copy_(layer.norm2.bias.data)
            getattr(wrapper, f"linear{i}_1").weight.data.copy_(layer.linear1.weight.data)
            getattr(wrapper, f"linear{i}_2").weight.data.copy_(layer.linear2.weight.data)

        wrapper.out_norm.weight.data.copy_(flow_lm_model.out_norm.weight.data)
        wrapper.out_norm.bias.data.copy_(flow_lm_model.out_norm.bias.data)
        wrapper.out_eos.weight.data.copy_(flow_lm_model.out_eos.weight.data)
        wrapper.out_eos.bias.data.copy_(flow_lm_model.out_eos.bias.data)

        return wrapper

    def _rope_rotate_t1(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Rotate a single-position tensor `x [1, H, D]` by RoPE at `pos [1]`.

        Interleaved-pair convention, identical to the original
        `_apply_rope_tensor`, but staying rank-4: pairs are exposed via a
        `[1, H, D/2, 2]` reshape instead of the rank-5 `[B, T, H, D/2, 2]`
        view.
        """
        H = self.num_heads
        half_d = self.head_dim // 2

        angles = self.rope_freqs * pos.view(1, 1, 1)  # [1, 1, 32]
        rotr = torch.cos(angles)  # [1, 1, 32]
        roti = torch.sin(angles)

        pairs = x.reshape(1, H, half_d, 2)  # rank-4
        xr = pairs[..., 0]  # [1, H, 32]
        xi = pairs[..., 1]

        xor_ = xr * rotr - xi * roti
        xoi = xr * roti + xi * rotr

        out = torch.stack([xor_, xoi], dim=-1)  # [1, H, 32, 2]
        return out.reshape(1, H, self.head_dim)

    def _streaming_attention_t1(
        self,
        x: torch.Tensor,  # [1, 1, 1024]
        in_proj: nn.Linear,
        out_proj: nn.Linear,
        k_cache: torch.Tensor,  # [1, L, H, D]
        v_cache: torch.Tensor,  # [1, L, H, D]
        position: torch.Tensor,  # [1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """T=1 streaming attention; scatter-free circular-buffer write."""
        H = self.num_heads
        D = self.head_dim
        L = self.max_seq_len
        L_float = float(L)

        pos = position.float() if position.dtype != torch.float32 else position  # [1]

        # Q, K, V for the single new position.
        qkv = in_proj(x).reshape(1, 3, H, D)  # T=1 folded away
        q = qkv[:, 0]  # [1, H, D]
        k = qkv[:, 1]
        v = qkv[:, 2]

        q = self._rope_rotate_t1(q, pos)
        k = self._rope_rotate_t1(k, pos)

        # Scatter-free circular-buffer write at slot = position % L.
        write_idx = pos - torch.floor(pos / L_float) * L_float  # [1]
        onehot = (self.slot_indices == write_idx).to(x.dtype)  # [L]
        onehot = onehot.view(1, L, 1, 1)
        new_k_cache = k_cache * (1.0 - onehot) + k.view(1, 1, H, D) * onehot
        new_v_cache = v_cache * (1.0 - onehot) + v.view(1, 1, H, D) * onehot

        # Attention of the single query over the full cache.
        keys = new_k_cache.permute(0, 2, 1, 3)  # [1, H, L, D]
        values = new_v_cache.permute(0, 2, 1, 3)
        q4 = q.view(1, H, 1, D)

        scale = 1.0 / (D**0.5)
        scores = torch.matmul(q4, keys.transpose(-2, -1)) * scale  # [1, H, 1, L]

        # For T=1, valid (slot < pos+1) and causal (slot <= pos) coincide.
        # Additive mask: 0 where attendable, -1e4 elsewhere (fp16-safe; the
        # masked weights underflow to exactly 0 after softmax).
        mask = (self.slot_indices <= pos.view(1, 1)).to(x.dtype)  # [1, L]
        scores = scores + (mask - 1.0).view(1, 1, 1, L) * 1e4

        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, values)  # [1, H, 1, D]

        attn = attn.permute(0, 2, 1, 3).reshape(1, 1, self.embed_dim)
        output = out_proj(attn)

        new_position = pos + 1.0

        return output, new_k_cache, new_v_cache, new_position

    def forward(
        self,
        sequence: torch.Tensor,  # [1, 1, 32] input latent (BOS latent on step 0)
        *cache_and_positions: torch.Tensor,
    ):
        """One generation step.

        `cache_and_positions` is grouped per layer:
        (k_cache0, v_cache0, position0, k_cache1, v_cache1, position1, ...).
        Length must be 3*num_layers.

        BOS: the host passes the BOS latent embedding as `sequence` on the
        first step (no NaN protocol — see module docstring, item 6).

        Returns:
            transformer_out: [1, 1, 1024]
            is_eos: [1, 1, 1]
            then per layer (new_k_cache{i}, new_v_cache{i}, new_position{i}).
        """
        x = self.input_linear(sequence)  # [1, 1, 1024]

        k_caches = list(cache_and_positions[0::3])
        v_caches = list(cache_and_positions[1::3])
        positions = list(cache_and_positions[2::3])
        tail = []

        for i in range(self.num_layers):
            residual = x
            x_norm = getattr(self, f"norm{i}_1")(x)
            attn_out, new_k, new_v, new_pos = self._streaming_attention_t1(
                x_norm,
                getattr(self, f"attn{i}_in_proj"),
                getattr(self, f"attn{i}_out_proj"),
                k_caches[i],
                v_caches[i],
                positions[i],
            )
            x = residual + attn_out

            residual = x
            x_norm = getattr(self, f"norm{i}_2")(x)
            ffn_out = getattr(self, f"linear{i}_2")(F.gelu(getattr(self, f"linear{i}_1")(x_norm)))
            x = residual + ffn_out

            tail.extend([new_k, new_v, new_pos])

        x = self.out_norm(x)
        is_eos = self.out_eos(x)

        return tuple([x, is_eos] + tail)


def test_parity_vs_original():
    """Multi-step parity vs the verified `TraceableFlowLMStep`.

    Builds both wrappers from the same FlowLM weights, fills a random
    conditioning prefix (zeros beyond the valid length, matching the Swift
    host's zero-filled cache), then runs several autoregressive steps
    feeding each model its own updated caches. Compares transformer_out,
    EOS, and cache contents at every step.
    """
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    max_seq_len = 512
    num_layers = 6
    H, D = 16, 64
    prefix_len = 136  # simulated voice+text conditioning length

    from traceable_flowlm_step import TraceableFlowLMStep

    print("Building original + ANE wrappers from the same weights...")
    original = TraceableFlowLMStep.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    original.eval()
    ane = TraceableFlowLMStepANE.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    ane.eval()

    torch.manual_seed(0)
    bos_emb = model.flow_lm.bos_emb.data

    # Shared conditioning prefix; unwritten slots are ZERO (host contract).
    caches5 = []  # original layout [2, 1, L, H, D]
    for _ in range(num_layers):
        cache = torch.zeros(2, 1, max_seq_len, H, D)
        cache[:, :, :prefix_len] = torch.randn(2, 1, prefix_len, H, D) * 0.5
        caches5.append(cache)

    orig_caches = [c.clone() for c in caches5]
    orig_positions = [torch.tensor([float(prefix_len)]) for _ in range(num_layers)]
    ane_k = [c[0].clone() for c in caches5]  # [1, L, H, D]
    ane_v = [c[1].clone() for c in caches5]
    ane_positions = [torch.tensor([float(prefix_len)]) for _ in range(num_layers)]

    num_steps = 5
    # BOS protocols differ: the original takes NaN + bos_emb and substitutes
    # in-graph; the ANE wrapper takes the BOS latent directly (host contract
    # — the ANE mangles NaN inputs, see module docstring item 6).
    sequence_orig = torch.full((1, 1, 32), float("nan"))
    sequence_ane = bos_emb.view(1, 1, 32).clone()
    worst = 0.0

    for step in range(num_steps):
        orig_args = []
        for c, p in zip(orig_caches, orig_positions):
            orig_args.extend([c, p])
        ane_args = []
        for k, v, p in zip(ane_k, ane_v, ane_positions):
            ane_args.extend([k, v, p])

        with torch.no_grad():
            ref = original(sequence_orig, bos_emb, *orig_args)
            got = ane(sequence_ane, *ane_args)

        ref_out, ref_eos = ref[0], ref[1]
        got_out, got_eos = got[0], got[1]

        d_out = (ref_out - got_out.view_as(ref_out)).abs().max().item()
        d_eos = (ref_eos - got_eos.view_as(ref_eos)).abs().max().item()
        worst = max(worst, d_out, d_eos)

        # Compare caches + positions layer by layer.
        for i in range(num_layers):
            new_c = ref[2 + 2 * i]
            new_p = ref[3 + 2 * i]
            nk, nv, npos = got[2 + 3 * i], got[3 + 3 * i], got[4 + 3 * i]
            d_k = (new_c[0] - nk).abs().max().item()
            d_v = (new_c[1] - nv).abs().max().item()
            d_p = (new_p - npos).abs().max().item()
            worst = max(worst, d_k, d_v, d_p)
            orig_caches[i] = new_c
            orig_positions[i] = new_p
            ane_k[i], ane_v[i], ane_positions[i] = nk, nv, npos

        print(
            f"  step {step}: pos={ane_positions[0].item():.0f} "
            f"d_out={d_out:.3e} d_eos={d_eos:.3e}"
        )

        # Next step input: feed the (shared) transformer_out projected back is
        # not the real pipeline (flow decode happens outside) — a random
        # latent is fine for math parity.
        sequence_orig = torch.randn(1, 1, 32)
        sequence_ane = sequence_orig.clone()

    print(f"\nworst abs diff across {num_steps} steps (outs + caches): {worst:.3e}")
    assert worst < 1e-4, f"ANE wrapper diverged from original: {worst}"
    print("Done! (parity vs TraceableFlowLMStep)")


if __name__ == "__main__":
    test_parity_vs_original()
