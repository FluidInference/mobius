"""ANE-eligible conditioning prefill: rank-4 KV cache, scatter-free write.

Trial 19 broke the "rank-5 KV cache -> ANECCompile FAILED" block for
`flowlm_step`; this applies the identical treatment to `cond_prefill`
(Trial 17), the last GPU model in the synthesis path. Math is unchanged
from the verified `TraceableCondPrefill` — only the formulation differs:

1. **Rank-4 cache I/O.** `cache{i} [2,1,L,16,64]` split into
   `k_cache{i}` / `v_cache{i}` `[1,L,16,64]`; same slot layout.
2. **Scatter-free block write.** The T_max-token scatter (with the
   Trial 17 dump-slot redirect for padded tokens) becomes a `[T,L]`
   one-hot assignment matrix applied as a matmul:
       M[t,l]   = (write_idx[t] == l)            # exactly one 1 per row
       written  = M^T @ k_flat                   # [L, H*D]
       covered  = (sum_t M[t,l]) > 0             # [L]
       new_k    = k_cache*(1-covered) + written
   For every real slot exactly one M entry is 1, so `written` is a
   single-term sum — bit-identical to scatter. Padded tokens all map to
   the dump slot (L-1), where `written` ACCUMULATES instead of taking
   the last write; that slot is excluded by the attention mask here and
   only ever read by flowlm after being legitimately overwritten (its
   write at pos==L-1 happens in the same step that first unmasks it),
   so the difference is unobservable.
3. **Rank-4 RoPE.** B=1 folds away: pairs viewed as `[T, H, D/2, 2]`.
4. **Additive attention mask** instead of `masked_fill(-inf)`.
5. **No cache NaN scrub** — the Swift host zero-fills caches
   (`emptyKVCacheState`). REQUIREMENT: unwritten slots are 0, not NaN.
6. **Layer-invariant hoisting.** The RoPE rotations, one-hot assign
   matrix, coverage vector, and mask bias depend only on (position,
   valid_len) — NOT on the layer — so they are computed ONCE in
   `forward` and reused by all 6 layers. Without this, the few
   scalar-driven ops the ANE compiler rejects (`sin`/`cos`/`equal`/
   `less_equal`) are rebuilt inside every layer and force six CPU<->ANE
   partition transitions (measured 17 ms under `.cpuAndNeuralEngine`);
   hoisted, there is a single CPU prologue. REQUIREMENT (already the
   host's behavior): all per-layer `position{i}` inputs are equal —
   the shared values are derived from `position0`. Per-layer
   `new_position{i}` outputs are still computed from each layer's own
   input.

I/O contract (per layer i in 0..5):
    inputs : conditioning [1, T_max, 1024], valid_len [1],
             k_cache{i} [1, L, H, D], v_cache{i} [1, L, H, D],
             position{i} [1]
    outputs: per layer (new_k_cache{i}, new_v_cache{i}, new_position{i});
             position advances by valid_len (not T_max).
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


class TraceableCondPrefillANE(nn.Module):
    """One-call conditioning prefill with rank-4 tensors and no scatter."""

    def __init__(self, num_layers: int = 6, max_seq_len: int = 512, t_max: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = 64
        self.max_seq_len = max_seq_len
        self.t_max = t_max  # Python-int constant: no aten::Int from traced shapes
        self.rope_max_period = 10000.0

        for i in range(self.num_layers):
            setattr(self, f"attn{i}_in_proj", nn.Linear(1024, 3 * 1024, bias=False))
            setattr(self, f"attn{i}_out_proj", nn.Linear(1024, 1024, bias=False))
            setattr(self, f"norm{i}_1", nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f"norm{i}_2", nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f"linear{i}_1", nn.Linear(1024, 4096, bias=False))
            setattr(self, f"linear{i}_2", nn.Linear(4096, 1024, bias=False))

        self.out_norm = nn.LayerNorm(1024, eps=1e-5)

        half_d = self.head_dim // 2
        ds = torch.arange(half_d, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(self.rope_max_period) * 2.0 / float(self.head_dim)))
        self.register_buffer("rope_freqs", freqs, persistent=False)  # [32]
        self.register_buffer(
            "slot_indices", torch.arange(max_seq_len, dtype=torch.float32), persistent=False
        )  # [L]
        self.register_buffer(
            "token_indices", torch.arange(t_max, dtype=torch.float32), persistent=False
        )  # [T]

    @classmethod
    def from_flowlm(
        cls, flow_lm_model, max_seq_len: int = 512, t_max: int = 256
    ) -> "TraceableCondPrefillANE":
        num_layers = len(flow_lm_model.transformer.layers)
        wrapper = cls(num_layers=num_layers, max_seq_len=max_seq_len, t_max=t_max)
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
        return wrapper

    def _shared_block_values(self, pos: torch.Tensor, valid_f: torch.Tensor, dtype):
        """Layer-invariant values, computed once per call.

        Returns (rotr, roti, assign, covered, mask_bias):
            rotr/roti  [T, 1, 32]  RoPE rotation at positions pos..pos+T-1
            assign     [T, L]      one-hot write rows (dump-slot redirect)
            covered    [L]         1.0 where any token writes
            mask_bias  [T, L]      0 attendable / -1e4 masked (additive)
        """
        T = self.t_max
        L = self.max_seq_len
        L_float = float(L)
        half_d = self.head_dim // 2

        ts = self.token_indices + pos.view(1)  # [T] absolute positions
        angles = self.rope_freqs.view(1, 1, half_d) * ts.view(T, 1, 1)  # [T, 1, 32]
        rotr = torch.cos(angles)
        roti = torch.sin(angles)

        # Write indices with the Trial 17 dump-slot redirect for padding.
        write_idx = ts - torch.floor(ts / L_float) * L_float  # mod L
        token_valid = (self.token_indices < valid_f.view(1)).to(dtype)  # [T]
        dump = float(L - 1)
        write_idx = write_idx * token_valid + dump * (1.0 - token_valid)

        assign = (write_idx.view(T, 1) == self.slot_indices.view(1, L)).to(dtype)  # [T, L]
        covered = torch.clamp(assign.sum(dim=0), max=1.0)  # [L]

        # valid: slot < pos + valid_len (true end, excludes padding + dump);
        # causal: slot <= pos + t. Additive form, fp16-safe.
        q_positions = pos.view(1, 1) + self.token_indices.view(T, 1)  # [T, 1]
        k_positions = self.slot_indices.view(1, L)  # [1, L]
        valid_end = (pos + valid_f).view(1, 1)
        mask = ((k_positions < valid_end) & (k_positions <= q_positions)).to(dtype)  # [T, L]
        mask_bias = (mask - 1.0) * 1e4

        return rotr, roti, assign, covered, mask_bias

    def _rope_apply(self, x: torch.Tensor, rotr: torch.Tensor, roti: torch.Tensor) -> torch.Tensor:
        """Apply precomputed RoPE rotation to `x [1, T, H, D]`, rank-4.

        Interleaved-pair convention, identical math to the original
        `_apply_rope_tensor`; pairs exposed via a `[T, H, D/2, 2]` reshape
        (B=1 folded) instead of the rank-5 `[B, T, H, D/2, 2]` view.
        """
        T = self.t_max
        H = self.num_heads
        half_d = self.head_dim // 2

        pairs = x.reshape(T, H, half_d, 2)  # rank-4
        xr = pairs[..., 0]  # [T, H, 32]
        xi = pairs[..., 1]

        xor_ = xr * rotr - xi * roti
        xoi = xr * roti + xi * rotr

        out = torch.stack([xor_, xoi], dim=-1)  # [T, H, 32, 2]
        return out.reshape(1, T, H, self.head_dim)

    def _streaming_attention_block(
        self,
        x: torch.Tensor,  # [1, T, 1024]
        in_proj: nn.Linear,
        out_proj: nn.Linear,
        k_cache: torch.Tensor,  # [1, L, H, D]
        v_cache: torch.Tensor,  # [1, L, H, D]
        position: torch.Tensor,  # [1]
        valid_f: torch.Tensor,  # [1]
        rotr: torch.Tensor,  # [T, 1, 32] (shared, hoisted)
        roti: torch.Tensor,  # [T, 1, 32]
        assign: torch.Tensor,  # [T, L]
        covered: torch.Tensor,  # [L]
        mask_bias: torch.Tensor,  # [T, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        T = self.t_max
        H = self.num_heads
        D = self.head_dim
        L = self.max_seq_len

        pos = position.float() if position.dtype != torch.float32 else position

        qkv = in_proj(x).reshape(1, T, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [1, T, H, D]

        q = self._rope_apply(q, rotr, roti)
        k = self._rope_apply(k, rotr, roti)

        # Scatter-free block write: [T, L] one-hot rows applied as a matmul.
        written_k = torch.matmul(assign.transpose(0, 1), k.reshape(T, H * D))  # [L, H*D]
        written_v = torch.matmul(assign.transpose(0, 1), v.reshape(T, H * D))
        keep = (1.0 - covered).view(1, L, 1, 1)
        new_k_cache = k_cache * keep + written_k.reshape(1, L, H, D)
        new_v_cache = v_cache * keep + written_v.reshape(1, L, H, D)

        # Attention of the T queries over the full cache.
        keys = new_k_cache.permute(0, 2, 1, 3)  # [1, H, L, D]
        values = new_v_cache.permute(0, 2, 1, 3)
        q4 = q.permute(0, 2, 1, 3)  # [1, H, T, D]

        scale = 1.0 / (D**0.5)
        scores = torch.matmul(q4, keys.transpose(-2, -1)) * scale  # [1, H, T, L]
        scores = scores + mask_bias.view(1, 1, T, L)

        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, values)  # [1, H, T, D]

        attn = attn.permute(0, 2, 1, 3).reshape(1, T, self.embed_dim)
        output = out_proj(attn)

        new_position = pos + valid_f.view(1)

        return output, new_k_cache, new_v_cache, new_position

    def forward(
        self,
        conditioning: torch.Tensor,  # [1, T_max, 1024] (host pads to T_max)
        valid_len: torch.Tensor,  # [1] true token count
        *cache_and_positions: torch.Tensor,
    ):
        """One-shot prefill of T_max conditioning tokens, gated by valid_len.

        `cache_and_positions` is grouped per layer:
        (k_cache0, v_cache0, position0, k_cache1, v_cache1, position1, ...).

        Returns per layer (new_k_cache{i}, new_v_cache{i}, new_position{i}).
        """
        x = conditioning
        k_caches = list(cache_and_positions[0::3])
        v_caches = list(cache_and_positions[1::3])
        positions = list(cache_and_positions[2::3])
        tail = []

        # Layer-invariant prologue (single CPU segment): all per-layer
        # positions are equal by host contract; derive shared values from
        # position0.
        pos0 = positions[0]
        pos0 = pos0.float() if pos0.dtype != torch.float32 else pos0
        valid_f = valid_len.float() if valid_len.dtype != torch.float32 else valid_len
        rotr, roti, assign, covered, mask_bias = self._shared_block_values(
            pos0, valid_f, x.dtype)

        for i in range(self.num_layers):
            residual = x
            x_norm = getattr(self, f"norm{i}_1")(x)
            attn_out, new_k, new_v, new_pos = self._streaming_attention_block(
                x_norm,
                getattr(self, f"attn{i}_in_proj"),
                getattr(self, f"attn{i}_out_proj"),
                k_caches[i],
                v_caches[i],
                positions[i],
                valid_f,
                rotr,
                roti,
                assign,
                covered,
                mask_bias,
            )
            x = residual + attn_out

            residual = x
            x_norm = getattr(self, f"norm{i}_2")(x)
            ffn_out = getattr(self, f"linear{i}_2")(F.gelu(getattr(self, f"linear{i}_1")(x_norm)))
            x = residual + ffn_out

            tail.extend([new_k, new_v, new_pos])

        return tuple(tail)


def test_parity_vs_original():
    """Parity vs the verified `TraceableCondPrefill` on the valid cache prefix.

    Caches start ZERO-FILLED (the Swift host contract), N=141 real tokens
    padded to T_max=256 with garbage. Compares each layer's K/V cache on
    [0, N) plus the returned positions. The dump slot (L-1) is excluded by
    design (accumulate vs last-write; unobservable through the mask).
    """
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    from traceable_cond_prefill import TraceableCondPrefill

    max_seq_len, T_max, N = 512, 256, 141
    print("Building original + ANE prefill wrappers from the same weights...")
    original = TraceableCondPrefill.from_flowlm(model.flow_lm, max_seq_len=max_seq_len, t_max=T_max)
    original.eval()
    ane = TraceableCondPrefillANE.from_flowlm(model.flow_lm, max_seq_len=max_seq_len, t_max=T_max)
    ane.eval()
    nl = ane.num_layers

    torch.manual_seed(0)
    cond = torch.randn(1, T_max, 1024)
    valid_len = torch.tensor([float(N)])

    orig_args = [cond, valid_len]
    ane_args = [cond, valid_len]
    for _ in range(nl):
        cache = torch.zeros(2, 1, max_seq_len, 16, 64)  # host contract: zeros
        orig_args.extend([cache.clone(), torch.zeros(1)])
        ane_args.extend([cache[0].clone(), cache[1].clone(), torch.zeros(1)])

    with torch.no_grad():
        ref = original(*orig_args)
        got = ane(*ane_args)

    worst = 0.0
    for li in range(nl):
        ref_cache, ref_pos = ref[2 * li], ref[2 * li + 1]
        nk, nv, npos = got[3 * li], got[3 * li + 1], got[3 * li + 2]
        d_k = (ref_cache[0][:, :N] - nk[:, :N]).abs().max().item()
        d_v = (ref_cache[1][:, :N] - nv[:, :N]).abs().max().item()
        d_p = (ref_pos - npos).abs().max().item()
        worst = max(worst, d_k, d_v, d_p)
        print(f"  layer {li}: d_k={d_k:.3e} d_v={d_v:.3e} d_pos={d_p:.3e}")

    print(f"\nworst abs diff on valid prefix [0,{N}): {worst:.3e}")
    assert worst < 1e-4, f"ANE prefill diverged from original: {worst}"
    print("Done! (parity vs TraceableCondPrefill)")


if __name__ == "__main__":
    test_parity_vs_original()
