"""Traceable conditioning PREFILL — fills the KV cache for the whole voice+text
conditioning block in a SINGLE call instead of one token at a time.

`TraceableCondStep` processes T=1 per call, so the host dispatches it once per
conditioning token (18-141 `predict()` calls per chunk; 122 ms / 13% of
wall-time in the profile). This wrapper processes T=T_max tokens in one call.

Why this is safe where Trial 5 / Trial 8 were not:
  - Trial 5 (RangeDim) broke because a DYNAMIC sequence dim makes CoreML's
    `scatter_along_axis` assert. Here T is a COMPILE-TIME constant (T_max);
    there are no dynamic shapes at all.
  - Trial 8 (zero-padding) corrupted the cache because padded tokens passed
    through LayerNorm/FFN bias, wrote non-zero KV entries, and advanced the
    position counter past the true length. Here a runtime `valid_len` VALUE
    (not a shape) gates both:
      * padded tokens' cache writes are redirected to a throw-away "dump"
        slot that the attention mask always excludes, so they never corrupt
        a real position, and
      * the returned position advances by `valid_len`, not T_max.
    The host pads `conditioning` to T_max and passes the true token count as
    `valid_len`; padded rows are computed but discarded (cond_step returns
    only caches + positions, never `x`).

Attention/RoPE math is copied verbatim from the verified `TraceableCondStep`
(Trial 10/11) so KV-cache contents stay bit-identical to the per-token path
for the valid prefix.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
import math
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...


class TraceableCondPrefill(nn.Module):
    """Processes T_max conditioning tokens in one call, gated by `valid_len`.

    Input:
        conditioning [1, T_max, 1024]  (host pads to T_max with anything)
        valid_len    [1]               (true token count; padded rows ignored)
        cache_i      [2, 1, max_seq_len, 16, 64], position_i [1]  (per layer)
    Output:
        cache_i_out, position_i_out per layer (position advanced by valid_len)
    """

    def __init__(self, num_layers: int = 6, max_seq_len: int = 512, t_max: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = 64
        self.max_seq_len = max_seq_len
        # Fixed conditioning-block length. Stored as a Python int so every
        # view/reshape/expand uses a CONSTANT, not a value unpacked from the
        # traced tensor shape — the latter emits aten::Int nodes that
        # coremltools 9 / torch 2.12 cannot const-fold ("only 0-dimensional
        # arrays can be converted to Python scalars").
        self.t_max = t_max
        self.rope_max_period = 10000.0

        for i in range(self.num_layers):
            setattr(self, f"attn{i}_in_proj", nn.Linear(1024, 3 * 1024, bias=False))
            setattr(self, f"attn{i}_out_proj", nn.Linear(1024, 1024, bias=False))
            setattr(self, f"norm{i}_1", nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f"norm{i}_2", nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f"linear{i}_1", nn.Linear(1024, 4096, bias=False))
            setattr(self, f"linear{i}_2", nn.Linear(4096, 1024, bias=False))

        self.out_norm = nn.LayerNorm(1024, eps=1e-5)

    @classmethod
    def from_flowlm(cls, flow_lm_model, max_seq_len: int = 512, t_max: int = 256) -> "TraceableCondPrefill":
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

    def _apply_rope_tensor(
        self, q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identical math to TraceableCondStep._apply_rope_tensor (interleaved
        pairs), but every shape is a Python-int CONSTANT (B=1, T=self.t_max,
        H=self.num_heads, D=self.head_dim) so no aten::Int nodes are emitted."""
        B, T, H, D = 1, self.t_max, self.num_heads, self.head_dim
        D_float = float(self.head_dim)
        half_d = self.head_dim // 2

        ds = torch.arange(half_d, device=q.device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(self.rope_max_period) * 2.0 / D_float))

        ts = torch.arange(T, device=q.device, dtype=torch.float32)
        offset_f = offset.float() if offset.dtype != torch.float32 else offset
        ts = ts + offset_f.view(B, 1)
        ts = ts.view(B, T, 1, 1)

        q_complex = q.view(B, T, H, half_d, 2)
        k_complex = k.view(B, T, H, half_d, 2)
        qr = q_complex[..., 0].float()
        qi = q_complex[..., 1].float()
        kr = k_complex[..., 0].float()
        ki = k_complex[..., 1].float()

        rotr = torch.cos(freqs * ts)
        roti = torch.sin(freqs * ts)
        qor = qr * rotr - qi * roti
        qoi = qr * roti + qi * rotr
        kor = kr * rotr - ki * roti
        koi = kr * roti + ki * rotr

        dtype = q.dtype
        qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
        ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
        return qo.view(B, T, H, D), ko.view(B, T, H, D)

    def _streaming_attention(
        self,
        x: torch.Tensor,
        in_proj: nn.Linear,
        out_proj: nn.Linear,
        cache: torch.Tensor,
        position: torch.Tensor,
        valid_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Constants only — no dims unpacked from traced tensor shapes.
        B, T = 1, self.t_max
        H = self.num_heads
        D = self.head_dim
        max_len = self.max_seq_len
        max_len_float = float(max_len)

        pos_float = position.float() if position.dtype != torch.float32 else position
        valid_f = valid_len.float() if valid_len.dtype != torch.float32 else valid_len

        qkv = in_proj(x).reshape(B, T, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k = self._apply_rope_tensor(q, k, pos_float)

        new_cache = cache.clone()
        token_idx = torch.arange(T, device=x.device, dtype=torch.float32)  # [T]
        write_base_float = pos_float.view(B, 1)
        write_indices_float = write_base_float + token_idx.view(1, T)  # [B, T]
        write_indices_float = (
            write_indices_float - torch.floor(write_indices_float / max_len_float) * max_len_float
        )

        # GATE: tokens at index >= valid_len are padding. Redirect their writes
        # to the dump slot (max_len - 1), which the mask below always excludes,
        # so they never overwrite a real conditioning position.
        token_valid = (token_idx.view(1, T) < valid_f.view(B, 1))  # [B, T] bool
        dump_slot = float(max_len - 1)
        write_indices_float = torch.where(
            token_valid, write_indices_float, torch.full_like(write_indices_float, dump_slot)
        )
        write_indices = write_indices_float.long().view(B, T, 1, 1).expand(B, T, H, D)

        new_cache[0] = new_cache[0].scatter(1, write_indices, k)
        new_cache[1] = new_cache[1].scatter(1, write_indices, v)

        keys = new_cache[0]
        values = new_cache[1]
        keys = torch.where(torch.isnan(keys), torch.zeros_like(keys), keys)
        values = torch.where(torch.isnan(values), torch.zeros_like(values), values)

        q = q.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        q_offsets = torch.arange(T, device=x.device, dtype=torch.float32).view(1, T, 1)
        q_positions = pos_float.view(B, 1, 1) + q_offsets
        k_positions = torch.arange(max_len, device=x.device, dtype=torch.float32).view(1, 1, max_len)

        # valid_mask uses the TRUE end (pos + valid_len), not pos + T_max, so the
        # padded tail and the dump slot are excluded from every query's context.
        valid_end = pos_float.view(B, 1, 1) + valid_f.view(B, 1, 1)
        valid_mask = k_positions < valid_end
        causal_mask = k_positions <= q_positions
        attn_mask = (valid_mask & causal_mask).unsqueeze(1)

        scale = 1.0 / (q.shape[-1] ** 0.5)
        attn_weights = torch.matmul(q, keys.transpose(-2, -1)) * scale
        attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        output = out_proj(attn_output)

        # Advance by the TRUE token count, not T_max.
        new_position = pos_float + valid_f.view(B)
        return output, new_cache, new_position

    def forward(
        self,
        conditioning: torch.Tensor,  # [B, T_max, 1024]
        valid_len: torch.Tensor,     # [1] true token count
        *cache_and_positions: torch.Tensor,
    ):
        x = conditioning
        caches = list(cache_and_positions[0::2])
        positions = list(cache_and_positions[1::2])
        new_caches = []
        new_positions = []

        for i in range(self.num_layers):
            residual = x
            x_norm = getattr(self, f"norm{i}_1")(x)
            attn_out, new_cache, new_pos = self._streaming_attention(
                x_norm,
                getattr(self, f"attn{i}_in_proj"),
                getattr(self, f"attn{i}_out_proj"),
                caches[i],
                positions[i],
                valid_len,
            )
            x = residual + attn_out

            residual = x
            x_norm = getattr(self, f"norm{i}_2")(x)
            ffn_out = getattr(self, f"linear{i}_2")(F.gelu(getattr(self, f"linear{i}_1")(x_norm)))
            x = residual + ffn_out

            new_caches.append(new_cache)
            new_positions.append(new_pos)

        outputs = []
        for nc, np_ in zip(new_caches, new_positions):
            outputs.append(nc)
            outputs.append(np_)
        return tuple(outputs)


if __name__ == "__main__":
    from pocket_tts import TTSModel
    from traceable_cond_step import TraceableCondStep

    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()
    max_seq_len = 512
    T_max = 256

    prefill = TraceableCondPrefill.from_flowlm(model.flow_lm, max_seq_len=max_seq_len, t_max=T_max)
    prefill.eval()
    step = TraceableCondStep.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    step.eval()
    nl = prefill.num_layers

    # Parity: prefill of N real tokens (padded to T_max) vs N sequential per-token calls.
    N = 141
    cond_real = torch.randn(1, N, 1024)
    cond_pad = torch.cat([cond_real, torch.randn(1, T_max - N, 1024)], dim=1)  # garbage tail
    valid_len = torch.tensor([float(N)])

    # Sequential reference.
    seq_caches = [torch.full((2, 1, max_seq_len, 16, 64), float("nan")) for _ in range(nl)]
    seq_pos = [torch.zeros(1) for _ in range(nl)]
    with torch.no_grad():
        for ti in range(N):
            args = [cond_real[:, ti : ti + 1, :]]
            for li in range(nl):
                args.append(seq_caches[li])
                args.append(seq_pos[li])
            out = step(*args)
            for li in range(nl):
                seq_caches[li] = out[2 * li]
                seq_pos[li] = out[2 * li + 1]

    # One-shot prefill.
    pf_caches = [torch.full((2, 1, max_seq_len, 16, 64), float("nan")) for _ in range(nl)]
    pf_pos = [torch.zeros(1) for _ in range(nl)]
    with torch.no_grad():
        args = [cond_pad, valid_len]
        for li in range(nl):
            args.append(pf_caches[li])
            args.append(pf_pos[li])
        out = prefill(*args)

    # Compare only the valid prefix [0, N) of each layer's K/V cache.
    max_diff = 0.0
    for li in range(nl):
        ref = seq_caches[li][:, :, :N]
        got = out[2 * li][:, :, :N]
        ref = torch.where(torch.isnan(ref), torch.zeros_like(ref), ref)
        got = torch.where(torch.isnan(got), torch.zeros_like(got), got)
        max_diff = max(max_diff, (ref - got).abs().max().item())
        assert abs(out[2 * li + 1].item() - float(N)) < 1e-3, "position mismatch"
    print(f"prefill vs per-token KV max abs diff (valid prefix): {max_diff:.3e}")
    assert max_diff < 1e-3, f"prefill diverged from per-token path: {max_diff}"
    print("Done! (one-shot prefill matches sequential per-token KV cache)")
