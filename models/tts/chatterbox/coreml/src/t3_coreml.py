"""CoreML-friendly re-implementation of Chatterbox T3 (Llama-520M) with static KV cache.

Two traceable modules sharing the loaded T3 weights, mirroring the
cosyvoice3 `llm_coreml.py` pattern with three Chatterbox-specific deltas:

  1. CFG batch of 2 (row 0 = cond, row 1 = uncond/text-zeroed). The whole
     graph runs B=2; the host combines `cond + w*(cond - uncond)`.
  2. Llama3-scaled RoPE — inv_freq is read straight from the loaded
     model's `rotary_emb.inv_freq` buffer so scaling is always exact.
  3. Alignment-head attention outputs. The multilingual sampler's
     AlignmentStreamAnalyzer consumes softmax rows of heads
     (layer 12, head 15), (layer 13, head 11), (layer 9, head 2) for the
     last query position, cond branch only. Both modules emit
     `align_attn: [3, max_len]` so the analyzer can run host-side.

  * ``T3Prefill`` — inputs ``inputs_embeds: [2, T_pre, 1024]``,
    ``input_len: [1] int32`` (right-padded past input_len).  Produces
    ``logits: [2, 8194]`` at position input_len-1,
    ``align_attn: [3, 2, max_len]`` (query rows input_len-2 and input_len-1 —
    the stock prefill ends with two BOS embeds and the analyzer's first chunk
    consumes both rows), and ``kv_k, kv_v: [30, 2, 16, max_len, 64]``
    populated at 0..input_len-1.

  * ``T3Decode`` — inputs ``inputs_embeds: [2, 1, 1024]``, ``kv_k``, ``kv_v``,
    ``cur_len: [1] int32`` (number of positions already in the cache; the new
    token occupies slot cur_len).  Produces ``logits: [2, 8194]``,
    ``align_attn: [3, max_len]`` and the updated caches.

Weights come from a loaded ``chatterbox.models.t3.T3``; no retraining.
Input embeddings (text/speech token tables + learned positional tables +
conditioning encoder) are applied by the host.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]  # (layer, head) — multilingual analyzer
NEG_INF = -3.0e4  # fp16-safe mask value


def _rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor,
                  attention_scaling: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """positions: [B, T] -> cos, sin each [B, T, D] (D = head_dim)."""
    freqs = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)  # [B, T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)                            # [B, T, D]
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    # q/k: [B, H, T, D], cos/sin: [B, T, D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().float())
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps))


class LlamaMLPReimpl(nn.Module):
    def __init__(self, hf_mlp):
        super().__init__()
        self.gate_proj = hf_mlp.gate_proj
        self.up_proj = hf_mlp.up_proj
        self.down_proj = hf_mlp.down_proj

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _AttnBase(nn.Module):
    def __init__(self, hf_attn, num_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = hf_attn.q_proj
        self.k_proj = hf_attn.k_proj
        self.v_proj = hf_attn.v_proj
        self.o_proj = hf_attn.o_proj
        self.h = num_heads
        self.d = head_dim
        self.scale = head_dim ** -0.5

    def _qkv(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        return q, k, v


class AttnPrefill(_AttnBase):
    def forward(self, x, cos, sin, attn_mask):
        """x: [B, T, C]; attn_mask: [1, 1, T, T] additive causal.

        Returns (out, k, v, probs) — probs: [B, H, T, T] post-softmax.
        """
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        q, k = _apply_rope(q, k, cos, sin)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale + attn_mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(B, T, self.h * self.d)
        return self.o_proj(out), k, v, probs


class AttnDecode(_AttnBase):
    def forward(self, x, cos, sin, k_cache, v_cache, update_mask, attn_mask):
        """x: [B, 1, C]; caches [B, H, M, D]; update_mask [1, 1, M, 1] one-hot
        at cur_len; attn_mask [1, 1, 1, M] additive (0 for keys <= cur_len).

        Returns (out, k_cache_new, v_cache_new, probs) — probs: [B, H, 1, M].
        """
        B = x.shape[0]
        q, k, v = self._qkv(x)
        q, k = _apply_rope(q, k, cos, sin)
        k_cache_new = k_cache * (1.0 - update_mask) + k * update_mask
        v_cache_new = v_cache * (1.0 - update_mask) + v * update_mask
        scores = torch.matmul(q, k_cache_new.transpose(-1, -2)) * self.scale + attn_mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_cache_new)
        out = out.transpose(1, 2).reshape(B, 1, self.h * self.d)
        return self.o_proj(out), k_cache_new, v_cache_new, probs


class _LayerBase(nn.Module):
    def __init__(self, hf_layer, attn_cls, num_heads, head_dim, eps):
        super().__init__()
        self.input_layernorm = RMSNorm(hf_layer.input_layernorm.weight, eps)
        self.post_attention_layernorm = RMSNorm(hf_layer.post_attention_layernorm.weight, eps)
        self.attn = attn_cls(hf_layer.self_attn, num_heads, head_dim)
        self.mlp = LlamaMLPReimpl(hf_layer.mlp)


class LayerPrefill(_LayerBase):
    def __init__(self, hf_layer, num_heads, head_dim, eps):
        super().__init__(hf_layer, AttnPrefill, num_heads, head_dim, eps)

    def forward(self, x, cos, sin, attn_mask):
        h, k, v, probs = self.attn(self.input_layernorm(x), cos, sin, attn_mask)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k, v, probs


class LayerDecode(_LayerBase):
    def __init__(self, hf_layer, num_heads, head_dim, eps):
        super().__init__(hf_layer, AttnDecode, num_heads, head_dim, eps)

    def forward(self, x, cos, sin, k_cache, v_cache, update_mask, attn_mask):
        h, k_new, v_new, probs = self.attn(
            self.input_layernorm(x), cos, sin, k_cache, v_cache, update_mask, attn_mask)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k_new, v_new, probs


def _llama_dims(llama) -> Tuple[int, int, int, float]:
    cfg = llama.config
    heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // heads)
    return cfg.num_hidden_layers, heads, head_dim, cfg.rms_norm_eps


def _inv_freq(llama) -> Tuple[torch.Tensor, float]:
    rot = llama.rotary_emb
    return rot.inv_freq.detach().clone().float(), float(rot.attention_scaling)


class T3Prefill(nn.Module):
    def __init__(self, llama, speech_head: nn.Linear, max_len: int, t_prefill: int):
        super().__init__()
        L, H, D, eps = _llama_dims(llama)
        self.L, self.H, self.D = L, H, D
        self.max_len = max_len
        self.t_prefill = t_prefill
        inv_freq, self.attention_scaling = _inv_freq(llama)
        self.register_buffer("inv_freq", inv_freq)
        self.layers = nn.ModuleList(
            LayerPrefill(l, H, D, eps) for l in llama.layers)
        self.norm = RMSNorm(llama.norm.weight, eps)
        self.speech_head = speech_head

        causal = torch.full((t_prefill, t_prefill), NEG_INF).triu(1)
        self.register_buffer("causal_mask", causal.view(1, 1, t_prefill, t_prefill))
        positions = torch.arange(t_prefill).view(1, -1).expand(2, -1)
        cos, sin = _rope_cos_sin(positions, inv_freq, self.attention_scaling)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, inputs_embeds: torch.Tensor, input_len: torch.Tensor):
        """inputs_embeds: [2, T_pre, C] fp32; input_len: [1] int32.

        Returns logits [2, V], align_attn [3, max_len], kv_k, kv_v
        [L, 2, H, max_len, D].
        """
        x = inputs_embeds
        B, T, C = x.shape
        last = (input_len.to(torch.int64) - 1).view(1)
        bos_rows = torch.cat([last - 1, last])         # queries for both BOS embeds

        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        align_rows: List[torch.Tensor] = []
        aligned = {layer: head for layer, head in ALIGNED_HEADS}
        for idx, layer in enumerate(self.layers):
            x, k, v, probs = layer(x, self.cos, self.sin, self.causal_mask)
            ks.append(k)
            vs.append(v)
            if idx in aligned:
                # cond branch, selected head, query rows input_len-2 / input_len-1
                align_rows.append(probs[0, aligned[idx]].index_select(0, bos_rows))

        x = self.norm(x)
        hidden_last = x.index_select(1, last)          # [2, 1, C]
        logits = self.speech_head(hidden_last)[:, 0]   # [2, V]

        # Rows in ascending layer order; the analyzer means over the three
        # heads, so ordering is contract-only, not numerics.
        rows = torch.stack(align_rows)                 # [3, 2, T]
        align_attn = F.pad(rows, (0, self.max_len - T))

        kv_k = torch.stack([F.pad(k, (0, 0, 0, self.max_len - T)) for k in ks])
        kv_v = torch.stack([F.pad(v, (0, 0, 0, self.max_len - T)) for v in vs])
        return logits, align_attn, kv_k, kv_v


class T3Decode(nn.Module):
    def __init__(self, llama, speech_head: nn.Linear, max_len: int):
        super().__init__()
        L, H, D, eps = _llama_dims(llama)
        self.L, self.H, self.D = L, H, D
        self.max_len = max_len
        inv_freq, self.attention_scaling = _inv_freq(llama)
        self.register_buffer("inv_freq", inv_freq)
        self.layers = nn.ModuleList(
            LayerDecode(l, H, D, eps) for l in llama.layers)
        self.norm = RMSNorm(llama.norm.weight, eps)
        self.speech_head = speech_head
        self.register_buffer("arange_m", torch.arange(max_len, dtype=torch.float32))

    def forward(self, inputs_embeds, kv_k, kv_v, cur_len):
        """inputs_embeds: [2, 1, C]; kv_k/kv_v: [L, 2, H, M, D]; cur_len: [1] i32.

        Returns logits [2, V], align_attn [3, M], kv_k_out, kv_v_out.
        """
        x = inputs_embeds
        cur = cur_len.to(torch.float32).view(1)
        positions = cur_len.to(torch.int64).view(1, 1).expand(2, 1)
        cos, sin = _rope_cos_sin(positions, self.inv_freq, self.attention_scaling)

        update_mask = (self.arange_m == cur).view(1, 1, self.max_len, 1).to(x.dtype)
        key_mask = torch.where(self.arange_m <= cur,
                               torch.zeros_like(self.arange_m),
                               torch.full_like(self.arange_m, NEG_INF))
        attn_mask = key_mask.view(1, 1, 1, self.max_len)

        ks_out: List[torch.Tensor] = []
        vs_out: List[torch.Tensor] = []
        align_rows: List[torch.Tensor] = []
        aligned = {layer: head for layer, head in ALIGNED_HEADS}
        for idx, layer in enumerate(self.layers):
            x, k_new, v_new, probs = layer(
                x, cos, sin, kv_k[idx], kv_v[idx], update_mask, attn_mask)
            ks_out.append(k_new)
            vs_out.append(v_new)
            if idx in aligned:
                align_rows.append(probs[0, aligned[idx], 0].view(1, -1))  # [1, M]

        x = self.norm(x)
        logits = self.speech_head(x)[:, 0]  # [2, V]

        align_attn = torch.cat(align_rows, dim=0)  # [3, M], ascending layer order

        return logits, align_attn, torch.stack(ks_out), torch.stack(vs_out)


class T3DecodeStateful(nn.Module):
    """Stateful single-step decode (macOS 15+ / iOS 18+, MLState KV).

    Per-layer buffers ``kv_k_{i}`` / ``kv_v_{i}``: [2, H, max_len, D], seeded
    from prefill's outputs before the first step, then mutated in place —
    Swift never round-trips the ~250 MB cache through the binding layer.
    Same logits/align_attn contract as ``T3Decode``.
    """

    def __init__(self, llama, speech_head: nn.Linear, max_len: int):
        super().__init__()
        L, H, D, eps = _llama_dims(llama)
        self.L, self.H, self.D = L, H, D
        self.max_len = max_len
        inv_freq, self.attention_scaling = _inv_freq(llama)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.layers = nn.ModuleList(
            LayerDecode(l, H, D, eps) for l in llama.layers)
        self.norm = RMSNorm(llama.norm.weight, eps)
        self.speech_head = speech_head
        self.register_buffer("arange_m", torch.arange(max_len, dtype=torch.float32),
                             persistent=False)
        for i in range(L):
            self.register_buffer(f"kv_k_{i}", torch.zeros(2, H, max_len, D),
                                 persistent=False)
            self.register_buffer(f"kv_v_{i}", torch.zeros(2, H, max_len, D),
                                 persistent=False)

    def forward(self, inputs_embeds, cur_len):
        x = inputs_embeds
        cur = cur_len.to(torch.float32).view(1)
        positions = cur_len.to(torch.int64).view(1, 1).expand(2, 1)
        cos, sin = _rope_cos_sin(positions, self.inv_freq, self.attention_scaling)

        update_mask = (self.arange_m == cur).view(1, 1, self.max_len, 1).to(x.dtype)
        key_mask = torch.where(self.arange_m <= cur,
                               torch.zeros_like(self.arange_m),
                               torch.full_like(self.arange_m, NEG_INF))
        attn_mask = key_mask.view(1, 1, 1, self.max_len)

        align_rows: List[torch.Tensor] = []
        aligned = {layer: head for layer, head in ALIGNED_HEADS}
        for idx, layer in enumerate(self.layers):
            k_i = getattr(self, f"kv_k_{idx}")
            v_i = getattr(self, f"kv_v_{idx}")
            x, k_new, v_new, probs = layer(x, cos, sin, k_i, v_i, update_mask, attn_mask)
            getattr(self, f"kv_k_{idx}")[:] = k_new
            getattr(self, f"kv_v_{idx}")[:] = v_new
            if idx in aligned:
                align_rows.append(probs[0, aligned[idx], 0].view(1, -1))

        x = self.norm(x)
        logits = self.speech_head(x)[:, 0]
        align_attn = torch.cat(align_rows, dim=0)
        return logits, align_attn
