"""CoreML-friendly re-implementation of Chatterbox Nano/Turbo T3 (GPT2 backbone).

Same prefill/decode/stateful split as ``t3_coreml.py`` (Llama MTL), with the
Nano/Turbo deltas:

  * GPT2 blocks (LayerNorm+bias, fused c_attn, gelu_new MLP, learned absolute
    ``wpe`` positions) instead of Llama (RMSNorm, RoPE).
  * No CFG — batch 1 everywhere (``inference_turbo`` never batches uncond).
  * No alignment-head outputs (Turbo/Nano have no AlignmentStreamAnalyzer).
  * ``wpe`` is applied in-graph: prefill adds rows 0..T-1, decode gathers the
    ``cur_len`` row — the host feeds bare cond/text/speech embeddings.
  * ``speech_head`` has a bias (GPT2 branch of T3).

  * ``T3NanoPrefill`` — inputs ``inputs_embeds: [1, T_pre, 768]``,
    ``input_len: [1] int32`` (right-padded past input_len). Produces
    ``logits: [1, 6563]`` at position input_len-1 and ``kv_k, kv_v:
    [12, 1, 12, max_len, 64]`` populated at 0..input_len-1.

  * ``T3NanoDecode`` — inputs ``inputs_embeds: [1, 1, 768]``, ``kv_k``,
    ``kv_v``, ``cur_len: [1] int32`` (the new token occupies slot cur_len).
    Produces ``logits: [1, 6563]`` and the updated caches.

  * ``T3NanoDecodeStateful`` — MLState variant (macOS 15+ / iOS 18+).

Padding safety: garbage K/V rows between input_len and T_pre are never
attended — decode masks keys > cur_len and every slot from input_len up is
overwritten by the step that first exposes it (same scheme as MTL).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -3.0e4  # fp16-safe mask value


def _conv1d_to_linear(conv) -> Tuple[torch.Tensor, torch.Tensor]:
    """HF Conv1D stores weight [in, out]; return ([out, in], [out]) for F.linear."""
    return conv.weight.detach().t().contiguous(), conv.bias.detach().clone()


class GPT2LayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.weight = nn.Parameter(ln.weight.detach().clone().float())
        self.bias = nn.Parameter(ln.bias.detach().clone().float())
        self.eps = ln.eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class GPT2MLPReimpl(nn.Module):
    def __init__(self, hf_mlp):
        super().__init__()
        w_fc, b_fc = _conv1d_to_linear(hf_mlp.c_fc)
        w_pr, b_pr = _conv1d_to_linear(hf_mlp.c_proj)
        self.w_fc = nn.Parameter(w_fc)
        self.b_fc = nn.Parameter(b_fc)
        self.w_pr = nn.Parameter(w_pr)
        self.b_pr = nn.Parameter(b_pr)

    def forward(self, x):
        h = F.linear(x, self.w_fc, self.b_fc)
        # gelu_new (NewGELUActivation) — matches HF exactly
        h = 0.5 * h * (1.0 + torch.tanh(0.7978845608028654 * (h + 0.044715 * h * h * h)))
        return F.linear(h, self.w_pr, self.b_pr)


class _AttnBase(nn.Module):
    def __init__(self, hf_attn, num_heads: int, head_dim: int):
        super().__init__()
        w_qkv, b_qkv = _conv1d_to_linear(hf_attn.c_attn)   # [3C, C], [3C]
        w_o, b_o = _conv1d_to_linear(hf_attn.c_proj)
        self.w_qkv = nn.Parameter(w_qkv)
        self.b_qkv = nn.Parameter(b_qkv)
        self.w_o = nn.Parameter(w_o)
        self.b_o = nn.Parameter(b_o)
        self.h = num_heads
        self.d = head_dim
        self.scale = head_dim ** -0.5

    def _qkv(self, x):
        B, T, C = x.shape
        qkv = F.linear(x, self.w_qkv, self.b_qkv)          # [B, T, 3C]
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)
        return q, k, v

    def _proj(self, out, B, T):
        out = out.transpose(1, 2).reshape(B, T, self.h * self.d)
        return F.linear(out, self.w_o, self.b_o)


class AttnPrefill(_AttnBase):
    def forward(self, x, attn_mask):
        """x: [B, T, C]; attn_mask: [1, 1, T, T] additive causal."""
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale + attn_mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return self._proj(out, B, T), k, v


class AttnDecode(_AttnBase):
    def forward(self, x, k_cache, v_cache, update_mask, attn_mask):
        """x: [B, 1, C]; caches [B, H, M, D]; update_mask [1, 1, M, 1] one-hot
        at cur_len; attn_mask [1, 1, 1, M] additive (0 for keys <= cur_len)."""
        B = x.shape[0]
        q, k, v = self._qkv(x)
        k_cache_new = k_cache * (1.0 - update_mask) + k * update_mask
        v_cache_new = v_cache * (1.0 - update_mask) + v * update_mask
        scores = torch.matmul(q, k_cache_new.transpose(-1, -2)) * self.scale + attn_mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_cache_new)
        return self._proj(out, B, 1), k_cache_new, v_cache_new


class _LayerBase(nn.Module):
    def __init__(self, hf_block, attn_cls, num_heads, head_dim):
        super().__init__()
        self.ln_1 = GPT2LayerNorm(hf_block.ln_1)
        self.ln_2 = GPT2LayerNorm(hf_block.ln_2)
        self.attn = attn_cls(hf_block.attn, num_heads, head_dim)
        self.mlp = GPT2MLPReimpl(hf_block.mlp)


class LayerPrefill(_LayerBase):
    def __init__(self, hf_block, num_heads, head_dim):
        super().__init__(hf_block, AttnPrefill, num_heads, head_dim)

    def forward(self, x, attn_mask):
        h, k, v = self.attn(self.ln_1(x), attn_mask)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, k, v


class LayerDecode(_LayerBase):
    def __init__(self, hf_block, num_heads, head_dim):
        super().__init__(hf_block, AttnDecode, num_heads, head_dim)

    def forward(self, x, k_cache, v_cache, update_mask, attn_mask):
        h, k_new, v_new = self.attn(self.ln_1(x), k_cache, v_cache, update_mask, attn_mask)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, k_new, v_new


def _gpt2_dims(gpt2) -> Tuple[int, int, int]:
    cfg = gpt2.config
    heads = cfg.num_attention_heads
    return cfg.num_hidden_layers, heads, cfg.hidden_size // heads


class T3NanoPrefill(nn.Module):
    def __init__(self, gpt2, speech_head: nn.Linear, max_len: int, t_prefill: int):
        super().__init__()
        L, H, D = _gpt2_dims(gpt2)
        self.L, self.H, self.D = L, H, D
        self.max_len = max_len
        self.t_prefill = t_prefill
        self.layers = nn.ModuleList(LayerPrefill(b, H, D) for b in gpt2.h)
        self.ln_f = GPT2LayerNorm(gpt2.ln_f)
        self.speech_head = speech_head
        self.register_buffer(
            "wpe", gpt2.wpe.weight.detach()[:t_prefill].clone().float())
        causal = torch.full((t_prefill, t_prefill), NEG_INF).triu(1)
        self.register_buffer("causal_mask", causal.view(1, 1, t_prefill, t_prefill))

    def forward(self, inputs_embeds: torch.Tensor, input_len: torch.Tensor):
        """inputs_embeds: [1, T_pre, C] fp32; input_len: [1] int32.

        Returns logits [1, V], kv_k, kv_v [L, 1, H, max_len, D].
        """
        x = inputs_embeds + self.wpe.unsqueeze(0)
        B, T, C = x.shape
        last = (input_len.to(torch.int64) - 1).view(1)

        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for layer in self.layers:
            x, k, v = layer(x, self.causal_mask)
            ks.append(k)
            vs.append(v)

        x = self.ln_f(x)
        hidden_last = x.index_select(1, last)          # [1, 1, C]
        logits = self.speech_head(hidden_last)[:, 0]   # [1, V]

        kv_k = torch.stack([F.pad(k, (0, 0, 0, self.max_len - T)) for k in ks])
        kv_v = torch.stack([F.pad(v, (0, 0, 0, self.max_len - T)) for v in vs])
        return logits, kv_k, kv_v


class T3NanoDecode(nn.Module):
    def __init__(self, gpt2, speech_head: nn.Linear, max_len: int):
        super().__init__()
        L, H, D = _gpt2_dims(gpt2)
        self.L, self.H, self.D = L, H, D
        self.max_len = max_len
        self.layers = nn.ModuleList(LayerDecode(b, H, D) for b in gpt2.h)
        self.ln_f = GPT2LayerNorm(gpt2.ln_f)
        self.speech_head = speech_head
        self.register_buffer("wpe", gpt2.wpe.weight.detach().clone().float())
        self.register_buffer("arange_m", torch.arange(max_len, dtype=torch.float32))

    def _masks(self, cur_len, dtype):
        cur = cur_len.to(torch.float32).view(1)
        update_mask = (self.arange_m == cur).view(1, 1, self.max_len, 1).to(dtype)
        key_mask = torch.where(self.arange_m <= cur,
                               torch.zeros_like(self.arange_m),
                               torch.full_like(self.arange_m, NEG_INF))
        return update_mask, key_mask.view(1, 1, 1, self.max_len)

    def forward(self, inputs_embeds, kv_k, kv_v, cur_len):
        """inputs_embeds: [1, 1, C]; kv_k/kv_v: [L, 1, H, M, D]; cur_len: [1] i32.

        Returns logits [1, V], kv_k_out, kv_v_out.
        """
        pos = cur_len.to(torch.int64).view(1)
        x = inputs_embeds + self.wpe.index_select(0, pos).unsqueeze(0)
        update_mask, attn_mask = self._masks(cur_len, x.dtype)

        ks_out: List[torch.Tensor] = []
        vs_out: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            x, k_new, v_new = layer(x, kv_k[idx], kv_v[idx], update_mask, attn_mask)
            ks_out.append(k_new)
            vs_out.append(v_new)

        x = self.ln_f(x)
        logits = self.speech_head(x)[:, 0]  # [1, V]
        return logits, torch.stack(ks_out), torch.stack(vs_out)


class T3NanoDecodeStateful(nn.Module):
    """Stateful single-step decode (macOS 15+ / iOS 18+, MLState KV).

    Per-layer buffers ``kv_k_{i}`` / ``kv_v_{i}``: [1, H, max_len, D], seeded
    from prefill's outputs before the first step, then mutated in place.
    Same logits contract as ``T3NanoDecode``.
    """

    def __init__(self, gpt2, speech_head: nn.Linear, max_len: int):
        super().__init__()
        L, H, D = _gpt2_dims(gpt2)
        self.L, self.H, self.D = L, H, D
        self.max_len = max_len
        self.layers = nn.ModuleList(LayerDecode(b, H, D) for b in gpt2.h)
        self.ln_f = GPT2LayerNorm(gpt2.ln_f)
        self.speech_head = speech_head
        self.register_buffer("wpe", gpt2.wpe.weight.detach().clone().float(),
                             persistent=False)
        self.register_buffer("arange_m", torch.arange(max_len, dtype=torch.float32),
                             persistent=False)
        for i in range(L):
            self.register_buffer(f"kv_k_{i}", torch.zeros(1, H, max_len, D),
                                 persistent=False)
            self.register_buffer(f"kv_v_{i}", torch.zeros(1, H, max_len, D),
                                 persistent=False)

    def forward(self, inputs_embeds, cur_len):
        pos = cur_len.to(torch.int64).view(1)
        x = inputs_embeds + self.wpe.index_select(0, pos).unsqueeze(0)
        cur = cur_len.to(torch.float32).view(1)
        update_mask = (self.arange_m == cur).view(1, 1, self.max_len, 1).to(x.dtype)
        key_mask = torch.where(self.arange_m <= cur,
                               torch.zeros_like(self.arange_m),
                               torch.full_like(self.arange_m, NEG_INF))
        attn_mask = key_mask.view(1, 1, 1, self.max_len)

        for idx, layer in enumerate(self.layers):
            k_i = getattr(self, f"kv_k_{idx}")
            v_i = getattr(self, f"kv_v_{idx}")
            x, k_new, v_new = layer(x, k_i, v_i, update_mask, attn_mask)
            getattr(self, f"kv_k_{idx}")[:] = k_new
            getattr(self, f"kv_v_{idx}")[:] = v_new

        x = self.ln_f(x)
        logits = self.speech_head(x)[:, 0]
        return logits
