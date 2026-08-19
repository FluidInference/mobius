"""CoreML-friendly re-implementation of the NeuTTS-2E Qwen3 backbone.

Same design as ``models/tts/cosyvoice3/coreml/src/llm_coreml.py`` (static KV
cache, three wrappers) with the Qwen3 differences:

  * per-head RMSNorm on q/k (``q_norm`` / ``k_norm``) before RoPE
  * ``head_dim`` decoupled from ``hidden_size`` (128 vs 512/12)
  * tied embeddings: one shared [V, C] weight used for both the input
    gather and the LM head, so CoreML const-dedup stores it once per model

Wrappers:

  * ``Qwen3Prefill`` — input_ids [1, T_pre] + input_len [1] →
      logits_last [1, V] (logits at position input_len-1),
      kv_k / kv_v [L, 1, Hkv, max_len, D] (positions [0, input_len) filled)
  * ``Qwen3Decode`` — input_ids [1, 1] + kv_k + kv_v + cur_len [1] →
      logits [1, V], kv_k_out, kv_v_out
  * ``Qwen3DecodeStateful`` — input_ids [1, 1] + cur_len [1] → logits [1, V];
      KV lives in per-layer StateType buffers (macOS 15+ / iOS 18+),
      seeded from prefill's kv_k / kv_v output by the host.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_rope_inv_freq(head_dim: int, rope_theta: float) -> torch.Tensor:
    return 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )


def _resolve_rope_theta(cfg) -> float:
    direct = getattr(cfg, "rope_theta", None)
    if direct is not None:
        return float(direct)
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict):
        for key in ("rope_theta", "base"):
            if key in params and params[key] is not None:
                return float(params[key])
    raise AttributeError("Config exposes neither rope_theta nor rope_parameters")


def _rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)  # [B, T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, T, D]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor, half: int) -> torch.Tensor:
    # `half` must be a Python int: x.shape[-1] // 2 traces into
    # aten::floor_divide/Int nodes the CoreML frontend cannot fold.
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin, half: int):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q, half) * sin)
    k_out = (k * cos) + (_rotate_half(k, half) * sin)
    return q_out, k_out


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().to(torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(var + self.eps)
        return self.weight * x


class _LinearLike(nn.Module):
    def __init__(self, hf_lin: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(hf_lin.weight.detach().clone().to(torch.float32))
        if hf_lin.bias is not None:
            self.bias = nn.Parameter(hf_lin.bias.detach().clone().to(torch.float32))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class Qwen3MLP(nn.Module):
    def __init__(self, hf_mlp: nn.Module):
        super().__init__()
        self.gate_proj = _LinearLike(hf_mlp.gate_proj)
        self.up_proj = _LinearLike(hf_mlp.up_proj)
        self.down_proj = _LinearLike(hf_mlp.down_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3AttnPrefill(nn.Module):
    def __init__(
        self, hf_attn: nn.Module, num_heads: int, num_kv_heads: int, head_dim: int, seq_len: int
    ):
        super().__init__()
        self.q_proj = _LinearLike(hf_attn.q_proj)
        self.k_proj = _LinearLike(hf_attn.k_proj)
        self.v_proj = _LinearLike(hf_attn.v_proj)
        self.o_proj = _LinearLike(hf_attn.o_proj)
        self.q_norm = RMSNorm(hf_attn.q_norm.weight, eps=hf_attn.q_norm.variance_epsilon)
        self.k_norm = RMSNorm(hf_attn.k_norm.weight, eps=hf_attn.k_norm.variance_epsilon)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.rep = num_heads // num_kv_heads
        self.scale = head_dim**-0.5

    def forward(self, x, cos, sin, attn_mask):
        # Static shapes (torch>=2.10 traces x.shape unpacking into aten::Int
        # nodes that coremltools' frontend cannot fold).
        B, T = 1, self.seq_len
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        # Qwen3: per-head RMSNorm on q/k before RoPE.
        q = self.q_norm(self.q_proj(x).view(B, T, H, D)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, Hkv, D)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin, D // 2)

        # GQA head expansion via expand+reshape (repeat_interleave traces into
        # aten::floor_divide/Int nodes the CoreML frontend cannot fold).
        k_rep = k.unsqueeze(2).expand(B, Hkv, self.rep, T, D).reshape(B, H, T, D)
        v_rep = v.unsqueeze(2).expand(B, Hkv, self.rep, T, D).reshape(B, H, T, D)

        attn = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_rep)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out), k, v


class Qwen3AttnDecode(nn.Module):
    def __init__(
        self, hf_attn: nn.Module, num_heads: int, num_kv_heads: int, head_dim: int, max_len: int
    ):
        super().__init__()
        self.q_proj = _LinearLike(hf_attn.q_proj)
        self.k_proj = _LinearLike(hf_attn.k_proj)
        self.v_proj = _LinearLike(hf_attn.v_proj)
        self.o_proj = _LinearLike(hf_attn.o_proj)
        self.q_norm = RMSNorm(hf_attn.q_norm.weight, eps=hf_attn.q_norm.variance_epsilon)
        self.k_norm = RMSNorm(hf_attn.k_norm.weight, eps=hf_attn.k_norm.variance_epsilon)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.rep = num_heads // num_kv_heads
        self.scale = head_dim**-0.5

    def forward(self, x, cos, sin, k_cache, v_cache, update_mask, attn_mask):
        B, T = 1, 1  # static: single-token decode step
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_norm(self.q_proj(x).view(B, T, H, D)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, Hkv, D)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin, D // 2)

        # Scatter (k, v) into the cache at position cur_len via one-hot mask.
        k_full = k_cache * (1.0 - update_mask) + k * update_mask
        v_full = v_cache * (1.0 - update_mask) + v * update_mask

        M = self.max_len
        k_rep = k_full.unsqueeze(2).expand(B, Hkv, self.rep, M, D).reshape(B, H, M, D)
        v_rep = v_full.unsqueeze(2).expand(B, Hkv, self.rep, M, D).reshape(B, H, M, D)

        attn = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_rep)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out), k_full, v_full


class Qwen3LayerPrefill(nn.Module):
    def __init__(self, hf_layer, num_heads, num_kv_heads, head_dim, seq_len):
        super().__init__()
        self.input_layernorm = RMSNorm(hf_layer.input_layernorm.weight)
        self.post_attention_layernorm = RMSNorm(hf_layer.post_attention_layernorm.weight)
        self.self_attn = Qwen3AttnPrefill(hf_layer.self_attn, num_heads, num_kv_heads, head_dim, seq_len)
        self.mlp = Qwen3MLP(hf_layer.mlp)

    def forward(self, x, cos, sin, attn_mask):
        h = self.input_layernorm(x)
        a, k_new, v_new = self.self_attn(h, cos, sin, attn_mask)
        x = x + a
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x, k_new, v_new


class Qwen3LayerDecode(nn.Module):
    def __init__(self, hf_layer, num_heads, num_kv_heads, head_dim, max_len):
        super().__init__()
        self.input_layernorm = RMSNorm(hf_layer.input_layernorm.weight)
        self.post_attention_layernorm = RMSNorm(hf_layer.post_attention_layernorm.weight)
        self.self_attn = Qwen3AttnDecode(hf_layer.self_attn, num_heads, num_kv_heads, head_dim, max_len)
        self.mlp = Qwen3MLP(hf_layer.mlp)

    def forward(self, x, cos, sin, k_cache, v_cache, update_mask, attn_mask):
        h = self.input_layernorm(x)
        a, k_full, v_full = self.self_attn(h, cos, sin, k_cache, v_cache, update_mask, attn_mask)
        x = x + a
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x, k_full, v_full


class _Qwen3Base(nn.Module):
    def _init_common(self, qwen_for_causal_lm: nn.Module, max_len: int):
        qw = qwen_for_causal_lm
        cfg = qw.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.rope_theta = _resolve_rope_theta(cfg)
        self.max_len = max_len

        self.register_buffer(
            "inv_freq", _build_rope_inv_freq(self.head_dim, self.rope_theta), persistent=False
        )
        self.register_buffer(
            "pos_ids", torch.arange(max_len, dtype=torch.int32), persistent=False
        )

        # Tied embedding / LM head weight [V, C]. Used twice (gather + linear);
        # CoreML const-dedup collapses it to a single stored constant.
        self.tok_weight = nn.Parameter(
            qw.model.embed_tokens.weight.detach().clone().to(torch.float32)
        )
        self.norm = RMSNorm(qw.model.norm.weight)
        return qw

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids.to(torch.long), self.tok_weight)

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.tok_weight)


class Qwen3Prefill(_Qwen3Base):
    """Static-shape prefill over token ids.

    Inputs:
        input_ids: [1, T_pre] int32 (right-padded past input_len; pad value irrelevant)
        input_len: [1] int32 — number of valid tokens
    Outputs:
        logits_last: [1, V] — logits at position input_len-1
        kv_k, kv_v:  [L, 1, Hkv, max_len, D] — positions [0, input_len) filled
    """

    def __init__(self, qwen_for_causal_lm: nn.Module, max_len: int, t_prefill: int):
        super().__init__()
        qw = self._init_common(qwen_for_causal_lm, max_len)
        self.t_prefill = t_prefill
        self.layers = nn.ModuleList(
            [
                Qwen3LayerPrefill(
                    qw.model.layers[i], self.num_heads, self.num_kv_heads, self.head_dim, t_prefill
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, input_ids: torch.Tensor, input_len: torch.Tensor):
        B, T = 1, self.t_prefill
        Hkv, D, M = self.num_kv_heads, self.head_dim, self.max_len

        positions = torch.arange(T, dtype=torch.int32).view(1, T)
        cos, sin = _rope_cos_sin(positions, self.inv_freq)

        idx = torch.arange(T, dtype=torch.int32).view(1, 1, T, 1)
        jdx = torch.arange(T, dtype=torch.int32).view(1, 1, 1, T)
        causal = jdx <= idx
        valid_key = jdx < input_len.view(1, 1, 1, 1).to(torch.int32)
        attendable = causal & valid_key
        neg_inf = torch.tensor(-1e4, dtype=torch.float32)  # fp16-safe
        attn_mask = torch.where(attendable, torch.zeros((), dtype=torch.float32), neg_inf)

        x = self._embed(input_ids)
        pad = M - T
        zero_pad = torch.zeros(B, Hkv, pad, D, dtype=x.dtype)
        k_all: List[torch.Tensor] = []
        v_all: List[torch.Tensor] = []
        for layer in self.layers:
            x, k_new, v_new = layer(x, cos, sin, attn_mask)
            k_all.append(torch.cat([k_new, zero_pad], dim=2))
            v_all.append(torch.cat([v_new, zero_pad], dim=2))

        x = self.norm(x)

        # Gather hidden state at the last valid position, then a single-row LM head.
        last_idx = (input_len.view(1) - 1).to(torch.long)
        h_last = x.index_select(1, last_idx)  # [1, 1, C]
        logits_last = self._logits(h_last).squeeze(1)  # [1, V]

        kv_k = torch.stack(k_all, dim=0)
        kv_v = torch.stack(v_all, dim=0)
        return logits_last, kv_k, kv_v


class Qwen3Decode(_Qwen3Base):
    """Single-step decode with pass-through KV cache (runs on macOS 14+)."""

    def __init__(self, qwen_for_causal_lm: nn.Module, max_len: int):
        super().__init__()
        qw = self._init_common(qwen_for_causal_lm, max_len)
        self.layers = nn.ModuleList(
            [
                Qwen3LayerDecode(
                    qw.model.layers[i], self.num_heads, self.num_kv_heads, self.head_dim, max_len
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, input_ids, kv_k, kv_v, cur_len):
        M = self.max_len
        cur = cur_len.view(1).to(torch.int32)

        positions = cur.view(1, 1)
        cos, sin = _rope_cos_sin(positions, self.inv_freq)

        pj = self.pos_ids.view(1, 1, M, 1)
        update_mask = (pj == cur.view(1, 1, 1, 1)).to(torch.float32)
        attendable = self.pos_ids.view(1, 1, 1, M) <= cur.view(1, 1, 1, 1)
        neg_inf = torch.tensor(-1e4, dtype=torch.float32)
        attn_mask = torch.where(attendable, torch.zeros((), dtype=torch.float32), neg_inf)

        x = self._embed(input_ids)
        k_all: List[torch.Tensor] = []
        v_all: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x, k_full, v_full = layer(x, cos, sin, kv_k[i], kv_v[i], update_mask, attn_mask)
            k_all.append(k_full)
            v_all.append(v_full)

        x = self.norm(x)
        logits = self._logits(x).squeeze(1)  # [1, V]
        return logits, torch.stack(k_all, dim=0), torch.stack(v_all, dim=0)


class Qwen3DecodeStateful(_Qwen3Base):
    """Single-step decode with in-place StateType KV cache (macOS 15+ / iOS 18+).

    Per-layer buffers keep each state's read/write symmetric — the pattern
    coremltools' stateful pass lowers cleanly (see cosyvoice3 notes).
    """

    def __init__(self, qwen_for_causal_lm: nn.Module, max_len: int):
        super().__init__()
        qw = self._init_common(qwen_for_causal_lm, max_len)
        Hkv, D = self.num_kv_heads, self.head_dim
        for i in range(self.num_layers):
            self.register_buffer(
                f"kv_k_{i}", torch.zeros(1, Hkv, max_len, D, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                f"kv_v_{i}", torch.zeros(1, Hkv, max_len, D, dtype=torch.float32), persistent=False
            )
        self.layers = nn.ModuleList(
            [
                Qwen3LayerDecode(
                    qw.model.layers[i], self.num_heads, self.num_kv_heads, self.head_dim, max_len
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, input_ids, cur_len):
        M = self.max_len
        cur = cur_len.view(1).to(torch.int32)

        positions = cur.view(1, 1)
        cos, sin = _rope_cos_sin(positions, self.inv_freq)

        pj = self.pos_ids.view(1, 1, M, 1)
        update_mask = (pj == cur.view(1, 1, 1, 1)).to(torch.float32)
        attendable = self.pos_ids.view(1, 1, 1, M) <= cur.view(1, 1, 1, 1)
        neg_inf = torch.tensor(-1e4, dtype=torch.float32)
        attn_mask = torch.where(attendable, torch.zeros((), dtype=torch.float32), neg_inf)

        x = self._embed(input_ids)
        for i, layer in enumerate(self.layers):
            k_i = getattr(self, f"kv_k_{i}")
            v_i = getattr(self, f"kv_v_{i}")
            x, k_full, v_full = layer(x, cos, sin, k_i, v_i, update_mask, attn_mask)
            # [:] write pattern → coremltools generate_tensor_assignment_ops
            # lowers each to an independent StateType update.
            getattr(self, f"kv_k_{i}")[:] = k_full
            getattr(self, f"kv_v_{i}")[:] = v_full

        x = self.norm(x)
        return self._logits(x).squeeze(1)  # [1, V]
