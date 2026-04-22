"""CoreML-friendly re-implementation of CosyVoice3's Qwen2 LLM with static KV cache.

Design: two separate traceable modules sharing the same loaded Qwen2 weights.

  * ``Qwen2Prefill``  — inputs ``inputs_embeds: [1, T_pre, 896]`` and
    ``input_len: [1] int32``.  Produces ``last_hidden: [1, T_pre, 896]``,
    ``kv_k: [L, 1, Hkv, max_len, D]`` and ``kv_v: [L, 1, Hkv, max_len, D]``
    with positions 0..input_len-1 populated, plus ``speech_logits: [1, T_pre, 6761]``
    (logits at every prefill position — Swift gathers the last valid one).

  * ``Qwen2Decode`` — inputs ``inputs_embeds: [1, 1, 896]``, ``kv_k``,
    ``kv_v`` (same layout as above), and ``cur_len: [1] int32``.  Produces
    ``speech_logits: [1, 6761]`` for the single new position and returns
    the KV cache with position ``cur_len`` populated.

Both modules re-implement Qwen2's forward path manually so the graph is fully
static and traceable.  They read weights from a loaded ``Qwen2ForCausalLM`` +
``nn.Linear`` (speech LM head), so no re-training is required.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  RoPE helpers
# --------------------------------------------------------------------------- #

def _build_rope_inv_freq(head_dim: int, rope_theta: float) -> torch.Tensor:
    """Compute the inverse-frequency vector used by Qwen2 RoPE, shape [D/2]."""
    return 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )


def _rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin for the given integer ``positions``.

    Args:
        positions: [B, T] int/long tensor of absolute positions.
        inv_freq:  [D/2] float32.

    Returns:
        cos, sin — each [B, T, D] float32 (D = head_dim).
    """
    freqs = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)  # [B, T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)                             # [B, T, D]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q: [B, Hq, T, D], k: [B, Hkv, T, D]
    # cos/sin: [B, T, D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


# --------------------------------------------------------------------------- #
#  Building blocks (re-use HF weights, new forward)
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    """Qwen2RMSNorm clone that reuses an existing weight tensor."""
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().to(torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(var + self.eps)
        return self.weight * x


class _LinearLike(nn.Module):
    """Wraps an existing nn.Linear's weight+bias as FP32 params (no re-init)."""
    def __init__(self, hf_lin: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(hf_lin.weight.detach().clone().to(torch.float32))
        if hf_lin.bias is not None:
            self.bias = nn.Parameter(hf_lin.bias.detach().clone().to(torch.float32))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class Qwen2MLPReimpl(nn.Module):
    def __init__(self, hf_mlp: nn.Module):
        super().__init__()
        self.gate_proj = _LinearLike(hf_mlp.gate_proj)
        self.up_proj   = _LinearLike(hf_mlp.up_proj)
        self.down_proj = _LinearLike(hf_mlp.down_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
#  Attention — prefill (T tokens, writes cache positions 0..T-1)
# --------------------------------------------------------------------------- #

class Qwen2AttnPrefill(nn.Module):
    def __init__(self, hf_attn: nn.Module, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = _LinearLike(hf_attn.q_proj)
        self.k_proj = _LinearLike(hf_attn.k_proj)
        self.v_proj = _LinearLike(hf_attn.v_proj)
        self.o_proj = _LinearLike(hf_attn.o_proj)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rep = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self,
                x: torch.Tensor,          # [1, T, C]
                cos: torch.Tensor,        # [1, T, D]
                sin: torch.Tensor,        # [1, T, D]
                attn_mask: torch.Tensor,  # [1, 1, T, T] additive
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H,   D).transpose(1, 2)   # [B, H,   T, D]
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)   # [B, Hkv, T, D]
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)

        k_rep = k.repeat_interleave(self.rep, dim=1)            # [B, H, T, D]
        v_rep = v.repeat_interleave(self.rep, dim=1)

        attn = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, v_rep)                        # [B, H, T, D]
        out  = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out  = self.o_proj(out)
        # k, v are returned un-padded; the caller scatters them into the static cache.
        return out, k, v


# --------------------------------------------------------------------------- #
#  Attention — decode (1 new token, reads + updates cache of length max_len)
# --------------------------------------------------------------------------- #

class Qwen2AttnDecode(nn.Module):
    def __init__(self, hf_attn: nn.Module, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = _LinearLike(hf_attn.q_proj)
        self.k_proj = _LinearLike(hf_attn.k_proj)
        self.v_proj = _LinearLike(hf_attn.v_proj)
        self.o_proj = _LinearLike(hf_attn.o_proj)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rep = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self,
                x: torch.Tensor,          # [1, 1, C]
                cos: torch.Tensor,        # [1, 1, D]
                sin: torch.Tensor,        # [1, 1, D]
                k_cache: torch.Tensor,    # [1, Hkv, max_len, D]
                v_cache: torch.Tensor,    # [1, Hkv, max_len, D]
                update_mask: torch.Tensor,  # [1, 1, max_len, 1] 1 at cur_len else 0
                attn_mask: torch.Tensor,  # [1, 1, 1, max_len] additive (0 valid, -inf invalid)
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape  # T == 1
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H,   D).transpose(1, 2)   # [B, H,   1, D]
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)   # [B, Hkv, 1, D]
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)

        # Scatter (k, v) into the cache at position cur_len using a one-hot update_mask.
        # update_mask: [1, 1, max_len, 1]  (Hkv broadcasts)
        # k_new_broadcast: [1, Hkv, max_len, D]  obtained via mul with update_mask
        k_new = k * 1.0  # keep [1, Hkv, 1, D]
        v_new = v * 1.0
        # scatter by: cache * (1 - mask) + new * mask  (broadcasting new over the max_len axis)
        k_full = k_cache * (1.0 - update_mask) + k_new * update_mask
        v_full = v_cache * (1.0 - update_mask) + v_new * update_mask

        k_rep = k_full.repeat_interleave(self.rep, dim=1)       # [B, H, max_len, D]
        v_rep = v_full.repeat_interleave(self.rep, dim=1)

        attn = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale  # [B, H, 1, max_len]
        attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, v_rep)                        # [B, H, 1, D]
        out  = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out  = self.o_proj(out)
        return out, k_full, v_full


# --------------------------------------------------------------------------- #
#  Decoder layer
# --------------------------------------------------------------------------- #

class Qwen2LayerPrefill(nn.Module):
    def __init__(self, hf_layer: nn.Module, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hf_layer.input_layernorm.weight)
        self.post_attention_layernorm = RMSNorm(hf_layer.post_attention_layernorm.weight)
        self.self_attn = Qwen2AttnPrefill(hf_layer.self_attn, num_heads, num_kv_heads, head_dim)
        self.mlp = Qwen2MLPReimpl(hf_layer.mlp)

    def forward(self, x, cos, sin, attn_mask):
        h = self.input_layernorm(x)
        a, k_new, v_new = self.self_attn(h, cos, sin, attn_mask)
        x = x + a
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x, k_new, v_new


class Qwen2LayerDecode(nn.Module):
    def __init__(self, hf_layer: nn.Module, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hf_layer.input_layernorm.weight)
        self.post_attention_layernorm = RMSNorm(hf_layer.post_attention_layernorm.weight)
        self.self_attn = Qwen2AttnDecode(hf_layer.self_attn, num_heads, num_kv_heads, head_dim)
        self.mlp = Qwen2MLPReimpl(hf_layer.mlp)

    def forward(self, x, cos, sin, k_cache, v_cache, update_mask, attn_mask):
        h = self.input_layernorm(x)
        a, k_full, v_full = self.self_attn(h, cos, sin, k_cache, v_cache, update_mask, attn_mask)
        x = x + a
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x, k_full, v_full


# --------------------------------------------------------------------------- #
#  Top-level wrappers
# --------------------------------------------------------------------------- #

class Qwen2Prefill(nn.Module):
    """Static-shape prefill: consumes inputs_embeds [1, T_pre, 896] + input_len.

    Returns:
        last_hidden: [1, T_pre, 896] — hidden state at every prefill position
                     (Swift gathers the one at input_len-1 to seed decode)
        speech_logits: [1, T_pre, speech_vocab] — logits via speech-LM head
        kv_k: [L, 1, Hkv, max_len, D], kv_v: [L, 1, Hkv, max_len, D]
              positions [0, input_len) populated, rest zero
    """
    def __init__(self,
                 qwen_for_causal_lm: nn.Module,   # Qwen2ForCausalLM
                 speech_lm_head: nn.Linear,        # CosyVoice3LM.llm_decoder
                 max_len: int,
                 t_prefill: int):
        super().__init__()
        qw = qwen_for_causal_lm
        cfg = qw.config
        self.num_layers   = cfg.num_hidden_layers
        self.num_heads    = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim     = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size  = cfg.hidden_size
        self.rope_theta   = cfg.rope_parameters["rope_theta"] if hasattr(cfg, "rope_parameters") else cfg.rope_theta
        self.max_len      = max_len
        self.t_prefill    = t_prefill

        # RoPE inv_freq as buffer
        self.register_buffer(
            "inv_freq",
            _build_rope_inv_freq(self.head_dim, self.rope_theta),
            persistent=False,
        )

        self.layers = nn.ModuleList([
            Qwen2LayerPrefill(qw.model.layers[i],
                              self.num_heads, self.num_kv_heads, self.head_dim)
            for i in range(self.num_layers)
        ])
        self.norm = RMSNorm(qw.model.norm.weight)
        self.speech_lm_head = _LinearLike(speech_lm_head)

    def forward(self,
                inputs_embeds: torch.Tensor,  # [1, T_pre, 896]
                input_len: torch.Tensor,       # [1] int32 (# of valid tokens)
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = 1
        T = self.t_prefill
        Hkv, D, L, M = self.num_kv_heads, self.head_dim, self.num_layers, self.max_len

        # Position ids 0..T-1
        positions = torch.arange(T, dtype=torch.int32).view(1, T)
        cos, sin = _rope_cos_sin(positions, self.inv_freq)

        # Build causal + padding mask.
        idx = torch.arange(T, dtype=torch.int32).view(1, 1, T, 1)  # query index
        jdx = torch.arange(T, dtype=torch.int32).view(1, 1, 1, T)  # key  index
        causal = (jdx <= idx)                                      # [1,1,T,T] bool
        valid_key = jdx < input_len.view(1, 1, 1, 1).to(torch.int32)
        attendable = causal & valid_key
        neg_inf = torch.tensor(-1e4, dtype=torch.float32)          # fp16-safe
        attn_mask = torch.where(attendable,
                                torch.zeros((), dtype=torch.float32),
                                neg_inf).to(inputs_embeds.dtype)

        x = inputs_embeds
        # Collect per-layer [1, Hkv, T_pre, D] k/v — Swift will scatter into the
        # static cache [1, Hkv, max_len, D] by zero-filling beyond input_len.
        # We pre-pad here so shapes are known at trace time.
        pad = M - T
        k_all: List[torch.Tensor] = []
        v_all: List[torch.Tensor] = []
        zero_pad = torch.zeros(B, Hkv, pad, D, dtype=inputs_embeds.dtype)

        for layer in self.layers:
            x, k_new, v_new = layer(x, cos, sin, attn_mask)
            # k_new/v_new: [1, Hkv, T_pre, D]. Concatenate zero-pad to reach max_len.
            k_all.append(torch.cat([k_new, zero_pad], dim=2))
            v_all.append(torch.cat([v_new, zero_pad], dim=2))

        x = self.norm(x)
        last_hidden = x
        speech_logits = self.speech_lm_head(x)

        kv_k = torch.stack(k_all, dim=0)  # [L, 1, Hkv, M, D]
        kv_v = torch.stack(v_all, dim=0)

        return last_hidden, speech_logits, kv_k, kv_v


class Qwen2DecodeStateful(nn.Module):
    """Stateful single-step decode. macOS 15+ / iOS 18+ only.

    Unlike ``Qwen2Decode``, the KV cache lives inside the model as a
    collection of ``register_buffer`` slots (one per layer per K/V)
    mutated in place each call. CoreML's ``StateType`` persists those
    buffers across calls so Swift no longer needs to round-trip ~18 MB
    of KV tensors through the binding layer every step.

    Per-layer buffers (instead of one stacked ``[L, …]`` tensor) keep
    each state's read / write symmetric — a single ``aten::copy_`` on
    the full buffer — which is the pattern coremltools' stateful pass
    recognises cleanly during trace lowering.

    Inputs:
        inputs_embeds: [1, 1, 896]
        cur_len:       [1] int32 — number of tokens already in cache;
                                    this step writes position cur_len and
                                    attends to positions [0, cur_len].
    States (persistent across calls):
        kv_k_{i}, kv_v_{i}: [1, Hkv, max_len, D] for i in 0..L-1,
                            populated externally by prefill before the
                            first decode step.
    Outputs:
        speech_logits: [1, 1, speech_vocab]
    """
    def __init__(self,
                 qwen_for_causal_lm: nn.Module,
                 speech_lm_head: nn.Linear,
                 max_len: int):
        super().__init__()
        qw = qwen_for_causal_lm
        cfg = qw.config
        self.num_layers   = cfg.num_hidden_layers
        self.num_heads    = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim     = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size  = cfg.hidden_size
        self.rope_theta   = cfg.rope_parameters["rope_theta"] if hasattr(cfg, "rope_parameters") else cfg.rope_theta
        self.max_len      = max_len

        self.register_buffer(
            "inv_freq",
            _build_rope_inv_freq(self.head_dim, self.rope_theta),
            persistent=False,
        )
        self.register_buffer(
            "pos_ids",
            torch.arange(max_len, dtype=torch.int32),
            persistent=False,
        )

        # Per-layer stateful KV cache buffers: 2*L entries, each
        # [1, Hkv, max_len, D] fp32 at trace time; coremltools will emit
        # them as StateType(fp16) when fp16 compute precision is used.
        Hkv, D = self.num_kv_heads, self.head_dim
        for i in range(self.num_layers):
            self.register_buffer(
                f"kv_k_{i}",
                torch.zeros(1, Hkv, max_len, D, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                f"kv_v_{i}",
                torch.zeros(1, Hkv, max_len, D, dtype=torch.float32),
                persistent=False,
            )

        self.layers = nn.ModuleList([
            Qwen2LayerDecode(qw.model.layers[i],
                             self.num_heads, self.num_kv_heads, self.head_dim)
            for i in range(self.num_layers)
        ])
        self.norm = RMSNorm(qw.model.norm.weight)
        self.speech_lm_head = _LinearLike(speech_lm_head)

    def forward(self,
                inputs_embeds: torch.Tensor,  # [1, 1, 896]
                cur_len: torch.Tensor,         # [1] int32
                ) -> torch.Tensor:
        M = self.max_len
        cur = cur_len.view(1).to(torch.int32)

        positions = cur.view(1, 1)
        cos, sin = _rope_cos_sin(positions, self.inv_freq)

        pj = self.pos_ids.view(1, 1, M, 1)
        update_mask = (pj == cur.view(1, 1, 1, 1)).to(inputs_embeds.dtype)

        attendable = self.pos_ids.view(1, 1, 1, M) <= cur.view(1, 1, 1, 1)
        neg_inf = torch.tensor(-1e4, dtype=torch.float32)
        attn_mask = torch.where(attendable,
                                torch.zeros((), dtype=torch.float32),
                                neg_inf).to(inputs_embeds.dtype)

        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            k_i = getattr(self, f"kv_k_{i}")
            v_i = getattr(self, f"kv_v_{i}")
            x, k_full, v_full = layer(x, cos, sin, k_i, v_i, update_mask, attn_mask)
            # In-place state writes. coremltools' generate_tensor_assignment_ops
            # pass requires `.copy_()` to be preceded by a `select` / `slice`;
            # writing via [:] satisfies that while still overwriting the whole
            # buffer. Each of these lowers to an independent StateType update.
            getattr(self, f"kv_k_{i}")[:] = k_full
            getattr(self, f"kv_v_{i}")[:] = v_full

        x = self.norm(x)
        speech_logits = self.speech_lm_head(x)
        return speech_logits


class Qwen2Decode(nn.Module):
    """Static-shape single-step decode with KV cache update.

    Inputs:
        inputs_embeds: [1, 1, 896]
        kv_k, kv_v:    [L, 1, Hkv, max_len, D]
        cur_len:       [1] int32 — number of tokens already in cache;
                                    this step writes position cur_len and
                                    attends to positions [0, cur_len].
    Outputs:
        speech_logits: [1, 1, speech_vocab]
        kv_k_out, kv_v_out: updated caches with position cur_len populated
    """
    def __init__(self,
                 qwen_for_causal_lm: nn.Module,
                 speech_lm_head: nn.Linear,
                 max_len: int):
        super().__init__()
        qw = qwen_for_causal_lm
        cfg = qw.config
        self.num_layers   = cfg.num_hidden_layers
        self.num_heads    = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim     = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size  = cfg.hidden_size
        self.rope_theta   = cfg.rope_parameters["rope_theta"] if hasattr(cfg, "rope_parameters") else cfg.rope_theta
        self.max_len      = max_len

        self.register_buffer(
            "inv_freq",
            _build_rope_inv_freq(self.head_dim, self.rope_theta),
            persistent=False,
        )
        # Precompute the arange(max_len) so we can build masks from a dynamic cur_len.
        self.register_buffer(
            "pos_ids",
            torch.arange(max_len, dtype=torch.int32),
            persistent=False,
        )

        self.layers = nn.ModuleList([
            Qwen2LayerDecode(qw.model.layers[i],
                             self.num_heads, self.num_kv_heads, self.head_dim)
            for i in range(self.num_layers)
        ])
        self.norm = RMSNorm(qw.model.norm.weight)
        self.speech_lm_head = _LinearLike(speech_lm_head)

    def forward(self,
                inputs_embeds: torch.Tensor,  # [1, 1, 896]
                kv_k: torch.Tensor,            # [L, 1, Hkv, max_len, D]
                kv_v: torch.Tensor,            # [L, 1, Hkv, max_len, D]
                cur_len: torch.Tensor,         # [1] int32
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M = self.max_len
        cur = cur_len.view(1).to(torch.int32)

        # Position ids for the new token: [[cur_len]]
        positions = cur.view(1, 1)
        cos, sin = _rope_cos_sin(positions, self.inv_freq)

        # Build a one-hot scatter mask of shape [1, 1, M, 1]:
        #   update_mask[:, :, j, :] = 1 iff j == cur_len else 0
        pj = self.pos_ids.view(1, 1, M, 1)                        # [1,1,M,1]
        update_mask = (pj == cur.view(1, 1, 1, 1)).to(inputs_embeds.dtype)

        # Build attention mask: allow j in [0, cur_len], deny otherwise.
        # Shape [1, 1, 1, M] additive.
        attendable = self.pos_ids.view(1, 1, 1, M) <= cur.view(1, 1, 1, 1)
        neg_inf = torch.tensor(-1e4, dtype=torch.float32)
        attn_mask = torch.where(attendable,
                                torch.zeros((), dtype=torch.float32),
                                neg_inf).to(inputs_embeds.dtype)

        x = inputs_embeds
        k_all: List[torch.Tensor] = []
        v_all: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            k_i = kv_k[i]   # [1, Hkv, M, D]
            v_i = kv_v[i]
            x, k_full, v_full = layer(x, cos, sin, k_i, v_i, update_mask, attn_mask)
            k_all.append(k_full)
            v_all.append(v_full)

        x = self.norm(x)
        speech_logits = self.speech_lm_head(x)   # [1, 1, speech_vocab]
        kv_k_out = torch.stack(k_all, dim=0)
        kv_v_out = torch.stack(v_all, dim=0)
        return speech_logits, kv_k_out, kv_v_out
