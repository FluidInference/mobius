"""Fixed-shape Qwen2.5 plus Kev pointer-head export wrapper."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

MASK_VALUE = -1e4


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class KevExport(nn.Module):
    """Run one typed question against one state in a fixed token bucket.

    Inputs use one-hot maps for the dynamic readout positions so Core ML never needs
    data-dependent gather indices.
    """

    def __init__(self, decision_model: nn.Module, length: int, max_options: int):
        super().__init__()
        language_model = decision_model.lm
        config = language_model.config
        self.embed_tokens = language_model.embed_tokens
        self.layers = language_model.layers
        self.norm = language_model.norm
        self.pointer_q = decision_model.head.q
        self.pointer_k = decision_model.head.k
        self.length = length
        self.max_options = max_options
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.attention_scale = self.head_dim**-0.5
        self.pointer_scale = 1 / math.sqrt(self.pointer_q.out_features)

        position_ids = torch.arange(length).unsqueeze(0)
        probe = torch.zeros(1, length, config.hidden_size)
        cos, sin = language_model.rotary_emb(probe, position_ids)
        self.register_buffer("position_cos", cos.unsqueeze(1).detach().clone())
        self.register_buffer("position_sin", sin.unsqueeze(1).detach().clone())
        causal = torch.full((length, length), MASK_VALUE)
        causal = torch.triu(causal, diagonal=1)
        self.register_buffer("causal_mask", causal.view(1, 1, length, length))

    def _rms_norm(self, hidden: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        doubled = torch.cat((hidden, -hidden), dim=-1)
        normalized = F.layer_norm(doubled, (self.hidden_size * 2,), eps=norm.variance_epsilon)
        return normalized[..., : self.hidden_size] * norm.weight

    def _attention(self, layer: nn.Module, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention = layer.self_attn
        residual = hidden
        normalized = self._rms_norm(hidden, layer.input_layernorm)
        queries = (
            attention.q_proj(normalized)
            .view(1, self.length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        keys = (
            attention.k_proj(normalized)
            .view(1, self.length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            attention.v_proj(normalized)
            .view(1, self.length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        queries = queries * self.position_cos + rotate_half(queries) * self.position_sin
        keys = keys * self.position_cos + rotate_half(keys) * self.position_sin
        keys = keys.repeat_interleave(self.num_key_value_groups, dim=1)
        values = values.repeat_interleave(self.num_key_value_groups, dim=1)
        weights = torch.matmul(queries, keys.transpose(2, 3)) * self.attention_scale + mask
        weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attended = torch.matmul(weights, values).transpose(1, 2).reshape(1, self.length, -1)
        hidden = residual + attention.o_proj(attended)
        residual = hidden
        normalized = self._rms_norm(hidden, layer.post_attention_layernorm)
        return residual + layer.mlp(normalized)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decide_map: torch.Tensor,
        option_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding = (1.0 - attention_mask.float()).view(1, 1, 1, self.length) * MASK_VALUE
        mask = self.causal_mask + padding
        hidden = self.embed_tokens(input_ids.long())
        for layer in self.layers:
            hidden = self._attention(layer, hidden, mask)
        hidden = self._rms_norm(hidden, self.norm).float()
        decide = torch.matmul(decide_map, hidden)
        options = torch.matmul(option_map, hidden)
        query = self.pointer_q(decide)
        keys = self.pointer_k(options)
        logits = torch.matmul(keys, query.transpose(1, 2)).squeeze(-1) * self.pointer_scale
        supplied = option_map.sum(-1)
        logits = logits * supplied + (1.0 - supplied) * MASK_VALUE
        return logits, torch.softmax(logits, dim=-1)
