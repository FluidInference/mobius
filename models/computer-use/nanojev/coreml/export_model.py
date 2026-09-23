"""Fixed-shape Qwen3 candidate encoder and complete NanoJev decision head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

MASK = -1e4


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rms_norm(hidden: torch.Tensor, norm: nn.Module, width: int) -> torch.Tensor:
    doubled = torch.cat((hidden, -hidden), dim=-1)
    normalized = F.layer_norm(doubled, (width * 2,), eps=norm.variance_epsilon)
    return normalized[..., :width] * norm.weight


class NanoEncoder(nn.Module):
    """Encode the complete set of candidate paths with the trained Qwen3 body."""

    def __init__(self, decision_model: nn.Module, length: int, candidates: int):
        super().__init__()
        body = decision_model.backbone
        config = body.config
        self.embed_tokens = body.embed_tokens
        self.layers = body.layers
        self.norm = body.norm
        self.length = length
        self.candidates = candidates
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.groups = self.heads // self.kv_heads
        self.head_dim = config.head_dim
        self.hidden = config.hidden_size
        self.scale = self.head_dim**-0.5
        positions = torch.arange(length).unsqueeze(0)
        probe = torch.zeros(1, length, config.hidden_size)
        cos, sin = body.rotary_emb(probe, positions)
        self.register_buffer("cos", cos.unsqueeze(1).detach().clone())
        self.register_buffer("sin", sin.unsqueeze(1).detach().clone())
        causal = torch.triu(torch.full((length, length), MASK), diagonal=1)
        self.register_buffer("causal", causal.view(1, 1, length, length))

    def _layer(self, layer: nn.Module, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention = layer.self_attn
        normalized = rms_norm(hidden, layer.input_layernorm, self.hidden)
        q = attention.q_proj(normalized).view(self.candidates, self.length, self.heads, self.head_dim)
        k = attention.k_proj(normalized).view(self.candidates, self.length, self.kv_heads, self.head_dim)
        v = attention.v_proj(normalized).view(self.candidates, self.length, self.kv_heads, self.head_dim)
        q = attention.q_norm(q).transpose(1, 2)
        k = attention.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * self.cos + rotate_half(q) * self.sin
        k = k * self.cos + rotate_half(k) * self.sin
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        weights = torch.matmul(q, k.transpose(2, 3)) * self.scale + mask
        weights = torch.softmax(weights.float(), dim=-1).to(q.dtype)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(
            self.candidates, self.length, self.heads * self.head_dim
        )
        hidden = hidden + attention.o_proj(attended)
        return hidden + layer.mlp(rms_norm(hidden, layer.post_attention_layernorm, self.hidden))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, eos_map: torch.Tensor
    ) -> torch.Tensor:
        padding = (1.0 - attention_mask.float()).view(self.candidates, 1, 1, self.length) * MASK
        mask = self.causal + padding
        hidden = self.embed_tokens(input_ids.long())
        for layer in self.layers:
            hidden = self._layer(layer, hidden, mask)
        hidden = rms_norm(hidden, self.norm, self.hidden).float()
        return torch.matmul(eos_map, hidden).transpose(0, 1)


class NanoHead(nn.Module):
    """The trained scalar and set-attention heads for Choice, Boolean, and Score."""

    def __init__(self, decision_model: nn.Module, candidates: int):
        super().__init__()
        self.norm = decision_model.norm
        self.scalar = decision_model.scalar
        self.set_project = decision_model.set_project
        self.set_attention = decision_model.set_attention
        self.set_output = decision_model.set_output
        self.candidates = candidates
        self.heads = 4
        self.head_dim = 32
        self.scale = 1.0 / math.sqrt(self.head_dim)
        boolean_mask = torch.zeros(1, candidates)
        boolean_mask[0, :2] = 1
        self.register_buffer("boolean_mask", boolean_mask)

    def forward(
        self,
        embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        use_set_head: torch.Tensor,
        is_boolean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(embeddings)
        scalar = self.scalar(h).squeeze(-1).float()
        log_k = candidate_mask.sum(-1).clamp(min=1).float().log().view(1, 1, 1)
        log_k = log_k.expand(1, self.candidates, 1)
        u = self.set_project(torch.cat((h, log_k.to(h.dtype)), dim=-1))
        projected = F.linear(u, self.set_attention.in_proj_weight, self.set_attention.in_proj_bias)
        q, k, v = projected.chunk(3, dim=-1)
        q = q.view(1, self.candidates, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(1, self.candidates, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(1, self.candidates, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        scores = scores + (1.0 - candidate_mask).view(1, 1, 1, self.candidates) * MASK
        mixed = torch.matmul(torch.softmax(scores.float(), dim=-1).to(v.dtype), v)
        mixed = mixed.transpose(1, 2).reshape(1, self.candidates, -1)
        mixed = self.set_attention.out_proj(mixed)
        delta = self.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        normal_logits = scalar + use_set_head * delta
        boolean_logits = torch.cat((scalar[:, :1] * 0, scalar[:, :1], scalar[:, 2:] * 0), dim=-1)
        logits = normal_logits * (1.0 - is_boolean) + boolean_logits * is_boolean
        output_mask = candidate_mask * (1.0 - is_boolean) + self.boolean_mask * is_boolean
        logits = logits * output_mask + (1.0 - output_mask) * MASK
        return logits, torch.softmax(logits, dim=-1)
