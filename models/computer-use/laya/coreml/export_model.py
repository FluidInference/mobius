"""Fixed-shape, eager-attention export adapter around the unmodified laya DecisionModel."""

from __future__ import annotations

import torch
from torch import nn
from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

MASK_VALUE = -1e4


class LayaExport(nn.Module):
    """One question per call. Inputs are padded to a fixed length; options use one-hot marker rows.

    input_ids      int32   [1, L]
    attention_mask int32   [1, L]     1 for real tokens
    marker_map     float32 [1, K, L]  one-hot row per option, zero rows for unused option slots
    question_type  float32 [1, 3]     one-hot over choice / score / noul

    logits               float32 [1, K]  unused option slots hold MASK_VALUE
    probabilities        float32 [1, K]  softmax over supplied options (temperature 1)
    action_probabilities float32 [1, 2]
    """

    def __init__(self, model: nn.Module, length: int, max_options: int):
        super().__init__()
        self.model = model
        self.length = length
        self.max_options = max_options
        encoder = model.encoder
        config = encoder.config
        config._attn_implementation = "eager"
        self.layers = encoder.layers
        self.embeddings = encoder.embeddings
        self.final_norm = encoder.final_norm
        self.layer_types = list(config.layer_types)
        self.head_dim = config.hidden_size // config.num_attention_heads

        position_ids = torch.arange(length).unsqueeze(0)
        probe = torch.zeros(1, length, config.hidden_size)
        for layer_type in sorted(set(self.layer_types)):
            cos, sin = encoder.rotary_emb(probe, position_ids, layer_type)
            self.register_buffer(f"cos_{layer_type}", cos.detach().clone())
            self.register_buffer(f"sin_{layer_type}", sin.detach().clone())
        # Exact HF sliding-window band, materialised once (additive, fp16-safe)
        ones = torch.ones(1, length, dtype=torch.long)
        band = create_bidirectional_sliding_window_mask(
            config=config, inputs_embeds=probe, attention_mask=ones, allow_is_bidirectional_skip=False
        )
        full = create_bidirectional_mask(
            config=config, inputs_embeds=probe, attention_mask=ones, allow_is_bidirectional_skip=False
        )
        self.register_buffer("band_mask", self._to_additive(band, length))
        self.register_buffer("full_mask", self._to_additive(full, length))
        head = model.head.layers
        self.head_layers = nn.ModuleList(head)
        self.type_weight = model.type_emb.weight  # [3, d]
        self.scorer = model.scorer
        self.act_head = model.act_head

    @staticmethod
    def _to_additive(mask, length: int) -> torch.Tensor:
        if mask is None:
            return torch.zeros(1, 1, length, length)
        if mask.dtype == torch.bool:
            return torch.where(mask, torch.zeros(()), torch.full((), MASK_VALUE)).float()
        return torch.where(mask < 0, torch.full((), MASK_VALUE), torch.zeros(())).float()

    def _head_layer(self, layer: nn.TransformerEncoderLayer, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # norm_first TransformerEncoderLayer with relu, written out so tracing sees plain matmuls.
        attn = layer.self_attn
        h = layer.norm1(x)
        b, s, d = h.shape
        qkv = nn.functional.linear(h, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        heads = attn.num_heads
        hd = d // heads
        q = q.view(b, s, heads, hd).transpose(1, 2)
        k = k.view(b, s, heads, hd).transpose(1, 2)
        v = v.view(b, s, heads, hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) * (hd**-0.5) + mask
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(b, s, d)
        x = x + attn.out_proj(out)
        h = layer.norm2(x)
        return x + layer.linear2(torch.relu(layer.linear1(h)))

    def forward(self, input_ids, attention_mask, marker_map, question_type):
        pad = (1.0 - attention_mask.float()).view(1, 1, 1, self.length) * MASK_VALUE
        full = self.full_mask + pad
        band = self.band_mask + pad
        x = self.embeddings(input_ids=input_ids.long())
        for layer, layer_type in zip(self.layers, self.layer_types):
            cos = getattr(self, f"cos_{layer_type}")
            sin = getattr(self, f"sin_{layer_type}")
            x = layer(
                x, attention_mask=full if layer_type == "full_attention" else band, position_embeddings=(cos, sin)
            )
        x = self.final_norm(x)
        x = x + torch.matmul(question_type, self.type_weight).unsqueeze(1)
        for layer in self.head_layers:
            x = self._head_layer(layer, x, full)
        markers = torch.matmul(marker_map, x)  # [1, K, d]
        logits = self.scorer(markers).squeeze(-1)  # [1, K]
        option_mask = marker_map.sum(-1)  # [1, K] 1 for supplied options
        logits = logits * option_mask + (1.0 - option_mask) * MASK_VALUE
        probabilities = torch.softmax(logits, dim=-1)
        k = option_mask.sum(-1, keepdim=True).clamp(min=2.0)
        entropy = -(probabilities * torch.log(probabilities.clamp(min=1e-9))).sum(-1, keepdim=True) / torch.log(k)
        top2 = probabilities.topk(2, dim=-1).values
        feats = torch.cat([top2[:, :1], top2[:, :1] - top2[:, 1:2], entropy, k / 255.0], dim=-1)
        pooled = x[:, 0]
        action = torch.softmax(self.act_head(torch.cat([pooled, feats], dim=-1)), dim=-1)
        return logits, probabilities, action
