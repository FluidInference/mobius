"""Tracing-only DeBERTa relative attention for fixed batch 1.

This is adapted from Hugging Face Transformers' Apache-2.0
``DisentangledSelfAttention.disentangled_attention_bias``. The trained weights
are untouched. The only semantic change is a literal repeat count of one for
the fixed B=1 Core ML export, avoiding an aten::Int conversion failure.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from transformers.models.deberta_v2.modeling_deberta_v2 import build_relative_position


def constant_attention_scale(query_layer: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Static head-width sqrt, equivalent to Transformers' fp32 calculation."""
    return torch.tensor(
        math.sqrt(query_layer.shape[-1] * scale_factor),
        dtype=torch.float32,
        device=query_layer.device,
    )


def batch_one_disentangled_attention_bias(
    self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
):
    if query_layer.shape[0] != self.num_attention_heads:
        raise ValueError("this tracing path requires batch size one")
    if relative_pos is None:
        relative_pos = build_relative_position(
            query_layer,
            key_layer,
            bucket_size=self.position_buckets,
            max_position=self.max_relative_positions,
        )
    if relative_pos.dim() == 2:
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
    elif relative_pos.dim() == 3:
        relative_pos = relative_pos.unsqueeze(1)
    elif relative_pos.dim() != 4:
        raise ValueError(f"relative position ids must have 2, 3 or 4 dims; got {relative_pos.dim()}")

    att_span = self.pos_ebd_size
    relative_pos = relative_pos.to(device=query_layer.device, dtype=torch.long)
    rel_embeddings = rel_embeddings[: att_span * 2, :].unsqueeze(0)
    if self.share_att_key:
        pos_query_layer = self.transpose_for_scores(
            self.query_proj(rel_embeddings), self.num_attention_heads
        ).repeat(1, 1, 1)
        pos_key_layer = self.transpose_for_scores(
            self.key_proj(rel_embeddings), self.num_attention_heads
        ).repeat(1, 1, 1)
    else:
        if "c2p" in self.pos_att_type:
            pos_key_layer = self.transpose_for_scores(
                self.pos_key_proj(rel_embeddings), self.num_attention_heads
            ).repeat(1, 1, 1)
        if "p2c" in self.pos_att_type:
            pos_query_layer = self.transpose_for_scores(
                self.pos_query_proj(rel_embeddings), self.num_attention_heads
            ).repeat(1, 1, 1)

    score = 0
    if "c2p" in self.pos_att_type:
        scale = constant_attention_scale(pos_key_layer, scale_factor)
        c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
        c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        c2p_att = torch.gather(
            c2p_att,
            dim=-1,
            index=c2p_pos.squeeze(0).expand(
                [query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]
            ),
        )
        score += c2p_att / scale.to(dtype=c2p_att.dtype)

    if "p2c" in self.pos_att_type:
        scale = constant_attention_scale(pos_query_layer, scale_factor)
        if query_layer.shape[-2] != key_layer.shape[-2]:
            raise ValueError("fixed classification encoder requires equal query and key lengths")
        r_pos = relative_pos
        p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
        p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
        p2c_att = torch.gather(
            p2c_att,
            dim=-1,
            index=p2c_pos.squeeze(0).expand(
                [query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]
            ),
        ).transpose(-1, -2)
        score += p2c_att / scale.to(dtype=p2c_att.dtype)
    return score


def install_trace_compatibility() -> None:
    """Install fixed-shape tracing helpers; call only after native baseline."""
    import gliformer.backbones.deberta_2d as gliformer_deberta
    import transformers.models.deberta_v2.modeling_deberta_v2 as hf_deberta

    gliformer_deberta.scaled_size_sqrt = constant_attention_scale
    hf_deberta.scaled_size_sqrt = constant_attention_scale
    gliformer_deberta.LayoutDisentangledSelfAttention.disentangled_attention_bias = (
        batch_one_disentangled_attention_bias
    )


@contextmanager
def finite_fp16_mask():
    """Trace finite attention-mask fill instead of fp32 minimum overflowing to FP16 -inf.

    Valid attention rows retain the same softmax in fp32. The unmasked native
    baseline and patched model are compared on every parity fixture.
    """
    original = torch.finfo

    class FiniteFinfo:
        def __init__(self, real):
            self.real = real

        def __getattr__(self, name):
            return getattr(self.real, name)

        @property
        def min(self):
            return -10000.0

    def patched(dtype):
        info = original(dtype)
        return FiniteFinfo(info) if dtype.is_floating_point else info

    torch.finfo = patched
    try:
        yield
    finally:
        torch.finfo = original
