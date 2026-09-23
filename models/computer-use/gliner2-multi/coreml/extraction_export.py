"""Core ML graph wrappers for GLiNER2.5's trained boundary extraction path."""

import math
from contextlib import contextmanager

import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.mil import types
from gliner2.models.boundary import encoding, heads
from gliner2.models.boundary.pool import PooledCandidates
from gliner2.models.boundary.proposal import BoundaryProposals
from transformers.models.deberta_v2 import modeling_deberta_v2

from export_model import coreml_safe_attention_forward


@register_torch_op(override=True)
def clamp_min(context, node):
    """Preserve the tensor dtype when TorchScript supplied a Python scalar."""
    x, y = _get_inputs(context, node, expected=2)
    if x.dtype != y.dtype:
        y = mb.cast(x=y, dtype=types.builtin_to_string(x.dtype))
    context.add(mb.maximum(x=x, y=y, name=node.name))


@register_torch_op(torch_alias=["clip"], override=True)
def clamp(context, node):
    """Avoid promoting integer span indices to float for an absent bound."""
    inputs = _get_inputs(context, node, expected=[1, 2, 3])
    x = inputs[0]
    lower = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
    upper = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    result = x
    for bound, op in ((upper, mb.minimum), (lower, mb.maximum)):
        if bound is None:
            continue
        if bound.dtype != x.dtype:
            bound = mb.cast(x=bound, dtype=types.builtin_to_string(x.dtype))
        result = op(x=result, y=bound)
    context.add(mb.identity(x=result, name=node.name))


def shift_left(text_states, bos_state):
    """Functional equivalent of the upstream in-place BOS placement."""
    bos = bos_state.to(text_states.dtype).view(1, 1, -1)
    return torch.cat((bos.expand(text_states.shape[0], 1, -1), text_states), 1)


def shift_right(text_states, text_lengths, eos_state):
    """Functional equivalent of the upstream in-place EOS placement."""
    batch, length, hidden = text_states.shape
    eos = eos_state.to(text_states.dtype).view(1, 1, hidden)
    right = torch.cat((text_states, eos.expand(batch, 1, hidden)), 1)
    positions = torch.arange(length + 1, device=text_states.device).view(1, length + 1, 1)
    return torch.where(
        positions == text_lengths.view(batch, 1, 1),
        eos.expand(batch, length + 1, hidden),
        right,
    )


def safe_boundary_attention(self, states, mask):
    """Explicit scaled attention with the same finite masked result as upstream."""
    batch, length, dim = states.shape
    qkv = self.qkv_projection(self.norm(states)).view(batch, length, 3, self.num_heads, self.head_dim)
    query, key, value = qkv.permute(2, 0, 3, 1, 4)
    allowed = mask.view(batch, 1, 1, length).expand(batch, 1, length, length)
    if self.window > 0:
        positions = torch.arange(length, device=states.device)
        local = (positions.view(length, 1) - positions.view(1, length)).abs() <= self.window
        allowed = allowed & local.view(1, 1, length, length)
    diagonal = torch.eye(length, dtype=torch.bool, device=states.device).view(1, 1, length, length)
    allowed = (allowed.float() + diagonal.float()) > 0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
    scores = scores.masked_fill(~allowed, -1e4)
    attended = torch.matmul(torch.softmax(scores, dim=-1), value)
    attended = attended.transpose(1, 2).reshape(batch, length, dim)
    return (states + self.dropout(self.output_projection(attended))) * mask.unsqueeze(-1).to(states.dtype)


def safe_query_head(self, boundary, boundary_mask, text, text_mask, query, query_mask):
    """Upstream marginals with a dtype-safe count clamp for Core ML."""
    scale = 1.0 / math.sqrt(self.boundary_dim)
    start = (
        torch.einsum(
            "bld,bqd->bql",
            self.dropout(self.start_boundary_projection(boundary)),
            self.start_query_projection(query),
        )
        * scale
    )
    end = (
        torch.einsum(
            "bld,bqd->bql",
            self.dropout(self.end_boundary_projection(boundary)),
            self.end_query_projection(query),
        )
        * scale
    )
    inside = (
        torch.einsum(
            "bld,bqd->bql",
            self.dropout(self.inside_text_projection(text)),
            self.inside_query_projection(query),
        )
        * scale
    )
    boundary_keep = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    text_keep = text_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    start = heads._masked_fill_min(start, boundary_keep)
    end = heads._masked_fill_min(end, boundary_keep)
    inside = heads._masked_fill_min(inside, text_keep)
    inside_for_prefix = inside.masked_fill(~text_keep, 0.0).float()
    count = torch.clamp(text_keep.sum(-1, keepdim=True).float(), min=1.0)
    mean = (inside_for_prefix.sum(-1, keepdim=True) / count).detach()
    centered = (inside_for_prefix - mean) * text_keep.to(inside_for_prefix.dtype)
    zeros = torch.zeros(centered.shape[0], centered.shape[1], 1, dtype=torch.float32, device=text.device)
    prefix = torch.cat((zeros, centered.cumsum(dim=-1)), dim=-1)
    return heads.BoundaryMarginals(start, end, inside, prefix, mean)


@contextmanager
def coreml_trace_patches():
    """Apply and restore mathematically equivalent trace-safe operations."""
    saved = (
        modeling_deberta_v2.scaled_size_sqrt,
        modeling_deberta_v2.build_rpos,
        modeling_deberta_v2.DisentangledSelfAttention.forward,
        encoding.shift_left_with_bos,
        encoding.shift_right_with_eos,
        encoding.BoundaryAttentionBlock.forward,
        heads.BoundaryQueryHead.forward,
    )

    def static_scale(query_layer, scale_factor):
        value = math.sqrt(float(query_layer.shape[-1] * scale_factor))
        return torch.tensor(value, dtype=torch.float32, device=query_layer.device)

    modeling_deberta_v2.scaled_size_sqrt = static_scale
    modeling_deberta_v2.build_rpos = lambda query, key, relative_pos, buckets, max_pos: relative_pos
    modeling_deberta_v2.DisentangledSelfAttention.forward = coreml_safe_attention_forward
    encoding.shift_left_with_bos = shift_left
    encoding.shift_right_with_eos = shift_right
    encoding.BoundaryAttentionBlock.forward = safe_boundary_attention
    heads.BoundaryQueryHead.forward = safe_query_head
    try:
        yield
    finally:
        (
            modeling_deberta_v2.scaled_size_sqrt,
            modeling_deberta_v2.build_rpos,
            modeling_deberta_v2.DisentangledSelfAttention.forward,
            encoding.shift_left_with_bos,
            encoding.shift_right_with_eos,
            encoding.BoundaryAttentionBlock.forward,
            heads.BoundaryQueryHead.forward,
        ) = saved


class ExtractionFeaturesExport(torch.nn.Module):
    """Trained encoder, boundary marginals, pool projections and null/count heads."""

    def __init__(self, native):
        super().__init__()
        self.encoder = native.encoder
        self.head = native.boundary_head
        self.classifier = native.classifier

    def forward(
        self,
        input_ids,
        attention_mask,
        text_indices,
        text_mask,
        query_indices,
        query_mask,
        cls_indices,
        cls_mask,
    ):
        hidden = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        text_idx = text_indices.long().unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        query_idx = query_indices.long().unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        text = hidden.gather(1, text_idx) * text_mask.unsqueeze(-1)
        query = hidden.gather(1, query_idx) * query_mask.unsqueeze(-1)
        cls_idx = cls_indices.long().unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        classification_states = hidden.gather(1, cls_idx)
        classification_logits = self.classifier(classification_states).squeeze(-1)
        classification_logits = torch.where(
            cls_mask > 0.5, classification_logits, torch.full_like(classification_logits, -1e4)
        )
        tm, qm = text_mask > 0.5, query_mask > 0.5
        encoded = self.head.boundary_encoder(text, tm)
        marginal = self.head.boundary_query_head(encoded.states, encoded.mask, text, tm, query, qm)
        return (
            text,
            query,
            encoded.states,
            encoded.mask.float(),
            marginal.start_logits,
            marginal.end_logits,
            marginal.inside_prefix,
            marginal.inside_prefix_mean,
            self.head.shared_pool_builder.start_projection(encoded.states),
            self.head.shared_pool_builder.end_projection(encoded.states),
            self.head.null_projection(query).squeeze(-1),
            self.head.count_head(query).squeeze(-1),
            classification_logits,
        )


class ExtractionScoreExport(torch.nn.Module):
    """Trained shared-pool reranker and record candidate state projection."""

    def __init__(self, native):
        super().__init__()
        self.scorer = native.boundary_head.shared_pool_scorer
        self.candidate_encoder = native.boundary_head.candidate_encoder

    def forward(
        self,
        text,
        text_mask,
        query,
        query_mask,
        boundary,
        starts,
        ends,
        inside,
        inside_mean,
        indices,
        pool_mask,
        compatibility,
    ):
        tm, qm = text_mask > 0.5, query_mask > 0.5
        pooled = PooledCandidates(indices.long(), pool_mask > 0.5, None, None, compatibility)
        score, _ = self.scorer(
            boundary,
            query,
            qm,
            pooled,
            starts,
            ends,
            inside,
            tm.sum(-1).long(),
            text,
            tm,
            inside_prefix_mean=inside_mean,
        )
        index = indices.long()
        start_states = boundary.gather(1, index[..., 0].unsqueeze(-1).expand(-1, -1, boundary.shape[-1]))
        end_states = boundary.gather(1, index[..., 1].unsqueeze(-1).expand(-1, -1, boundary.shape[-1]))
        candidate_states = self.candidate_encoder(torch.cat((start_states, end_states), -1))
        candidate_states = candidate_states * pool_mask.unsqueeze(-1)
        return score.transpose(1, 2), candidate_states


class ExtractionRelationExport(torch.nn.Module):
    """The trained sparse relation scorer with tensor-only pair routing."""

    def __init__(self, native):
        super().__init__()
        self.scorer = native.relation_scorer

    def forward(
        self,
        text,
        text_length,
        relation,
        batch_index,
        relation_index,
        head_start,
        head_end,
        tail_start,
        tail_end,
        pair_mask,
    ):
        scorer = self.scorer
        length = text.shape[1]
        batch_valid = (batch_index >= 0) & (batch_index < text.shape[0])
        relation_valid = (relation_index >= 0) & (relation_index < relation.shape[1])
        valid = batch_valid & relation_valid & (pair_mask > 0.5)
        batch = batch_index.long().clamp(0, text.shape[0] - 1)
        rel_index = relation_index.long().clamp(0, relation.shape[1] - 1)

        def gather(position):
            return text[batch, position.long().clamp(0, length - 1)]

        h_start = gather(head_start)
        h_end = gather(head_end - 1)
        t_start = gather(tail_start)
        t_end = gather(tail_end - 1)
        rel = relation[batch, rel_index]
        delta = (tail_start - head_start).to(text.dtype)
        order = torch.sign(delta).unsqueeze(-1)
        distance = (delta.abs() / text_length.float().clamp_min(1.0)).unsqueeze(-1)
        features = torch.cat((h_start, h_end, t_start, t_end, rel, order, distance), -1)
        score = scorer.mlp(features).squeeze(-1)
        if scorer.use_biaffine_content:
            prefix = torch.cat(
                (text.new_zeros(text.shape[0], 1, scorer.hidden_size), text.float().cumsum(1).to(text.dtype)),
                dim=1,
            )

            def pool(start, end):
                total = prefix[batch, end.long().clamp(0, length)] - prefix[batch, start.long().clamp(0, length)]
                width = (end - start).clamp_min(1).unsqueeze(-1).to(total.dtype)
                return total / width

            head_content = scorer.head_content_projection(pool(head_start, head_end))
            tail_content = scorer.tail_content_projection(pool(tail_start, tail_end))
            gate = torch.sigmoid(scorer.relation_content_gate(rel))
            biaffine = (head_content * gate * tail_content).sum(-1) / (scorer.hidden_size**0.5)
            linear = scorer.content_linear(torch.cat((head_content, tail_content, rel), -1)).squeeze(-1)
            score = score + biaffine + linear
        return score.masked_fill(~valid, 0.0)


class ExtractionRecordAssignmentExport(torch.nn.Module):
    """All trained natural/latent/anchorless object and field assignment layers."""

    def __init__(self, native):
        super().__init__()
        self.head = native.record_decoder

    def forward(self, instances, field_queries, field_candidates):
        head = self.head
        instance_projection = head.inst_proj(instances)
        field_projection = head.field_proj(field_queries)
        query = instance_projection.unsqueeze(1) + field_projection.unsqueeze(0)
        null_scores = torch.einsum("ifd,d->if", query, head.null_embed)
        candidate_scores = torch.einsum("ifd,fcd->ifc", query, head.cand_proj(field_candidates))
        assignment = torch.cat((null_scores.unsqueeze(-1), candidate_scores), -1)
        object_scores = head.object_head(instances).squeeze(-1)
        latent_scores = head.latent_seed_head(instances).squeeze(-1)
        return assignment, object_scores, latent_scores


class ExtractionRecordAnchorlessExport(torch.nn.Module):
    """Trained learned-instance and contextual attention path for records."""

    def __init__(self, native):
        super().__init__()
        self.head = native.record_decoder

    def forward(self, context_states, context_mask):
        head = self.head
        instances = head.instance_embed
        query = head.q_proj(instances)
        key = head.k_proj(context_states)
        value = head.v_proj(context_states)
        attention = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head.record_dim)
        attention = attention.masked_fill(context_mask.unsqueeze(0) < 0.5, -1e4)
        pooled = torch.matmul(torch.softmax(attention, -1), value)
        return instances + pooled * (context_mask.sum() > 0).to(pooled.dtype)


class ExtractionExplicitSpanExport(torch.nn.Module):
    """Trained proposal prior and reranker for forced attribute/enum spans."""

    def __init__(self, native):
        super().__init__()
        self.proposer = native.boundary_head.boundary_proposer
        self.scorer = native.boundary_head.pair_scorer

    def forward(
        self,
        text,
        text_mask,
        query,
        query_mask,
        boundary,
        starts,
        ends,
        inside,
        inside_mean,
        indices,
        valid_mask,
    ):
        tm, qm = text_mask > 0.5, query_mask > 0.5
        idx = indices.long()
        legal = (
            (idx[..., 0] >= 0)
            & (idx[..., 1] > idx[..., 0])
            & (idx[..., 1] <= tm.sum(-1).view(-1, 1, 1))
            & qm.unsqueeze(-1)
            & (valid_mask > 0.5)
        )
        compatibility = self.proposer.score_explicit_pairs(boundary, query, idx, legal)
        proposals = BoundaryProposals(
            indices=idx,
            logits=None,
            valid_mask=legal,
            compat_logits=compatibility,
        )
        return self.scorer(
            boundary,
            query,
            proposals,
            starts,
            ends,
            inside,
            tm.sum(-1).long(),
            text,
            tm,
            inside_prefix_mean=inside_mean,
        )
