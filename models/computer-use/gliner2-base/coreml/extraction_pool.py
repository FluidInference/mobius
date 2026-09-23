"""Weight-free GLiNER2 boundary candidate selection between Core ML stages.

The learned start/end projections are outputs of the first Core ML stage. This
module preserves GLiNER2 2.0.0's stable ranking and deduplication on the host.
"""

import math

import torch
from gliner2.models.boundary.constants import MASK_LOGIT
from gliner2.models.boundary.indexing import gather_rows
from gliner2.models.boundary.pool import PooledCandidates, _deduplicate_pool
from gliner2.models.boundary.proposal import select_top_boundaries


def select_candidates(
    start_projection: torch.Tensor,
    end_projection: torch.Tensor,
    boundary_mask: torch.Tensor,
    query_mask: torch.Tensor,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    *,
    boundary_top_k: int,
    pool_size: int,
    min_pool_per_query: int,
) -> PooledCandidates:
    """Select the native shared pool using already projected Core ML states."""
    batch, n_boundaries, dim = start_projection.shape
    n_queries = query_mask.shape[1]
    if end_projection.shape != start_projection.shape:
        raise ValueError("Start and end projections must have the same shape")
    if start_logits.shape != (batch, n_queries, n_boundaries):
        raise ValueError("Start logits do not match boundary and query dimensions")
    if end_logits.shape != start_logits.shape:
        raise ValueError("End logits do not match start logits")
    if boundary_mask.shape != (batch, n_boundaries) or query_mask.shape != (batch, n_queries):
        raise ValueError("Boundary or query mask has an unexpected shape")

    floor = torch.full_like(start_logits, MASK_LOGIT)
    valid_boundary = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    union_start = torch.where(valid_boundary, start_logits, floor).amax(1)
    union_end = torch.where(valid_boundary, end_logits, floor).amax(1)
    union_valid = boundary_mask & query_mask.any(-1, keepdim=True)
    _, starts, starts_valid = select_top_boundaries(union_start.unsqueeze(1), union_valid.unsqueeze(1), boundary_top_k)
    _, ends, ends_valid = select_top_boundaries(union_end.unsqueeze(1), union_valid.unsqueeze(1), boundary_top_k)
    starts, ends = starts[:, 0], ends[:, 0]
    starts_valid, ends_valid = starts_valid[:, 0], ends_valid[:, 0]
    n_starts, n_ends = starts.shape[1], ends.shape[1]
    pair_start = starts.unsqueeze(-1).expand(batch, n_starts, n_ends).reshape(batch, -1)
    pair_end = ends.unsqueeze(1).expand(batch, n_starts, n_ends).reshape(batch, -1)
    pair_valid = (
        starts_valid.unsqueeze(-1) & ends_valid.unsqueeze(1) & (ends.unsqueeze(1) > starts.unsqueeze(-1))
    ).reshape(batch, -1)

    selected_start = gather_rows(start_projection, pair_start)
    selected_end = gather_rows(end_projection, pair_end)
    compatibility = (selected_start * selected_end).sum(-1) / math.sqrt(dim)
    union_pair_score = (
        compatibility
        + union_start.gather(1, pair_start.clamp(0, n_boundaries - 1))
        + union_end.gather(1, pair_end.clamp(0, n_boundaries - 1))
    )

    quota = min(min_pool_per_query, pair_start.shape[-1])
    quota_keys = pair_start.new_zeros((batch, 0))
    quota_scores = union_pair_score.new_zeros((batch, 0))
    quota_valid = pair_valid.new_zeros((batch, 0))
    if quota:
        start_idx = pair_start.clamp(0, start_logits.shape[2] - 1).unsqueeze(1).expand(batch, n_queries, -1)
        end_idx = pair_end.clamp(0, end_logits.shape[2] - 1).unsqueeze(1).expand(batch, n_queries, -1)
        per_query = start_logits.gather(2, start_idx) + end_logits.gather(2, end_idx) + compatibility.unsqueeze(1)
        per_query_valid = pair_valid.unsqueeze(1) & query_mask.unsqueeze(-1)
        ranked = torch.argsort(
            per_query.masked_fill(~per_query_valid, MASK_LOGIT),
            dim=-1,
            descending=True,
            stable=True,
        )[..., :quota]
        quota_start = start_idx.gather(-1, ranked)
        quota_end = end_idx.gather(-1, ranked)
        quota_valid = per_query_valid.gather(-1, ranked).reshape(batch, -1)
        quota_keys = (quota_start * n_boundaries + quota_end).reshape(batch, -1)
        rank_bonus = torch.arange(quota, 0, -1, device=start_projection.device, dtype=union_pair_score.dtype)
        quota_scores = (
            union_pair_score.new_full((batch, n_queries, quota), -MASK_LOGIT * 0.5) + rank_bonus.view(1, 1, quota)
        ).reshape(batch, -1)

    global_keys = pair_start * n_boundaries + pair_end
    all_keys = torch.cat((quota_keys, global_keys), -1)
    all_scores = torch.cat((quota_scores, union_pair_score.detach()), -1)
    all_valid = torch.cat((quota_valid, pair_valid), -1)
    selected_keys, selected_valid = _deduplicate_pool(all_keys, all_scores, all_valid, pool_size, n_boundaries)
    selected_keys = torch.where(selected_valid, selected_keys, torch.zeros_like(selected_keys))
    selected_s = torch.div(selected_keys, n_boundaries, rounding_mode="floor")
    selected_e = selected_keys - selected_s * n_boundaries
    indices = torch.stack((selected_s, selected_e), -1)
    indices = torch.where(selected_valid.unsqueeze(-1), indices, torch.zeros_like(indices))

    gathered_start = gather_rows(start_projection, selected_s)
    gathered_end = gather_rows(end_projection, selected_e)
    selected_compat = (gathered_start * gathered_end).sum(-1) / math.sqrt(dim)
    selected_score = (
        selected_compat
        + union_start.gather(1, selected_s.clamp(0, n_boundaries - 1))
        + union_end.gather(1, selected_e.clamp(0, n_boundaries - 1))
    )
    selected_score = selected_score.masked_fill(~selected_valid, MASK_LOGIT)
    selected_compat = torch.where(selected_valid, selected_compat, torch.zeros_like(selected_compat))
    return PooledCandidates(
        indices=indices,
        mask=selected_valid,
        proposal_logits=selected_score,
        gold_mask=None,
        compat_logits=selected_compat,
        stats=None,
    )
