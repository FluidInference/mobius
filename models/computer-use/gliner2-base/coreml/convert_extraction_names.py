"""Ordered tensor contract shared by extraction conversion and verification."""

FEATURE_NAMES = (
    "text_states",
    "query_states",
    "boundary_states",
    "boundary_mask",
    "start_logits",
    "end_logits",
    "inside_prefix",
    "inside_prefix_mean",
    "pool_start_projection",
    "pool_end_projection",
    "null_logits",
    "count_log_rates",
    "classification_logits",
)
SCORE_INPUT_NAMES = (
    "text_states",
    "text_mask",
    "query_states",
    "query_mask",
    "boundary_states",
    "start_logits",
    "end_logits",
    "inside_prefix",
    "inside_prefix_mean",
    "candidate_indices",
    "candidate_mask",
    "candidate_compatibility",
)
