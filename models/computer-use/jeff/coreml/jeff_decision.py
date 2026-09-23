"""The trained GLiFormer Large classification path used by Jeff.

This checkpoint's classification config uses CLS pooling, parent anchors,
no anchor refinement/normalization, linear anchor modeling, and dot scoring.
Under those exact settings the word-level RNN is computed upstream but cannot
influence classification logits. We still assert the settings at construction.
"""

from __future__ import annotations

import torch
from torch import nn


class JeffDecision(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        config = model.config.classification_config
        expected = {
            "pooling_type": "cls",
            "anchor_mode": "parent",
            "anchor_modeling": "linear",
            "anchor_normalization": "none",
            "scorer_type": "dot",
            "anchor_refine_layers": 0,
        }
        for name, value in expected.items():
            actual = getattr(config, name)
            if actual != value:
                raise ValueError(f"unsupported classification config {name}={actual!r}; expected {value!r}")
        if not config.embed_parent_token or not config.embed_cat_token:
            raise ValueError("this path requires embeddings at the parent and category marker tokens")
        if model.config.hidden_size != 1024:
            raise ValueError("this fixed classifier requires the pinned 1024-wide checkpoint")
        head = model.heads["classification"]
        if hasattr(head, "anchor_refine"):
            raise ValueError("classification anchor refinement cannot be omitted")
        if type(head.anchor_layer).__name__ != "ParentAnchorLayer":
            raise ValueError("unsupported anchor layer")
        if type(head.anchor_modeling).__name__ != "LinearAnchorModeling":
            raise ValueError("unsupported anchor model")
        self.encoder = model.token_rep_layer
        self.projection = head.anchor_modeling.proj

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        parent_position: torch.Tensor,
        category_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return unnormalized logits for one classification group, padded to C=8."""
        encoded = self.encoder(input_ids.long(), attention_mask.long())
        cls = encoded[:, 0, :]
        parent_index = parent_position.long().unsqueeze(-1).expand(1, 1, 1024)
        parent = torch.gather(encoded, 1, parent_index)
        child_index = category_positions.long().unsqueeze(-1).expand(1, 8, 1024)
        children = torch.gather(encoded, 1, child_index)
        combined = torch.cat((parent.expand(1, 8, 1024), children), dim=-1)
        fused = self.projection(combined)
        return (cls.unsqueeze(1) * fused).sum(dim=-1)


def marker_positions(
    input_ids: torch.Tensor, config, max_categories: int = 8
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Find the actual learned prompt markers in a one-row Jeff collator batch."""
    if input_ids.shape[0] != 1:
        raise ValueError("one classification group per call is required")
    parent = torch.nonzero(input_ids[0] == config.classification_config.parent_token_index).flatten()
    children = torch.nonzero(input_ids[0] == config.classification_config.cat_token_index).flatten()
    if parent.numel() != 1 or not (1 <= children.numel() <= max_categories):
        raise ValueError(
            f"expected 1 parent and 1..{max_categories} categories; got {parent.numel()}, {children.numel()}"
        )
    count = int(children.numel())
    category_positions = torch.zeros((1, max_categories), dtype=torch.int32)
    category_positions[0, :count] = children.to(torch.int32)
    return parent.to(torch.int32).view(1, 1), category_positions, count
