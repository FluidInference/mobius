"""Tensor-only export adapter around the complete upstream trained network."""

import torch
from torch import nn


class ExportScorer(nn.Module):
    """Retain both context layers, the option encoder, pooling, and attention head."""

    def __init__(self, reference: nn.Module):
        super().__init__()
        self.model = reference

    def forward(self, context_ids, option_ids, option_mask):
        live_options = option_mask != 0
        context_mask = context_ids != 0
        # Equivalent to upstream mask[:, 0] = True, without bool tensor mutation,
        # which coremltools cannot lower. Embeddings still use the original IDs.
        safe_context_ids = torch.cat((torch.ones_like(context_ids[:, :1]), context_ids[:, 1:]), dim=1)
        context = self.model.encoder(self.model._embed(context_ids), src_key_padding_mask=safe_context_ids == 0)
        batch_size, option_count, token_count = option_ids.shape
        flat_ids = option_ids.reshape(batch_size * option_count, token_count)
        flat_mask = flat_ids != 0
        safe_ids = torch.cat((torch.ones_like(flat_ids[:, :1]), flat_ids[:, 1:]), dim=1)
        hidden = self.model.option_encoder(self.model._embed(flat_ids), src_key_padding_mask=safe_ids == 0)
        weights = flat_mask.unsqueeze(-1).float()
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        options = pooled.reshape(batch_size, option_count, -1)
        logits = self.model.head(context, context_mask, options, live_options)
        # Keep padded output logits finite in FP16. Live option logits are unchanged.
        logits = torch.where(live_options, logits, torch.full_like(logits, -10000.0))
        return logits, logits.softmax(-1)
