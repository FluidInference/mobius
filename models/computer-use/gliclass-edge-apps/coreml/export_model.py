"""Fixed-shape Core ML export adapter for the tuned GLiClass Edge checkpoint."""

import torch
from torch import nn
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)

MASK_VALUE = -1e4


class GLiClassExport(nn.Module):
    def __init__(self, model, length, max_options):
        super().__init__()
        inner = model.model
        encoder = inner.encoder_model
        config = encoder.config
        config._attn_implementation = "eager"
        self.length = length
        self.max_options = max_options
        self.layers = encoder.layers
        self.embeddings = encoder.embeddings
        self.final_norm = encoder.final_norm
        self.layer_types = list(config.layer_types)
        self.text_projector = inner.text_projector
        self.classes_projector = inner.classes_projector
        self.scorer = inner.scorer

        position_ids = torch.arange(length).unsqueeze(0)
        probe = torch.zeros(1, length, config.hidden_size)
        for layer_type in sorted(set(self.layer_types)):
            cos, sin = encoder.rotary_emb(probe, position_ids, layer_type)
            self.register_buffer(f"cos_{layer_type}", cos.detach().clone())
            self.register_buffer(f"sin_{layer_type}", sin.detach().clone())
        ones = torch.ones(1, length, dtype=torch.long)
        band = create_bidirectional_sliding_window_mask(
            config=config,
            inputs_embeds=probe,
            attention_mask=ones,
            allow_is_bidirectional_skip=False,
        )
        full = create_bidirectional_mask(
            config=config,
            inputs_embeds=probe,
            attention_mask=ones,
            allow_is_bidirectional_skip=False,
        )
        self.register_buffer("band_mask", self._to_additive(band, length))
        self.register_buffer("full_mask", self._to_additive(full, length))

    @staticmethod
    def _to_additive(mask, length):
        if mask is None:
            return torch.zeros(1, 1, length, length)
        if mask.dtype == torch.bool:
            return torch.where(mask, torch.zeros(()), torch.full((), MASK_VALUE)).float()
        return torch.where(mask < 0, torch.full((), MASK_VALUE), torch.zeros(())).float()

    def forward(self, input_ids, attention_mask, class_marker_map):
        pad = (1.0 - attention_mask.float()).view(1, 1, 1, self.length) * MASK_VALUE
        full = self.full_mask + pad
        band = self.band_mask + pad
        hidden = self.embeddings(input_ids=input_ids.long())
        for layer, layer_type in zip(self.layers, self.layer_types):
            hidden = layer(
                hidden,
                attention_mask=full if layer_type == "full_attention" else band,
                position_embeddings=(
                    getattr(self, f"cos_{layer_type}"),
                    getattr(self, f"sin_{layer_type}"),
                ),
            )
        hidden = self.final_norm(hidden)
        classes = torch.matmul(class_marker_map, hidden)
        text = hidden[:, 0]
        text = self.text_projector(text)
        classes = self.classes_projector(classes)
        logits = self.scorer(text, classes)
        option_mask = class_marker_map.sum(-1)
        logits = logits * option_mask + (1.0 - option_mask) * MASK_VALUE
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities
