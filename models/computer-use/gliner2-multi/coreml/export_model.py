"""Native GLiNER2 classification path with explicit marker routing."""
import torch
from torch import nn
from transformers.models.deberta_v2 import modeling_deberta_v2


def coreml_safe_attention_forward(
    self, hidden_states, attention_mask, output_attentions=False,
    query_states=None, relative_pos=None, rel_embeddings=None,
):
    """Native DeBERTa attention with a finite mask sentinel for FP16 Core ML."""
    if query_states is None:
        query_states = hidden_states
    query = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    key = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    value = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    factor = 1 + int("c2p" in self.pos_att_type) + int("p2c" in self.pos_att_type)
    scale = modeling_deberta_v2.scaled_size_sqrt(query, factor)
    scores = torch.bmm(query, key.transpose(-1, -2) / scale.to(dtype=query.dtype))
    if self.relative_attention:
        relative = self.disentangled_attention_bias(
            query, key, relative_pos, self.pos_dropout(rel_embeddings), factor
        )
        scores = scores + relative
    scores = scores.view(-1, self.num_attention_heads, scores.size(-2), scores.size(-1))
    scores = scores.masked_fill(~attention_mask.bool(), -1e4)
    probabilities = self.dropout(torch.softmax(scores, dim=-1))
    context = torch.bmm(probabilities.view(-1, probabilities.size(-2), probabilities.size(-1)), value)
    context = context.view(-1, self.num_attention_heads, context.size(-2), context.size(-1))
    context = context.permute(0, 2, 1, 3).contiguous()
    context = context.view(context.size()[:-2] + (-1,))
    return (context, probabilities) if output_attentions else (context, None)

class GLiNER2ClassificationExport(nn.Module):
    def __init__(self, native: nn.Module):
        super().__init__()
        self.encoder = native.encoder
        self.classifier = native.classifier
        self.temperature = float(native.boundary_settings.classification_temperature)

    def forward(self, input_ids, attention_mask, marker_indices, marker_mask):
        hidden = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        indices = marker_indices.long().unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        states = hidden.gather(1, indices)
        logits = self.classifier(states).squeeze(-1) / self.temperature
        logits = torch.where(marker_mask > 0.5, logits, torch.full_like(logits, -1e4))
        return logits, torch.softmax(logits, dim=-1)
