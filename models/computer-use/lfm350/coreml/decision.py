"""Pinned RLCD checkpoint loader and static candidate-scoring graph."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCE_REPO = "notnotsamuel/LFM2.5-350M-RLCD"
SOURCE_REVISION = "deb589d803d141cabd158ef55f6617b128529f36"


def source_path() -> Path:
    """Return the exact trained-weight snapshot, downloading it if needed."""
    return Path(snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION))


def load_native(device: str = "cpu"):
    path = source_path()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager")
    model.to(device).eval().requires_grad_(False)
    return tokenizer, model


class CandidateScorer(torch.nn.Module):
    """Static B x L LFM graph plus full-value log-likelihood reduction."""

    def __init__(self, model: torch.nn.Module, length: int, stable_logprob: bool = False):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head
        self.hidden_size = model.config.hidden_size
        self.stable_logprob = stable_logprob
        self.register_buffer("causal", torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0))

    def forward(self, input_ids, attention_mask, value_positions, value_targets, value_mask):
        valid_keys = attention_mask[:, None, None, :].float()
        full_attention = (1.0 - self.causal * valid_keys) * -10000.0
        hidden = self.model(
            input_ids=input_ids.long(),
            attention_mask={"full_attention": full_attention, "conv": attention_mask.long()},
            use_cache=False,
        )
        states = hidden.last_hidden_state
        gather_index = value_positions.long().unsqueeze(-1).expand(-1, -1, self.hidden_size)
        selected_states = torch.gather(states, 1, gather_index)
        logits = self.lm_head(selected_states).float()
        targets = value_targets.long().unsqueeze(-1)
        if self.stable_logprob:
            chosen_logits = torch.gather(logits, -1, targets).squeeze(-1)
            chosen = chosen_logits - torch.logsumexp(logits, dim=-1)
        else:
            log_probabilities = torch.log_softmax(logits, dim=-1)
            chosen = torch.gather(log_probabilities, -1, targets).squeeze(-1)
        return (chosen * value_mask).sum(dim=-1)
