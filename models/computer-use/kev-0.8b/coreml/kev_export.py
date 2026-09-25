"""Kev (Qwen3.5 backbone + pointer head) as one fixed-length, prefill-only question row.

A Kev question row is the state followed by one question branch, run as plain causal text (Kev's row form, the
only form its Qwen3.5 backbones support). The pointer head reads the final-normed hidden state at `<decide>` and at
each option's `</opt>`; one-hot maps select those rows so the graph needs no data-dependent gather.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from qwen35_export import DecoderChunk, RMSNorm, TextConfig, rope_cos_sin

MASK_VALUE = -1e4


class KevRow(nn.Module):
    def __init__(self, cfg: TextConfig, seq_len: int, max_options: int, pointer_dim: int = 256, chunk_size: int = 64):
        super().__init__()
        self.decoder = DecoderChunk(cfg, 0, cfg.num_layers, seq_len, chunk_size=chunk_size, with_head=False)
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps)
        self.q = nn.Linear(cfg.hidden_size, pointer_dim)
        self.k = nn.Linear(cfg.hidden_size, pointer_dim)
        self.max_options = max_options
        self.register_buffer("pointer_scale", torch.tensor(pointer_dim ** -0.5))
        self.register_buffer("inverse_temperature", torch.tensor(1.0))

    def forward(self, hidden, cos, sin, decide_onehot, option_onehot, option_mask):
        """hidden [1, L, D] (host embedding gather), cos/sin [L, R], decide_onehot [1, L], option_onehot [K, L],
        option_mask [K] (1 = real option) -> logits [K] (temperature applied), probabilities [K]."""
        states = self.decoder(hidden, cos, sin)[0]
        decide = self.norm(torch.matmul(decide_onehot, states))          # [1, D]
        options = self.norm(torch.matmul(option_onehot, states))         # [K, D]
        logits = (self.k(options) * self.q(decide)).sum(-1) * self.pointer_scale * self.inverse_temperature
        logits = torch.where(option_mask > 0.5, logits, torch.full_like(logits, MASK_VALUE))
        return logits, torch.softmax(logits, dim=-1)


def load_kev_row(merged: Path, seq_len: int, max_options: int, chunk_size: int = 64):
    meta = json.loads((merged / "meta.json").read_text())
    cfg = TextConfig(meta["text_config"])
    row = KevRow(cfg, seq_len, max_options, chunk_size=chunk_size)
    state = load_file(str(merged / "text.safetensors"))
    row.decoder.load_merged(state)
    row.norm.load_state_dict({"weight": state["norm.weight"]})
    head = load_file(str(merged / "head.safetensors"))
    row.q.load_state_dict({"weight": head["q.weight"], "bias": head["q.bias"]})
    row.k.load_state_dict({"weight": head["k.weight"], "bias": head["k.bias"]})
    row.pointer_scale.fill_(meta["pointer_scale"])
    row.inverse_temperature.fill_(1.0 / meta["temperature"])
    return row.eval(), cfg, meta, state["embed_tokens.weight"]


def row_inputs(cfg: TextConfig, embed: torch.Tensor, ids, pos, decide: int, options, seq_len: int, max_options: int,
               pad_id: int):
    """Right-pad one causal row to `seq_len` and build the export inputs. Padding follows every real token, and the
    model is causal with a forward-scan recurrence, so it cannot change any real token's state."""
    n = len(ids)
    if n > seq_len or len(options) > max_options:
        raise ValueError(f"row needs {n} tokens / {len(options)} options; bucket holds {seq_len} / {max_options}")
    token_ids = torch.full((seq_len,), pad_id, dtype=torch.long)
    token_ids[:n] = torch.tensor(ids)
    positions = torch.arange(seq_len, dtype=torch.long)
    positions[:n] = torch.tensor(pos)
    cos, sin = rope_cos_sin(cfg, positions.unsqueeze(0).expand(3, -1))
    decide_onehot = torch.zeros(1, seq_len)
    decide_onehot[0, decide] = 1
    option_onehot = torch.zeros(max_options, seq_len)
    option_mask = torch.zeros(max_options)
    for i, index in enumerate(options):
        option_onehot[i, index] = 1
        option_mask[i] = 1
    return embed[token_ids].unsqueeze(0), cos, sin, decide_onehot, option_onehot, option_mask
