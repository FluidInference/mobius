"""Host-side port of AlignmentStreamAnalyzer consuming exported attention rows.

Upstream (chatterbox/models/t3/inference/alignment_stream_analyzer.py) hooks
three Llama attention heads and reads their softmax rows. The CoreML T3 models
emit those rows as the `align_attn` output ([3, 2, max_len] from prefill —
both BOS query rows — and [3, max_len] per decode step). This class replays
the exact upstream step() logic on those rows; it is also the spec for the
future Swift port.
"""
from __future__ import annotations

import torch


class AlignmentAnalyzerPort:
    def __init__(self, text_tokens_slice: tuple[int, int], eos_idx: int):
        self.i, self.j = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, self.j - self.i)
        self.curr_frame_pos = 0
        self.text_position = 0
        self.started = False
        self.started_at = None
        self.complete = False
        self.completed_at = None
        self.generated_tokens: list[int] = []

    def step(self, logits: torch.Tensor, align_rows: torch.Tensor,
             next_token: int | None = None) -> torch.Tensor:
        """logits: (1, V) CFG-combined; align_rows: (3, ctx) or (3, R, ctx)."""
        aligned_attn = align_rows.mean(dim=0)          # (ctx,) or (R, ctx)
        if aligned_attn.dim() == 1:
            aligned_attn = aligned_attn.unsqueeze(0)   # (R=1, ctx)
        A_chunk = aligned_attn[:, self.i:self.j].clone()  # (R, S)
        # upstream: A_chunk[:, self.curr_frame_pos + 1:] = 0
        A_chunk[:, self.curr_frame_pos + 1:] = 0

        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)
        A = self.alignment
        T, S = A.shape

        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 5)
        alignment_repetition = self.complete and (
            A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)

        if next_token is not None:
            self.generated_tokens.append(next_token)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]
        token_repetition = (len(self.generated_tokens) >= 3
                            and len(set(self.generated_tokens[-2:])) == 1)

        if cur_text_posn < S - 3 and S > 5:
            logits[..., self.eos_idx] = -2**15

        if long_tail or alignment_repetition or token_repetition:
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        # Upstream increments by 1 per step() call even though the first
        # chunk contributes two rows — replicate exactly.
        self.curr_frame_pos += 1
        return logits
