#!/usr/bin/env python3
"""CTC Beam Search decoder ported from FluidAudio Swift implementation.

Based on: Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CTC/CtcDecoder.swift
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CtcBeam:
    """A single hypothesis in CTC beam search."""
    prefix: List[int]
    p_blank: float
    p_non_blank: float

    @property
    def total_acoustic(self) -> float:
        """Combined acoustic score (log-sum-exp of blank and non-blank paths)."""
        return log_add_exp(self.p_blank, self.p_non_blank)

    @property
    def last_token(self) -> Optional[int]:
        """Last token in the prefix."""
        return self.prefix[-1] if self.prefix else None


def log_add_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf') and b == float('-inf'):
        return float('-inf')
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))


def ctc_beam_search(
    log_probs: np.ndarray,
    vocabulary: List[str],
    beam_width: int = 100,
    blank_id: int = 7000,
    token_candidates: int = 40
) -> str:
    """CTC prefix beam search decoder.

    Implements CTC beam search with corrected repeat-token handling (Graves 2006).

    Args:
        log_probs: Log-probabilities of shape [time_steps, vocab_size]
        vocabulary: List of vocabulary tokens (index -> string)
        beam_width: Number of hypotheses to maintain (default 100)
        blank_id: CTC blank token index (default 7000 for zh-CN)
        token_candidates: Number of top tokens to consider per frame (default 40)

    Returns:
        Decoded text string with SentencePiece markers replaced by spaces
    """
    if len(log_probs) == 0:
        return ""

    time_steps, vocab_size = log_probs.shape

    # Initialize with empty beam
    beams: Dict[Tuple[int, ...], CtcBeam] = {
        (): CtcBeam(prefix=[], p_blank=0.0, p_non_blank=float('-inf'))
    }

    for t in range(time_steps):
        frame = log_probs[t]
        blank_lp = frame[blank_id] if 0 <= blank_id < vocab_size else float('-inf')

        # Find top token candidates (excluding blank)
        top_tokens = np.argsort(frame)[::-1]
        top_tokens = [tok for tok in top_tokens if tok != blank_id][:token_candidates]

        new_beams: Dict[Tuple[int, ...], CtcBeam] = {}

        def merge_beam(beam: CtcBeam):
            """Merge beam into new_beams using log-sum-exp."""
            key = tuple(beam.prefix)
            if key in new_beams:
                existing = new_beams[key]
                existing.p_blank = log_add_exp(existing.p_blank, beam.p_blank)
                existing.p_non_blank = log_add_exp(existing.p_non_blank, beam.p_non_blank)
            else:
                new_beams[key] = beam

        for prefix_tuple, beam in beams.items():
            prev_total = beam.total_acoustic

            # 1. Blank extension
            blank_beam = CtcBeam(
                prefix=beam.prefix.copy(),
                p_blank=prev_total + blank_lp,
                p_non_blank=float('-inf')
            )
            merge_beam(blank_beam)

            # 2. Token extensions
            for token_id in top_tokens:
                token_lp = frame[token_id]
                is_repeat = (beam.last_token == token_id)

                if is_repeat:
                    # Repeat token: continue same beam (non-blank path)
                    same_beam = CtcBeam(
                        prefix=beam.prefix.copy(),
                        p_blank=float('-inf'),
                        p_non_blank=beam.p_non_blank + token_lp
                    )
                    merge_beam(same_beam)

                    # Repeat token after blank: start new token
                    new_beam = CtcBeam(
                        prefix=beam.prefix + [token_id],
                        p_blank=float('-inf'),
                        p_non_blank=beam.p_blank + token_lp
                    )
                    merge_beam(new_beam)
                else:
                    # Non-repeat: extend with new token
                    new_beam = CtcBeam(
                        prefix=beam.prefix + [token_id],
                        p_blank=float('-inf'),
                        p_non_blank=prev_total + token_lp
                    )
                    merge_beam(new_beam)

        # Prune to beam_width
        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1].total_acoustic, reverse=True)[:beam_width]
        )

    # Return best hypothesis
    if not beams:
        return ""

    best_prefix = max(beams.values(), key=lambda b: b.total_acoustic).prefix
    return decode_ctc_token_ids(best_prefix, vocabulary)


def decode_ctc_token_ids(token_ids: List[int], vocabulary: List[str]) -> str:
    """Decode CTC token IDs to text.

    Args:
        token_ids: List of token IDs
        vocabulary: Token vocabulary (index -> string)

    Returns:
        Decoded text with SentencePiece markers replaced by spaces
    """
    tokens = [vocabulary[i] for i in token_ids if i < len(vocabulary)]
    text = "".join(tokens)
    # Replace SentencePiece marker with space
    text = text.replace("▁", " ").strip()
    return text
