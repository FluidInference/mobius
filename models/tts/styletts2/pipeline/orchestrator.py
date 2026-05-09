"""End-to-end orchestrator over the decomposed pipeline stages.

Mirrors the call-order of `run_inference.make_inference_fn` exactly.
The only behavioural difference vs. the monolithic version is:

  * `ref_s` is frozen at the boundary (clone + detach + contiguous).
  * Optional `RefSGuard` snapshots `ref_s` and asserts it is unmutated
    when the call returns.
  * Intermediate tensors are returned via `StageOutputs` for parity
    introspection.

For audio-only use, `synthesize(...)` returns just the waveform.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .ref_s_guard import RefSGuard, freeze_ref_s
from .stages import (
    StageInputs,
    StageOutputs,
    decode_audio,
    phonemize_and_tokenize,
    predict_duration_and_alignment,
    predict_f0_and_noise,
    sample_and_blend_style,
    text_encode_and_bert,
)


def run_pipeline(
    *,
    model: Any,
    model_params: Any,
    sampler: Any,
    phonemizer: Any,
    cleaner: Any,
    device: str,
    inputs: StageInputs,
    noise: Optional[torch.Tensor] = None,
    guard_ref_s: bool = True,
) -> StageOutputs:
    """Run all 6 stages and return every intermediate tensor."""
    ref_s = freeze_ref_s(inputs.ref_s)

    guard = RefSGuard(ref_s) if guard_ref_s else None
    try:
        tokens, input_lengths, text_mask = phonemize_and_tokenize(
            inputs.text, phonemizer=phonemizer, cleaner=cleaner, device=device
        )
        t_en, bert_dur, d_en = text_encode_and_bert(model, tokens, input_lengths, text_mask)
        s_pred, ref, s = sample_and_blend_style(
            sampler,
            bert_dur=bert_dur,
            ref_s=ref_s,
            alpha=inputs.alpha,
            beta=inputs.beta,
            diffusion_steps=inputs.diffusion_steps,
            embedding_scale=inputs.embedding_scale,
            device=device,
            noise=noise,
        )
        d, pred_dur, pred_aln_trg = predict_duration_and_alignment(
            model, d_en=d_en, s=s, input_lengths=input_lengths, text_mask=text_mask, device=device
        )
        en, asr, f0_pred, n_pred = predict_f0_and_noise(
            model, model_params, t_en=t_en, d=d, pred_aln_trg=pred_aln_trg, s=s
        )
        waveform = decode_audio(model, asr=asr, f0_pred=f0_pred, n_pred=n_pred, ref=ref)
    finally:
        if guard is not None:
            guard.assert_unchanged()

    return StageOutputs(
        tokens=tokens,
        input_lengths=input_lengths,
        text_mask=text_mask,
        t_en=t_en,
        bert_dur=bert_dur,
        d_en=d_en,
        s_pred=s_pred,
        ref=ref,
        s=s,
        d=d,
        pred_dur=pred_dur,
        pred_aln_trg=pred_aln_trg,
        en=en,
        asr=asr,
        f0_pred=f0_pred,
        n_pred=n_pred,
        waveform=waveform,
    )


def synthesize(
    *,
    model: Any,
    model_params: Any,
    sampler: Any,
    phonemizer: Any,
    cleaner: Any,
    device: str,
    text: str,
    ref_s: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
    noise: Optional[torch.Tensor] = None,
    guard_ref_s: bool = True,
) -> np.ndarray:
    """Convenience wrapper returning only the waveform."""
    outputs = run_pipeline(
        model=model,
        model_params=model_params,
        sampler=sampler,
        phonemizer=phonemizer,
        cleaner=cleaner,
        device=device,
        inputs=StageInputs(
            text=text,
            ref_s=ref_s,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        ),
        noise=noise,
        guard_ref_s=guard_ref_s,
    )
    return outputs.waveform
