"""Decomposed inference stages for StyleTTS2.

Single source of truth: every stage here calls into the *same* model
components and helpers loaded by `run_inference.load_styletts2`. There
is no separate model construction, no separate text cleaner, no separate
sampler. This module only carves the monolithic
`run_inference.make_inference_fn` into named stages so they can be
exercised individually for parity testing and (later) traced for CoreML.

Stage map (mirrors run_inference.make_inference_fn line-by-line):

    1. phonemize_and_tokenize:       text  ->  tokens, input_lengths, text_mask
    2. text_encode_and_bert:         tokens -> t_en, bert_dur, d_en
    3. sample_and_blend_style:       (bert_dur, ref_s, noise) -> ref, s
    4. predict_duration_and_alignment: (d_en, s, mask, lens) -> pred_aln_trg, d
    5. predict_f0_and_noise:         (t_en, d, pred_aln_trg, s) -> en, asr, f0_pred, n_pred
    6. decode_audio:                 (asr, f0_pred, n_pred, ref) -> waveform[np.ndarray]

The `StageInputs` / `StageOutputs` dataclasses make the data contract
between stages explicit so the orchestrator (and CoreML stage tracer)
cannot accidentally drop a field.

`ref_s` is *never* mutated by these stages: blending always produces a
new tensor via out-of-place ops, and the caller is expected to pass a
`freeze_ref_s(...)` copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch

# Reuse run_inference's helpers verbatim — single source of truth.
from run_inference import length_to_mask


@dataclass
class StageInputs:
    text: str
    ref_s: torch.Tensor          # [1, 256], frozen by caller
    alpha: float = 0.3
    beta: float = 0.7
    diffusion_steps: int = 5
    embedding_scale: float = 1.0


@dataclass
class StageOutputs:
    """All intermediate tensors a CoreML conversion would need to capture."""

    tokens: torch.Tensor
    input_lengths: torch.Tensor
    text_mask: torch.Tensor
    t_en: torch.Tensor
    bert_dur: torch.Tensor
    d_en: torch.Tensor
    s_pred: torch.Tensor
    ref: torch.Tensor
    s: torch.Tensor
    d: torch.Tensor
    pred_dur: torch.Tensor
    pred_aln_trg: torch.Tensor
    en: torch.Tensor
    asr: torch.Tensor
    f0_pred: torch.Tensor
    n_pred: torch.Tensor
    waveform: np.ndarray


def phonemize_and_tokenize(
    text: str,
    *,
    phonemizer: Any,
    cleaner: Any,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage 1 — text -> token ids, lengths, mask.

    Equivalent to lines 185–194 of run_inference.make_inference_fn.
    """
    from nltk.tokenize import word_tokenize

    text = text.strip()
    ps = phonemizer.phonemize([text])
    ps = " ".join(word_tokenize(ps[0]))
    tokens = cleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    return tokens, input_lengths, text_mask


def text_encode_and_bert(
    model: Any,
    tokens: torch.Tensor,
    input_lengths: torch.Tensor,
    text_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage 2 — text encoder + PL-BERT + bert_encoder.

    Equivalent to lines 196–198.
    """
    with torch.no_grad():
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    return t_en, bert_dur, d_en


def sample_and_blend_style(
    sampler: Any,
    *,
    bert_dur: torch.Tensor,
    ref_s: torch.Tensor,
    alpha: float,
    beta: float,
    diffusion_steps: int,
    embedding_scale: float,
    device: str,
    noise: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage 3 — diffusion-sampled style + alpha/beta blend with ref_s.

    Equivalent to lines 200–211.

    `ref_s` is treated as immutable: we slice into views but never assign
    back into them, and the blend uses out-of-place arithmetic so the
    caller's tensor is bit-for-bit unchanged on return.
    """
    with torch.no_grad():
        if noise is None:
            noise = torch.randn((1, 256)).unsqueeze(1).to(device)
        s_pred = sampler(
            noise=noise,
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        # Out-of-place blend; ref_s is read-only here.
        s_diff = s_pred[:, 128:]
        ref_diff = s_pred[:, :128]
        ref = alpha * ref_diff + (1.0 - alpha) * ref_s[:, :128]
        s = beta * s_diff + (1.0 - beta) * ref_s[:, 128:]
    return s_pred, ref, s


def predict_duration_and_alignment(
    model: Any,
    *,
    d_en: torch.Tensor,
    s: torch.Tensor,
    input_lengths: torch.Tensor,
    text_mask: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage 4 — predictor.text_encoder + LSTM + duration head + alignment.

    Equivalent to lines 213–223. Returns `(d, pred_dur, pred_aln_trg)`.
    """
    with torch.no_grad():
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().item()))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].item())] = 1
            c_frame += int(pred_dur[i].item())
    return d, pred_dur, pred_aln_trg.to(device)


def predict_f0_and_noise(
    model: Any,
    model_params: Any,
    *,
    t_en: torch.Tensor,
    d: torch.Tensor,
    pred_aln_trg: torch.Tensor,
    s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage 5 — F0/noise prediction with hifigan shift quirk.

    Equivalent to lines 225–239. Returns `(en, asr, f0_pred, n_pred)`.
    """
    with torch.no_grad():
        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        f0_pred, n_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new
    return en, asr, f0_pred, n_pred


def decode_audio(
    model: Any,
    *,
    asr: torch.Tensor,
    f0_pred: torch.Tensor,
    n_pred: torch.Tensor,
    ref: torch.Tensor,
) -> np.ndarray:
    """Stage 6 — HiFi-GAN decoder + tail trim.

    Equivalent to lines 241–244.
    """
    with torch.no_grad():
        out = model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))
    # Repo notes: trim weird tail pulse.
    return out.squeeze().cpu().numpy()[..., :-50]
