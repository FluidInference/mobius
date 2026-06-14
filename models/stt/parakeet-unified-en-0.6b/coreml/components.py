#!/usr/bin/env python3
"""Torch wrappers for exporting parakeet-unified-en-0.6b components to CoreML.

Adapted from models/stt/parakeet-tdt-v3-0.6b/coreml/individual_components.py.
Differences from the TDT v3 pipeline:
  - Plain RNNT joint: no duration head (num_extra_outputs == 0), so the
    decision wrappers emit token_id/token_prob/top-k only.
  - The encoder supports two trace modes: offline (full attention,
    att_context_size=[-1,-1,-1]) and streaming (chunked_limited_with_rc mask
    with a fixed [left, chunk, right] context, baked in at trace time).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import torch


@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    deployment_target: Optional[object]
    compute_precision: Optional[object]
    max_audio_seconds: float
    max_symbol_steps: int


class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.module(input_signal=audio_signal, length=length.to(dtype=torch.long))
        return mel, mel_length


class EncoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, features: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, encoded_lengths = self.module(audio_signal=features, length=length.to(dtype=torch.long))
        return encoded, encoded_lengths


class MelEncoderWrapper(torch.nn.Module):
    """Fused waveform -> mel -> encoder."""

    def __init__(self, preprocessor: PreprocessorWrapper, encoder: EncoderWrapper) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, audio_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.preprocessor(audio_signal, audio_length)
        encoded, enc_len = self.encoder(mel, mel_length.to(dtype=torch.int32))
        return encoded, enc_len


class DecoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = [h_in, c_in]
        decoder_output, _, new_state = self.module(
            targets=targets.to(dtype=torch.long),
            target_length=target_lengths.to(dtype=torch.long),
            states=state,
        )
        return decoder_output, new_state[0], new_state[1]


class JointWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor) -> torch.Tensor:
        # Input: encoder_outputs [B, D, T], decoder_outputs [B, D, U]
        encoder_outputs = encoder_outputs.transpose(1, 2)  # [B, T, D]
        decoder_outputs = decoder_outputs.transpose(1, 2)  # [B, U, D]

        enc_proj = self.module.enc(encoder_outputs)  # [B, T, 640]
        dec_proj = self.module.pred(decoder_outputs)  # [B, U, 640]

        # Explicit broadcasting along T and U to avoid converter ambiguity
        x = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)  # [B, T, U, 640]
        x = self.module.joint_net[0](x)  # ReLU
        x = self.module.joint_net[1](x)  # Dropout (no-op in eval)
        out = self.module.joint_net[2](x)  # Linear -> logits [B, T, U, V+1]
        return out


class JointDecisionSingleStep(torch.nn.Module):
    """Single-step joint + decision for the greedy RNNT loop.

    Inputs:
      - encoder_step: [1, 1024, 1]
      - decoder_step: [1, 640, 1]

    Returns:
      - token_id: [1, 1, 1] int32 (argmax over vocab+blank)
      - token_prob: [1, 1, 1] float32
      - top_k_ids: [1, 1, 1, K] int32
      - top_k_logits: [1, 1, 1, K] float32
    """

    def __init__(self, joint: JointWrapper, vocab_size: int, top_k: int = 64) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.top_k = int(top_k)

    def forward(self, encoder_step: torch.Tensor, decoder_step: torch.Tensor):
        logits = self.joint(encoder_step, decoder_step)  # [1, 1, 1, V+1]
        token_logits = logits[..., : self.vocab_with_blank]

        token_ids = torch.argmax(token_logits, dim=-1).to(dtype=torch.int32)
        token_probs_all = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)

        topk_logits, topk_ids_long = torch.topk(
            token_logits, k=min(self.top_k, token_logits.shape[-1]), dim=-1
        )
        topk_ids = topk_ids_long.to(dtype=torch.int32)
        return token_ids, token_prob, topk_ids, topk_logits


def coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs,
    outputs,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
) -> ct.models.MLModel:
    cu = compute_units_override if compute_units_override is not None else settings.compute_units
    kwargs = {
        "convert_to": "mlprogram",
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": cu,
    }
    if settings.deployment_target is not None:
        kwargs["minimum_deployment_target"] = settings.deployment_target
    if settings.compute_precision is not None:
        kwargs["compute_precision"] = settings.compute_precision
    return ct.convert(traced, **kwargs)
