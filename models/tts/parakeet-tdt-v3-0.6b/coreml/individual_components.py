#!/usr/bin/env python3
"""Export Parakeet TDT v3 RNNT components into CoreML and validate outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import  Optional, Tuple

import coremltools as ct
import torch


@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    deployment_target: Optional[ct.target.iOS17]
    compute_precision: Optional[ct.precision]
    max_audio_seconds: float
    max_symbol_steps: int


@dataclass
class ValidationSettings:
    audio_path: Optional[Path]
    seconds: float
    seed: Optional[int]
    rtol: float
    atol: float
    skip: bool


@dataclass
class ValidationDiff:
    name: str
    max_abs_diff: float
    max_rel_diff: float


@dataclass
class ValidationResult:
    source: str
    audio_num_samples: int
    audio_seconds: float
    token_length: int
    atol: float
    rtol: float
    diffs: Tuple[ValidationDiff, ...]


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
        # Transpose to match what projection layers expect
        encoder_outputs = encoder_outputs.transpose(1, 2)  # [B, T, D]
        decoder_outputs = decoder_outputs.transpose(1, 2)  # [B, U, D]

        # Apply projections
        enc_proj = self.module.enc(encoder_outputs)        # [B, T, 640]
        dec_proj = self.module.pred(decoder_outputs)       # [B, U, 640]

        # Explicit broadcasting along T and U to avoid converter ambiguity
        x = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)  # [B, T, U, 640]
        x = self.module.joint_net[0](x)                   # ReLU
        x = self.module.joint_net[1](x)                   # Dropout (no-op in eval)
        out = self.module.joint_net[2](x)                 # Linear -> logits [B, T, U, 8198]
        return out


class MelEncoderWrapper(torch.nn.Module):
    """Fused wrapper: waveform -> mel -> encoder.

    Inputs:
      - audio_signal: [B, S]
      - audio_length: [B]

    Outputs:
      - encoder: [B, D, T_enc]
      - encoder_length: [B]
    """
    def __init__(self, preprocessor: PreprocessorWrapper, encoder: EncoderWrapper) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, audio_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.preprocessor(audio_signal, audio_length)
        encoded, enc_len = self.encoder(mel, mel_length.to(dtype=torch.int32))
        return encoded, enc_len


class JointDecisionWrapper(torch.nn.Module):
    """Joint + decision head: outputs label id, label prob, duration frames.

    Splits joint logits into token logits and duration logits, applies softmax
    over tokens, argmax for both heads, and gathers probability of the chosen token.

    Inputs:
      - encoder_outputs: [B, D, T]
      - decoder_outputs: [B, D, U]

    Returns:
      - token_id: [B, T, U] int32
      - token_prob: [B, T, U] float32
      - duration: [B, T, U] int32  (frames; for v3 bins=[0,1,2,3,4])
    """
    def __init__(self, joint: JointWrapper, vocab_size: int, num_extra: int) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.num_extra = int(num_extra)

    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor):
        logits = self.joint(encoder_outputs, decoder_outputs)
        token_logits = logits[..., : self.vocab_with_blank]
        duration_logits = logits[..., -self.num_extra :]

        # Token selection
        token_ids = torch.argmax(token_logits, dim=-1).to(dtype=torch.int32)
        token_probs_all = torch.softmax(token_logits, dim=-1)
        # gather expects int64 (long) indices; cast only for gather
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)

        # Duration prediction (bins are identity mapping to frames for v3)
        duration = torch.argmax(duration_logits, dim=-1).to(dtype=torch.int32)
        return token_ids, token_prob, duration


class JointDecisionSingleStep(torch.nn.Module):
    """Single-step variant for streaming: encoder_step [1, 1024, 1] -> [1,1,1].

    Inputs:
      - encoder_step: [B=1, D=1024, T=1]
      - decoder_step: [B=1, D=640, U=1]

    Returns:
      - token_id: [1, 1, 1] int32
      - token_prob: [1, 1, 1] float32
      - duration: [1, 1, 1] int32
    """
    def __init__(self, joint: JointWrapper, vocab_size: int, num_extra: int) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.num_extra = int(num_extra)

    def forward(self, encoder_step: torch.Tensor, decoder_step: torch.Tensor):
        # Reuse JointWrapper which expects [B, D, T] and [B, D, U]
        logits = self.joint(encoder_step, decoder_step)  # [1, 1, 1, V+extra]
        token_logits = logits[..., : self.vocab_with_blank]
        duration_logits = logits[..., -self.num_extra :]

        token_ids = torch.argmax(token_logits, dim=-1, keepdim=False).to(dtype=torch.int32)
        token_probs_all = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)
        duration = torch.argmax(duration_logits, dim=-1, keepdim=False).to(dtype=torch.int32)
        return token_ids, token_prob, duration


def _coreml_convert(traced: torch.jit.ScriptModule, inputs, outputs, settings: ExportSettings) -> ct.models.MLModel:
    kwargs = {
        "convert_to": "mlprogram",
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": settings.compute_units,
    }
    print("Converting:", traced.__class__.__name__)
    print("Conversion kwargs:", kwargs)
    if settings.deployment_target is not None:
        kwargs["minimum_deployment_target"] = settings.deployment_target
    if settings.compute_precision is not None:
        kwargs["compute_precision"] = settings.compute_precision
    return ct.convert(traced, **kwargs)
