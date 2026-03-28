#!/usr/bin/env python3
"""Wrapper modules for exporting hybrid TDT-CTC model components to CoreML.

Adapted from the pure-TDT 0.6B export pipeline to support
EncDecHybridRNNTCTCBPEModel (parakeet-tdt_ctc-110m and similar).

The hybrid model exposes the same .preprocessor, .encoder, .decoder (LSTM
prediction network), and .joint (joint network with TDT duration outputs)
as a pure TDT model.  The only loading difference is the NeMo model class.
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
    deployment_target: Optional[ct.target.iOS17]
    compute_precision: Optional[ct.precision]
    max_audio_seconds: float
    max_symbol_steps: int


# ---------------------------------------------------------------------------
# Component wrappers – thin shims that normalise NeMo I/O for tracing
# ---------------------------------------------------------------------------

class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, audio_signal: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.module(
            input_signal=audio_signal, length=length.to(dtype=torch.long)
        )
        return mel, mel_length


class EncoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, features: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, encoded_lengths = self.module(
            audio_signal=features, length=length.to(dtype=torch.long)
        )
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

    def forward(
        self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        # Input: encoder_outputs [B, D, T], decoder_outputs [B, D, U]
        # Transpose to [B, T, D] / [B, U, D] for projection layers
        encoder_outputs = encoder_outputs.transpose(1, 2)
        decoder_outputs = decoder_outputs.transpose(1, 2)

        enc_proj = self.module.enc(encoder_outputs)
        dec_proj = self.module.pred(decoder_outputs)

        # Explicit broadcasting along T and U
        x = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)  # [B, T, U, D_joint]
        x = self.module.joint_net[0](x)   # ReLU
        x = self.module.joint_net[1](x)   # Dropout (no-op in eval)
        out = self.module.joint_net[2](x)  # Linear -> logits
        return out


class CTCDecoderWrapper(torch.nn.Module):
    """CTC decoder head: encoder -> log-probabilities."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.module(encoder_output=encoder_output)


class MelEncoderWrapper(torch.nn.Module):
    """Fused waveform -> mel -> encoder."""

    def __init__(
        self, preprocessor: PreprocessorWrapper, encoder: EncoderWrapper
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(
        self, audio_signal: torch.Tensor, audio_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.preprocessor(audio_signal, audio_length)
        encoded, enc_len = self.encoder(mel, mel_length.to(dtype=torch.int32))
        return encoded, enc_len


class MelEncoderCtcWrapper(torch.nn.Module):
    """Fused waveform -> mel -> encoder + CTC logits.

    Outputs both TDT encoder features and CTC logits for unified custom vocabulary.
    This eliminates the need for a separate CTC encoder model.
    """

    def __init__(
        self,
        preprocessor: PreprocessorWrapper,
        encoder: EncoderWrapper,
        ctc_decoder: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.ctc_decoder = ctc_decoder

    def forward(
        self, audio_signal: torch.Tensor, audio_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel, mel_length = self.preprocessor(audio_signal, audio_length)
        encoded, enc_len = self.encoder(mel, mel_length.to(dtype=torch.int32))
        ctc_logits = self.ctc_decoder(encoder_output=encoded)
        return encoded, enc_len, ctc_logits


class JointDecisionWrapper(torch.nn.Module):
    """Joint + decision head: token_id, token_prob, duration.

    Splits joint logits into token logits ([..., :vocab+1]) and duration
    logits ([..., -num_extra:]), applies softmax / argmax.
    """

    def __init__(
        self, joint: JointWrapper, vocab_size: int, num_extra: int
    ) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.num_extra = int(num_extra)

    def forward(
        self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor
    ):
        logits = self.joint(encoder_outputs, decoder_outputs)
        token_logits = logits[..., : self.vocab_with_blank]
        duration_logits = logits[..., self.vocab_with_blank :]

        token_ids = torch.argmax(token_logits, dim=-1).to(dtype=torch.int32)
        token_probs_all = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)

        if self.num_extra > 0:
            duration = torch.argmax(duration_logits, dim=-1).to(dtype=torch.int32)
        else:
            duration = torch.zeros_like(token_ids)
        return token_ids, token_prob, duration


class JointDecisionSingleStep(torch.nn.Module):
    """Single-step variant for streaming: [1, D, 1] -> scalars."""

    def __init__(
        self,
        joint: JointWrapper,
        vocab_size: int,
        num_extra: int,
        top_k: int = 64,
    ) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.num_extra = int(num_extra)
        self.top_k = int(top_k)

    def forward(
        self, encoder_step: torch.Tensor, decoder_step: torch.Tensor
    ):
        logits = self.joint(encoder_step, decoder_step)  # [1, 1, 1, V+extra]
        token_logits = logits[..., : self.vocab_with_blank]
        duration_logits = logits[..., self.vocab_with_blank :]

        token_ids = torch.argmax(token_logits, dim=-1, keepdim=False).to(
            dtype=torch.int32
        )
        token_probs_all = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)
        if self.num_extra > 0:
            duration = torch.argmax(duration_logits, dim=-1, keepdim=False).to(
                dtype=torch.int32
            )
        else:
            duration = torch.zeros_like(token_ids)

        topk_logits, topk_ids_long = torch.topk(
            token_logits,
            k=min(self.top_k, token_logits.shape[-1]),
            dim=-1,
        )
        topk_ids = topk_ids_long.to(dtype=torch.int32)
        return token_ids, token_prob, duration, topk_ids, topk_logits


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def _coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs,
    outputs,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
) -> ct.models.MLModel:
    cu = (
        compute_units_override
        if compute_units_override is not None
        else settings.compute_units
    )
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
