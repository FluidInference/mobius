#!/usr/bin/env python3
"""Export Parakeet TDT+CTC Japanese components into CoreML."""
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


class PreprocessorWrapper(torch.nn.Module):
    """Preprocessor: audio waveform -> mel spectrogram.

    Input:  audio_signal [B, S] - raw audio samples
            length [B] - number of samples per batch item
    Output: mel_features [B, 80, T] - mel spectrogram
            mel_length [B] - number of time frames
    """
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
    """Encoder: mel spectrogram -> encoder features.

    Input:  features [B, 80, T] - mel spectrogram
            length [B] - number of time frames
    Output: encoded [B, 1024, T'] - encoder features (downsampled)
            encoded_length [B] - encoder output length
    """
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


class CTCDecoderWrapper(torch.nn.Module):
    """CTC decoder head: encoder features -> CTC logits (RAW, not log-softmax).

    Input:  encoder_output [B, 1024, T] - encoder features
    Output: logits [B, T, V+1] - Raw CTC logits (apply log_softmax in post-processing)

    NOTE: Bypasses the CTC decoder's forward() which applies log_softmax.
          We only use decoder_layers (Conv1d) + transpose to avoid CoreML
          conversion issues with log_softmax.
    """
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # Bypass the CTC decoder's forward() to avoid log_softmax
        # The decoder.forward() does: log_softmax(decoder_layers(x).transpose(1,2))
        # We only want: decoder_layers(x).transpose(1,2) (raw logits)
        conv_output = self.module.decoder_layers(encoder_output)  # [B, V, T]
        logits = conv_output.transpose(1, 2)  # [B, T, V]
        return logits


class MelEncoderWrapper(torch.nn.Module):
    """Fused wrapper: audio waveform -> encoder features.

    Input:  audio_signal [B, S] - raw audio samples
            audio_length [B] - number of samples
    Output: encoder [B, 1024, T] - encoder features
            encoder_length [B] - encoder output length
    """
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


class MelEncoderCTCWrapper(torch.nn.Module):
    """Fully fused wrapper: audio waveform -> CTC logits (RAW).

    Input:  audio_signal [B, S] - raw audio samples
            audio_length [B] - number of samples
    Output: logits [B, T, V+1] - Raw CTC logits (apply log_softmax in post-processing)
            encoder_length [B] - encoder output length

    NOTE: Uses CTCDecoderWrapper which bypasses log_softmax to avoid CoreML conversion issues.
    """
    def __init__(
        self,
        preprocessor: PreprocessorWrapper,
        encoder: EncoderWrapper,
        ctc_decoder: CTCDecoderWrapper,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.ctc_decoder = ctc_decoder

    def forward(
        self, audio_signal: torch.Tensor, audio_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.preprocessor(audio_signal, audio_length)
        encoded, enc_len = self.encoder(mel, mel_length.to(dtype=torch.int32))
        logits = self.ctc_decoder(encoded)  # Uses raw logits (no log_softmax)
        return logits, enc_len


def _coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs,
    outputs,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
) -> ct.models.MLModel:
    """Convert traced PyTorch model to CoreML."""
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
