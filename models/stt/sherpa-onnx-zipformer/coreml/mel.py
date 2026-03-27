#!/usr/bin/env python3
"""Compute 80-dim log-mel filterbank features matching icefall/kaldifeat defaults.

This replicates the feature extraction used by sherpa-onnx Zipformer2 models:
    - 16 kHz sample rate
    - 25 ms window, 10 ms hop (400 / 160 samples)
    - 80 mel bins, Hann window
    - log energy (ln), no dithering

Usage:
    from mel import compute_fbank_features
    features = compute_fbank_features("audio.wav")  # -> (T, 80) numpy array
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio


SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160  # 10 ms
WIN_LENGTH = 400  # 25 ms
N_MELS = 80
FMIN = 20.0
FMAX = -400.0  # kaldifeat convention: negative means Nyquist - |fmax|
# Effective fmax = 8000 - 400 = 7600 Hz for 16 kHz SR


def compute_fbank_features(
    audio: Union[str, Path, np.ndarray, torch.Tensor],
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Compute log-mel filterbank features.

    Args:
        audio: Path to a WAV file, or a 1-D waveform array/tensor (float32, mono).
        sample_rate: Expected sample rate (resamples if needed).
        n_mels: Number of mel bins.

    Returns:
        (T, n_mels) float32 numpy array of log-mel features.
    """
    if isinstance(audio, (str, Path)):
        waveform, sr = torchaudio.load(str(audio))
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        waveform = waveform[0]  # mono
    elif isinstance(audio, np.ndarray):
        waveform = torch.from_numpy(audio).float()
    else:
        waveform = audio.float()

    if waveform.dim() > 1:
        waveform = waveform[0]

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        sample_frequency=sample_rate,
        num_mel_bins=n_mels,
        frame_length=WIN_LENGTH / sample_rate * 1000,  # 25.0 ms
        frame_shift=HOP_LENGTH / sample_rate * 1000,    # 10.0 ms
        window_type="povey",
        dither=0.0,
        energy_floor=1.0,
        snip_edges=False,
    )

    return fbank.numpy()  # (T, 80)
