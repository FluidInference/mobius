"""Corrected Cohere Transcribe mel spectrogram preprocessing.

This is a numpy-only port of `FilterbankFeatures` from the official
Cohere feature extractor (`cohere-pytorch/processing_cohere_asr.py`).

The previous `cohere_mel_spectrogram.py` in f16/ and q8/ was wrong on every
parameter that matters (n_fft, window size, filterbank normalization, log
base, and — most critically — was missing per-feature CMVN entirely).
That produced features outside the encoder's training distribution and
caused the "multilingual failure" that the agent attributed to model
training bias. It is not model bias: the features were wrong.

Matches the official extractor by construction:
- n_fft    = 2**ceil(log2(400)) = 512
- win_len  = 400 (25 ms @ 16 kHz), hann window, non-periodic
- hop      = 160 (10 ms @ 16 kHz)
- preemph  = 0.97 applied over valid samples only
- STFT     = center=True, pad_mode="constant"
- power    = |X|**2 (mag_power=2.0)
- mel      = librosa.filters.mel(n_mels=128, norm="slaney")
- log      = natural log with add-guard value 2**-24
- norm     = per-feature CMVN (zero mean, unit std per mel-bin over
             valid frames), ddof=1, epsilon = 1e-5
- mask     = invalid (post-valid-length) mel frames zeroed

Dither is omitted (training-time only; irrelevant at inference).
`pad_to=16` is omitted (the downstream CoreML encoder requires a fixed
3500-frame input regardless; the caller pads/truncates to that shape).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import librosa
import numpy as np


DITHER_CONSTANT = 1e-5
DEFAULT_LOG_ZERO_GUARD = 2 ** -24  # matches FilterbankFeatures default


@dataclass
class CohereMelConfig:
    sample_rate: int = 16000
    n_window_size: int = 400   # win_length (samples)
    n_window_stride: int = 160  # hop_length
    n_mels: int = 128
    lowfreq: float = 0.0
    highfreq: float | None = None  # defaults to sr/2
    preemph: float | None = 0.97
    mag_power: float = 2.0
    log_zero_guard_value: float = DEFAULT_LOG_ZERO_GUARD


class CohereMelSpectrogram:
    """Mel spectrogram matching the official Cohere Transcribe feature extractor."""

    def __init__(self, config: CohereMelConfig | None = None):
        self.cfg = config or CohereMelConfig()

        self.sample_rate = self.cfg.sample_rate
        self.win_length = self.cfg.n_window_size
        self.hop_length = self.cfg.n_window_stride
        self.n_fft = 2 ** math.ceil(math.log2(self.win_length))
        self.n_mels = self.cfg.n_mels
        self.preemph = self.cfg.preemph
        self.mag_power = self.cfg.mag_power
        self.log_zero_guard_value = self.cfg.log_zero_guard_value

        # Symmetric hann window — matches torch.hann_window(periodic=False).
        self.window = np.hanning(self.win_length).astype(np.float32)

        # If win_length < n_fft, torch.stft zero-pads the window to n_fft
        # centered on the signal window.
        if self.win_length < self.n_fft:
            pad_total = self.n_fft - self.win_length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            self.padded_window = np.pad(
                self.window, (pad_left, pad_right), mode="constant"
            ).astype(np.float32)
        else:
            self.padded_window = self.window

        # Slaney-normalized mel filterbank — matches librosa.filters.mel(norm="slaney").
        highfreq = self.cfg.highfreq if self.cfg.highfreq is not None else self.sample_rate / 2
        self.fb = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.cfg.lowfreq,
            fmax=highfreq,
            norm="slaney",
        ).astype(np.float32)  # shape: (n_mels, n_fft // 2 + 1)

    # ------------------------------------------------------------------
    # Exposed helpers
    # ------------------------------------------------------------------
    def get_seq_len(self, n_samples: int) -> int:
        """Valid mel frame count given raw sample count.

        Mirrors FilterbankFeatures.get_seq_len with exact_pad=False:
            pad = n_fft // 2 * 2 = n_fft
            return (n + pad - n_fft) // hop = n // hop
        """
        return int(n_samples) // self.hop_length

    def __call__(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (mel, valid_length).

        Args:
            audio: 1D float32 waveform in approximately [-1, 1].

        Returns:
            mel: (1, n_mels, n_frames) float32 — includes one trailing
                 STFT frame beyond `valid_length` (per torch.stft semantics
                 with center=True). Frames at index >= valid_length are
                 masked to 0.0.
            valid_length: int — number of valid mel frames to pass to the
                          encoder as `feature_length`.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Expected 1D waveform")

        valid_length = self.get_seq_len(audio.shape[0])

        # --- Preemphasis over valid samples only ---
        if self.preemph is not None:
            filtered = np.empty_like(audio)
            filtered[0] = audio[0]
            filtered[1:] = audio[1:] - self.preemph * audio[:-1]
            # Reference code masks post-valid samples to zero; our audio
            # already ends at valid_samples (no trailing padding) so this
            # is implicit. If a caller passes a padded waveform they must
            # trim first.
            audio = filtered

        # --- STFT: center=True, pad_mode="constant" ---
        pad = self.n_fft // 2
        padded = np.pad(audio, (pad, pad), mode="constant")

        n_frames = 1 + (padded.shape[0] - self.n_fft) // self.hop_length
        stft = np.empty((self.n_fft // 2 + 1, n_frames), dtype=np.complex64)
        window = self.padded_window
        for i in range(n_frames):
            start = i * self.hop_length
            stft[:, i] = np.fft.rfft(padded[start : start + self.n_fft] * window)

        # --- Magnitude then power ---
        mag = np.abs(stft).astype(np.float32)
        if self.mag_power != 1.0:
            mag = mag ** self.mag_power

        # --- Mel filterbank ---
        mel = self.fb @ mag  # (n_mels, n_frames)

        # --- Natural log with add-guard ---
        mel = np.log(mel + self.log_zero_guard_value)

        # --- Per-feature CMVN over valid frames, ddof=1 ---
        if valid_length > 1:
            valid = mel[:, :valid_length]
            mean = valid.mean(axis=1, keepdims=True)
            # var with ddof=1 => divide by (N-1)
            var = ((valid - mean) ** 2).sum(axis=1, keepdims=True) / (valid_length - 1)
            std = np.sqrt(var)
            std = np.where(np.isnan(std), 0.0, std)
            std = std + DITHER_CONSTANT
            mel = (mel - mean) / std
        # else: degenerate, leave as-is

        # --- Zero invalid (trailing) frames ---
        if valid_length < mel.shape[1]:
            mel[:, valid_length:] = 0.0

        # --- Batch dim ---
        return mel[np.newaxis, :, :].astype(np.float32), valid_length


def pad_or_truncate_to_fixed(
    mel: np.ndarray, valid_length: int, fixed_frames: int = 3500
) -> tuple[np.ndarray, int]:
    """Pad/truncate mel to `fixed_frames` for the CoreML encoder.

    Returns (mel_fixed, feature_length) where feature_length is the valid
    count clamped to fixed_frames.
    """
    assert mel.ndim == 3 and mel.shape[0] == 1
    cur = mel.shape[2]
    if cur > fixed_frames:
        return mel[:, :, :fixed_frames], min(valid_length, fixed_frames)
    if cur < fixed_frames:
        pad = fixed_frames - cur
        mel = np.pad(mel, ((0, 0), (0, 0), (0, pad)), mode="constant")
    return mel, min(valid_length, fixed_frames)
