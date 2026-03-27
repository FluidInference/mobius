#!/usr/bin/env python3
"""Traceable kaldi-compatible fbank feature extractor for CoreML fusion.

Implements 80-dim log-mel filterbank features matching icefall/kaldifeat
using only ops that coremltools can convert (no as_strided, no dynamic shapes).

The module takes fixed-length raw audio waveform and produces mel frames
that can be fed directly to the Zipformer2 encoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KaldiFbank(nn.Module):
    """Kaldi-compatible fbank feature extractor, traceable by torch.jit.trace.

    Matches torchaudio.compliance.kaldi.fbank with:
        sample_frequency=16000, num_mel_bins=80, frame_length=25.0,
        frame_shift=10.0, window_type='povey', dither=0.0,
        energy_floor=1.0, snip_edges=False

    Args:
        n_mels: Number of mel bins (default: 80).
        sample_rate: Audio sample rate (default: 16000).
        n_fft: FFT size (default: 512).
        hop_length: Hop size in samples (default: 160, = 10ms).
        win_length: Window size in samples (default: 400, = 25ms).
    """

    def __init__(
        self,
        n_mels: int = 80,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        preemph: float = 0.97,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.preemph = preemph
        # Kaldi fbank uses machine epsilon as the log floor (not energy_floor=1.0,
        # which is only for raw signal energy). This preserves low-energy mel bins.
        self.log_floor = torch.finfo(torch.float32).eps

        # Povey window = hann^0.85 (win_length samples, NOT zero-padded to n_fft)
        hann = torch.hann_window(win_length, periodic=False)
        povey = hann.pow(0.85)
        self.register_buffer("window", povey)  # (win_length,)

        # Use torchaudio's kaldi mel filterbank for exact match
        import torchaudio.compliance.kaldi as kaldi_mod

        kaldi_fb, _ = kaldi_mod.get_mel_banks(
            n_mels, n_fft, sample_rate,
            low_freq=20.0, high_freq=0.0,  # 0.0 = Nyquist (fbank() default)
            vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0,
        )
        # kaldi_fb is (n_mels, n_fft//2) — pad with zero column for Nyquist bin
        fb = F.pad(kaldi_fb, (0, 1))  # (n_mels, n_fft//2 + 1)
        self.register_buffer("mel_filterbank", fb)

    @staticmethod
    def _hz_to_mel_htk(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz_htk(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def _create_mel_filterbank(
        cls, n_fft: int, n_mels: int, sample_rate: int
    ) -> torch.Tensor:
        """Create HTK-style mel filterbank (matches kaldi)."""
        num_bins = n_fft // 2 + 1
        low_freq = 20.0
        high_freq = sample_rate / 2.0 - 400.0  # kaldi default: nyquist - 400

        low_mel = cls._hz_to_mel_htk(low_freq)
        high_mel = cls._hz_to_mel_htk(high_freq)

        # Mel center frequencies
        mel_points = torch.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # FFT bin frequencies
        fft_freqs = torch.linspace(0, sample_rate / 2.0, num_bins)

        fb = torch.zeros(n_mels, num_bins)
        for m in range(n_mels):
            f_left = hz_points[m]
            f_center = hz_points[m + 1]
            f_right = hz_points[m + 2]

            for k in range(num_bins):
                freq = fft_freqs[k]
                if f_left < freq <= f_center:
                    fb[m, k] = (freq - f_left) / (f_center - f_left)
                elif f_center < freq < f_right:
                    fb[m, k] = (f_right - freq) / (f_right - f_center)

        return fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel fbank features.

        Matches torchaudio.compliance.kaldi.fbank exactly by following kaldi's
        processing order: reflection pad → frame → DC offset → preemph per-frame
        → povey window → zero-pad → FFT → power → mel → log.

        Args:
            waveform: (1, num_samples) raw audio at 16kHz.

        Returns:
            (num_frames, n_mels) log-mel features matching kaldi fbank output.
        """
        # Remove batch dim for processing
        x = waveform[0]  # (num_samples,)
        num_samples = x.shape[0]

        # --- Step 1: Reflection padding (matches kaldi snip_edges=False) ---
        # kaldi: pad = window_size//2 - window_shift//2
        left_pad = self.win_length // 2 - self.hop_length // 2  # 200 - 80 = 120
        num_frames = (num_samples + self.hop_length // 2) // self.hop_length

        # Kaldi uses reflection: cat(reversed[-pad:], signal, reversed_full)
        x_reversed = x.flip(0)
        if left_pad > 0:
            pad_left = x_reversed[-left_pad:]
        else:
            pad_left = x[:0]  # empty tensor if no left pad needed
        x_padded = torch.cat([pad_left, x, x_reversed])

        # --- Step 2: Frame extraction via index gather (coremltools compatible) ---
        frame_starts = torch.arange(num_frames) * self.hop_length
        sample_offsets = torch.arange(self.win_length)
        indices = frame_starts.unsqueeze(1) + sample_offsets.unsqueeze(0)
        frames = x_padded[indices]  # (num_frames, win_length)

        # --- Step 3: Remove DC offset per frame (kaldi default) ---
        frames = frames - frames.mean(dim=1, keepdim=True)

        # --- Step 4: Preemphasis PER-FRAME (kaldi applies after framing) ---
        # kaldi: pad first sample with replicate, then frame[j] -= coeff * frame[j-1]
        if self.preemph > 0:
            first_col = frames[:, :1]  # replicate first sample
            shifted = torch.cat([first_col, frames[:, :-1]], dim=1)
            frames = frames - self.preemph * shifted

        # --- Step 5: Apply povey window ---
        frames = frames * self.window  # (num_frames, win_length)

        # --- Step 6: Zero-pad to n_fft for FFT ---
        frames = F.pad(frames, (0, self.n_fft - self.win_length))

        # --- Step 7: FFT → power spectrum ---
        spectrum = torch.fft.rfft(frames, n=self.n_fft)
        power = spectrum.abs().pow(2)  # matches torchaudio: .abs().pow(2.0)

        # --- Step 8: Mel filterbank → log ---
        mel = torch.matmul(power, self.mel_filterbank.t())
        mel = torch.max(mel, torch.tensor(self.log_floor)).log()

        return mel  # (T, n_mels)
