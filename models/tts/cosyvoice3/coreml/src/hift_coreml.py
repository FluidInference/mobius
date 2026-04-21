"""CoreML-traceable wrapper for CosyVoice3 CausalHiFTGenerator (finalize=True).

Substitutions vs upstream:
  * `_stft` / `_istft` → matmul STFT/ISTFT from `src.stft_coreml`
  * `m_source.l_sin_gen` → `SineGen2CoreML` from `src.sinegen_coreml`
  * weight_norm parametrizations folded via `src.weight_norm_fold`
  * f0_predictor kept in FP32 (upstream uses FP64 for precision; ANE is FP32-only)

Only the finalize=True (whole-utterance) inference path is supported; this is
the path used when the model is called once per utterance from Swift.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sinegen_coreml import patch_source_module
from src.stft_coreml import ISTFT, STFT
from src.weight_norm_fold import fold_weight_norm


class HiFTCoreML(nn.Module):
    """Wraps a loaded CausalHiFTGenerator and exposes a single forward(mel)->audio.

    The wrapper binds the upstream generator's submodules directly — no weight
    copying — so `load_state_dict` is done on the upstream object before
    construction.
    """

    def __init__(self, gen: nn.Module):
        super().__init__()

        # Patch: CoreML-friendly SineGen2 (bit-exact, see test_sinegen_parity).
        patch_source_module(gen.m_source)
        # Patch: fold all weight_norm parametrizations to plain tensors.
        fold_weight_norm(gen)

        # Share the full generator submodule tree.
        self.gen = gen

        # Matmul STFT / ISTFT matching torch.stft(..., center=True) default.
        n_fft = gen.istft_params["n_fft"]
        hop = gen.istft_params["hop_len"]
        self.stft = STFT(n_fft, hop, window="hann")
        self.istft = ISTFT(n_fft, hop, window="hann")

        self.upsample_total = int(np.prod(gen.upsample_rates))
        self.hop_len = hop
        self.n_fft = n_fft
        self.audio_limit = gen.audio_limit
        self.lrelu_slope = gen.lrelu_slope
        self.num_upsamples = gen.num_upsamples
        self.num_kernels = gen.num_kernels

    def _decode(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Finalize=True decode path from CausalHiFTGenerator.decode."""
        # STFT of source (s: (B,1,L) → squeeze to (B,L))
        s_2d = s.squeeze(1)
        s_real, s_imag = self.stft(s_2d)  # (B, F, T)
        x = self.gen.conv_pre(x)
        s_stft = torch.cat([s_real, s_imag], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.gen.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.gen.reflection_pad(x)

            si = self.gen.source_downs[i](s_stft)
            si = self.gen.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                rb = self.gen.resblocks[i * self.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.gen.conv_post(x)
        n_bins = self.n_fft // 2 + 1
        magnitude = torch.exp(x[:, :n_bins, :])
        phase = torch.sin(x[:, n_bins:, :])

        # Match upstream _istft: clip magnitude before polar-to-rect conversion.
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        audio = self.istft(real, imag)  # (B, L)
        audio = torch.clamp(audio, -self.audio_limit, self.audio_limit)
        return audio

    def forward(
        self, mel: torch.Tensor, num_valid_frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            mel: (B, 80, T)  — possibly right-zero-padded to a fixed T
            num_valid_frames: (1,) int32 — real mel frames before padding
        Returns:
            audio: (B, T*480) — full padded audio
            audio_length_samples: (1,) int32 — num_valid_frames * hop_len
        """
        # F0 prediction in FP32 (upstream uses FP64; ANE doesn't support FP64).
        f0 = self.gen.f0_predictor(mel, finalize=True)  # (B, T)
        # f0_upsamp to sample rate, then source module.
        s = self.gen.f0_upsamp(f0[:, None]).transpose(1, 2)  # (B, L, 1)
        s, _, _ = self.gen.m_source(s)  # (B, L, H+1)
        s = s.transpose(1, 2)  # (B, H+1, L)
        # But CausalHiFTGenerator's decode expects s as (B,1,L) — it squeezes(1).
        # m_source.forward returns sine_merge via l_linear which collapses dim to 1.
        # So s after transpose is (B, 1, L).
        audio = self._decode(mel, s)
        audio_length_samples = (num_valid_frames * self.hop_len * self.upsample_total).to(
            torch.int32
        )
        return audio, audio_length_samples
