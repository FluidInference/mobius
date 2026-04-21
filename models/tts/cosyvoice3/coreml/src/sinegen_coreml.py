"""CoreML-traceable replacement for cosyvoice.hifigan.generator.SineGen2.

The upstream SineGen2 in causal+eval mode is already deterministic (reads from
pre-sampled `rand_ini` and `sine_waves` buffers rather than drawing new noise).
The only tracing-level blockers are:

  1. `torch.multiply(f0, torch.FloatTensor([[range(1, H+2)]]))` at forward()
     — constructs a tensor from a Python range iterator and relies on implicit
     broadcast, which coremltools lowers via `aten::broadcast_tensors`.
  2. The `% 1` mod and `interpolate(scale_factor=1/upsample)` chain are fine
     for trace but worth stabilising.

This module reproduces causal-eval SineGen2 using plain pre-registered buffers
and explicit `expand` broadcasts so the traced graph lowers cleanly to MIL ops.

It is intentionally only correct for:
    * training=False
    * causal=True
    * flag_for_pulse=False
which is the configuration used by CausalHiFTGenerator at 24 kHz.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SineGen2CoreML(nn.Module):
    """Deterministic causal-eval replacement for SineGen2.

    Mirrors state: `rand_ini` (1, H+1), `sine_waves` (1, Lmax, H+1).
    Copies these from a source SineGen2 instance via `load_from(source)`.
    """

    def __init__(
        self,
        samp_rate: int,
        upsample_scale: int,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        max_len_samples: int = 300 * 24000,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

        # Harmonic multipliers 1..H+1 as a fixed buffer of shape (1, 1, H+1).
        harmonics = torch.arange(1, harmonic_num + 2, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("harmonics", harmonics)

        # Placeholders; populated by load_from().
        self.register_buffer("rand_ini", torch.zeros(1, harmonic_num + 1))
        self.register_buffer("sine_waves", torch.zeros(1, max_len_samples, harmonic_num + 1))

    @torch.no_grad()
    def load_from(self, src: nn.Module) -> "SineGen2CoreML":
        """Copy `rand_ini` and `sine_waves` tensors from an upstream SineGen2."""
        self.rand_ini.copy_(src.rand_ini.detach())
        # Ensure length matches (upstream allocates 300*24000 too).
        L = min(self.sine_waves.shape[1], src.sine_waves.shape[1])
        self.sine_waves[:, :L].copy_(src.sine_waves[:, :L].detach())
        return self

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        return (f0 > self.voiced_threshold).to(f0.dtype)

    def _f02sine(self, f0_values: torch.Tensor) -> torch.Tensor:
        """f0_values: (B, L, dim). Returns (B, L, dim) of sines."""
        rad_values = (f0_values / self.sampling_rate) % 1
        # causal-eval: first-timestep phase offset from fixed buffer.
        rand_ini = self.rand_ini.to(rad_values.dtype)  # (1, dim)
        rad_values = rad_values.clone()
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # Downsample phase rate by upsample_scale, then cumulative sum, then nearest-upsample back.
        # (B, L, dim) -> (B, dim, L) for interpolate.
        x = rad_values.transpose(1, 2)
        x = F.interpolate(x, scale_factor=1.0 / self.upsample_scale, mode="linear")
        x = x.transpose(1, 2)  # (B, L', dim)

        phase = torch.cumsum(x, dim=1) * (2.0 * np.pi)
        # upsample with nearest (causal path) after scaling by upsample_scale.
        phase_up = F.interpolate(
            (phase * self.upsample_scale).transpose(1, 2),
            scale_factor=self.upsample_scale,
            mode="nearest",
        ).transpose(1, 2)
        return torch.sin(phase_up)

    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            f0: (B, L, 1)
        Returns:
            sine_waves: (B, L, H+1) post amplitude + uv mask + noise
            uv: (B, L, 1)
            noise: (B, L, H+1)
        """
        # Replace torch.multiply(f0, torch.FloatTensor([[range(1,H+2)]])) with
        # explicit expand-multiply. f0: (B,L,1), harmonics: (1,1,H+1).
        fn = f0 * self.harmonics  # (B, L, H+1)
        sine_waves = self._f02sine(fn) * self.sine_amp  # (B, L, H+1)

        uv = self._f02uv(f0)  # (B, L, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        # Causal-eval: read from fixed buffer.
        L = sine_waves.shape[1]
        noise = noise_amp * self.sine_waves[:, :L].to(sine_waves.dtype)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


def patch_source_module(source_module: nn.Module) -> nn.Module:
    """Replace `source_module.l_sin_gen` with a CoreML-friendly clone.

    Works on cosyvoice.hifigan.generator.SourceModuleHnNSF instances whose
    l_sin_gen is an upstream SineGen2 in causal mode.
    """
    src = source_module.l_sin_gen
    # Only patch if it looks like SineGen2 (has rand_ini + sine_waves).
    if not (hasattr(src, "rand_ini") and hasattr(src, "sine_waves")):
        return source_module

    replacement = SineGen2CoreML(
        samp_rate=src.sampling_rate,
        upsample_scale=src.upsample_scale,
        harmonic_num=src.harmonic_num,
        sine_amp=src.sine_amp,
        noise_std=src.noise_std,
        voiced_threshold=src.voiced_threshold,
        max_len_samples=src.sine_waves.shape[1],
    )
    replacement.load_from(src)
    source_module.l_sin_gen = replacement
    return source_module
