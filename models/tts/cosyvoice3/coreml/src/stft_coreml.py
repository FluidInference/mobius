"""CoreML-compatible STFT / iSTFT using Conv1d + ConvTranspose1d.

For the tiny n_fft=16 / hop_len=4 STFT used in HiFTGenerator, a full DFT matrix
is small enough to encode as a conv kernel directly. This sidesteps the
torch.stft / torch.istft ops which coremltools does not support.

Matches the configuration used in cosyvoice.hifigan.generator.HiFTGenerator:
    torch.stft(x, n_fft, hop_len, n_fft, window=hann, return_complex=True)
    torch.istft(complex, n_fft, hop_len, n_fft, window=hann)

Limitations:
    * center=False semantics (no reflection padding prepended). Matches the
      CosyVoice usage which calls torch.stft without `center=` and gets the
      default (center=True). We match by optionally center-padding the input.
    * window is assumed "hann" with fftbins=True.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window


def _dft_matrix(n_fft: int) -> np.ndarray:
    """Return a (n_fft, n_fft) complex DFT matrix."""
    k = np.arange(n_fft)
    n = np.arange(n_fft).reshape(-1, 1)
    return np.exp(-2j * np.pi * k * n / n_fft)


class STFT(nn.Module):
    """Matmul-based STFT matching torch.stft(center=True, return_complex=True).

    Output shapes:
        real, imag:  (B, F=n_fft//2+1, T)
    """

    def __init__(self, n_fft: int, hop_length: int, window: str = "hann"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        w = get_window(window, n_fft, fftbins=True).astype(np.float32)  # (n_fft,)
        dft = _dft_matrix(n_fft)  # (n_fft, n_fft)
        n_bins = n_fft // 2 + 1

        # Windowed DFT basis. Shape (n_bins, 1, n_fft) for Conv1d kernel.
        basis = dft[:n_bins] * w[np.newaxis, :]
        real_kernel = np.real(basis).astype(np.float32)
        imag_kernel = np.imag(basis).astype(np.float32)

        self.register_buffer("real_kernel", torch.from_numpy(real_kernel).unsqueeze(1))  # (F,1,N)
        self.register_buffer("imag_kernel", torch.from_numpy(imag_kernel).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (B, L) audio.
        Returns:
            real, imag: each (B, F, T).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        # center=True: reflect-pad by n_fft//2 on both sides
        pad = self.n_fft // 2
        x_padded = F.pad(x, (pad, pad), mode="reflect")
        real = F.conv1d(x_padded, self.real_kernel, stride=self.hop_length)
        imag = F.conv1d(x_padded, self.imag_kernel, stride=self.hop_length)
        return real, imag


class ISTFT(nn.Module):
    """Matmul-based iSTFT matching torch.istft(center=True, window=hann).

    Input:
        real, imag: each (B, F=n_fft//2+1, T).
    Output:
        audio: (B, L) where L = (T-1)*hop_length (center=True removes padding).
    """

    def __init__(self, n_fft: int, hop_length: int, window: str = "hann"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        w = get_window(window, n_fft, fftbins=True).astype(np.float32)
        dft = _dft_matrix(n_fft)
        n_bins = n_fft // 2 + 1

        # IDFT basis (conjugate of DFT, divided by n_fft) times window.
        # For real output we use the trick: full-spectrum IDFT from half-spectrum
        # requires adding conjugate symmetric mirror. We compute the full (n_fft,n_fft)
        # synthesis matrix applied to the half-spectrum via explicit formula.
        #
        # x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(2j*pi*k*n/N)
        # For real x, X[N-k] = conj(X[k]), so the sum over k=1..N/2-1 of (X[k]*e^... + conj(X[k])*e^-...) = 2 Re(X[k]*e^...)
        #
        # We'll just construct two synthesis kernels (for real and imag parts of the
        # half spectrum) that together reconstruct the time-domain frame.
        idft = np.conj(dft) / n_fft  # (n_fft, n_fft)
        # Scale half-spectrum: DC and Nyquist are not duplicated, others are.
        scale = np.ones(n_bins, dtype=np.float32) * 2.0
        scale[0] = 1.0
        if n_fft % 2 == 0:
            scale[-1] = 1.0

        # idft[:, :n_bins] has shape (n_fft, n_bins). Scale by spectrum scale factor.
        syn = idft[:, :n_bins] * scale[np.newaxis, :]  # (n_fft, n_bins)
        real_syn = np.real(syn).astype(np.float32)  # maps real(X) to time
        imag_syn = -np.imag(syn).astype(np.float32)  # maps imag(X) to time; sign for exp(+j) convention

        # Apply synthesis window
        real_syn = real_syn * w[:, np.newaxis]
        imag_syn = imag_syn * w[:, np.newaxis]

        # Build ConvTranspose1d kernels.
        # Layout: (in_channels=F, out_channels=1, kernel_size=n_fft).
        self.register_buffer(
            "real_kernel",
            torch.from_numpy(real_syn.T).unsqueeze(1),  # (F,1,N)
        )
        self.register_buffer(
            "imag_kernel",
            torch.from_numpy(imag_syn.T).unsqueeze(1),
        )

        # Window sum for overlap-add normalization: sum of shifted w^2
        # center=True chops n_fft//2 samples from each side of the OLA output.
        # For constant-overlap-add we divide by the sum of squared windows at each position.
        # Precompute the normalization kernel as a 1-channel Conv kernel of ones.
        # We'll compute the normalization on-the-fly in forward to handle variable T.
        self.register_buffer("window_sq", torch.from_numpy((w * w).astype(np.float32)))

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        # real, imag: (B, F, T)
        # Each frame = real_syn.T @ real + imag_syn.T @ imag  (length n_fft)
        # ConvTranspose1d with stride=hop_length performs the overlap-add.
        ola = F.conv_transpose1d(real, self.real_kernel, stride=self.hop_length) + F.conv_transpose1d(
            imag, self.imag_kernel, stride=self.hop_length
        )
        # ola: (B, 1, (T-1)*hop + n_fft)

        # Window-sum normalization via OLA of window^2.
        B, _, L = ola.shape
        T = real.shape[-1]
        ones = torch.ones(1, 1, T, device=real.device, dtype=real.dtype)
        win_sq_kernel = self.window_sq.view(1, 1, -1)
        norm = F.conv_transpose1d(ones, win_sq_kernel, stride=self.hop_length)  # (1,1,L)
        # Avoid div-by-near-zero at OLA edges. clamp_min is friendlier to CoreML
        # than torch.where(abs>eps, norm, ones) for FP32 precision.
        norm = torch.clamp(norm, min=1e-7)
        ola = ola / norm

        # Center=True: strip n_fft//2 samples from each side.
        pad = self.n_fft // 2
        ola = ola[..., pad : L - pad]
        return ola.squeeze(1)
