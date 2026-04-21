"""Verify matmul STFT/ISTFT matches torch.stft/torch.istft."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from scipy.signal import get_window

from src.stft_coreml import STFT, ISTFT


def main():
    n_fft, hop = 16, 4
    torch.manual_seed(0)

    # Run across a few lengths to check shape handling
    for L in [480, 1200, 4800, 120000]:
        x = torch.randn(1, L)
        window = torch.from_numpy(get_window("hann", n_fft, fftbins=True).astype(np.float32))

        # Torch reference (center=True by default)
        spec = torch.stft(x, n_fft, hop, n_fft, window=window, return_complex=True)
        real_t = spec.real  # (1, F, T)
        imag_t = spec.imag

        # Ours
        stft = STFT(n_fft, hop)
        real_o, imag_o = stft(x)

        mae_r = (real_t - real_o).abs().mean().item()
        mae_i = (imag_t - imag_o).abs().mean().item()
        max_r = (real_t - real_o).abs().max().item()

        # ISTFT parity
        x_rec_t = torch.istft(torch.complex(real_t, imag_t), n_fft, hop, n_fft, window=window)
        istft = ISTFT(n_fft, hop)
        x_rec_o = istft(real_o, imag_o)

        # Align lengths (torch.istft may return slightly different length than ours near edges)
        L_min = min(x_rec_t.shape[-1], x_rec_o.shape[-1])
        x_rec_t = x_rec_t[..., :L_min]
        x_rec_o = x_rec_o[..., :L_min]
        mae_wav = (x_rec_t - x_rec_o).abs().mean().item()
        max_wav = (x_rec_t - x_rec_o).abs().max().item()

        # Round-trip identity check
        rt_mae = (x[..., :L_min] - x_rec_o).abs().mean().item()

        print(f"L={L:>6d}  real_t={tuple(real_t.shape)} real_o={tuple(real_o.shape)}")
        print(f"        STFT  MAE: real={mae_r:.2e} imag={mae_i:.2e} max_r={max_r:.2e}")
        print(f"        ISTFT MAE: {mae_wav:.2e}   max={max_wav:.2e}")
        print(f"        round-trip MAE vs input: {rt_mae:.2e}")


if __name__ == "__main__":
    main()
