import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class CCM(nn.Module):
    """Complex Convolving Mask — real-valued implementation.

    Decomposes the cube-root-of-unity basis into explicit real/imag parts.
    Avoids torch.complex for GGML compatibility.

    Input mask: 27 channels = 3 (cube-root basis) x 9 (3x3 convolution kernel)
    Output: complex-valued enhanced STFT via real-valued arithmetic.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "v_real",
            torch.tensor([1, -1 / 2, -1 / 2], dtype=torch.float32),
        )
        self.register_buffer(
            "v_imag",
            torch.tensor(
                [0, np.sqrt(3) / 2, -np.sqrt(3) / 2], dtype=torch.float32
            ),
        )
        self.unfold = nn.Sequential(
            nn.ZeroPad2d([1, 1, 2, 0]),  # causal: pad top only
            nn.Unfold(kernel_size=(3, 3)),
        )

    def forward(self, m, x):
        """
        m: (B,27,T,F) — mask from decoder
        x: (B,F,T,2) — input STFT (real, imag)
        Returns: (B,F,T,2) — enhanced STFT
        """
        # Decompose mask into real/imag via cube-root basis
        m = rearrange(m, "b (r c) t f -> b r c t f", r=3)
        H_real = torch.sum(
            self.v_real[None, :, None, None, None] * m, dim=1
        )  # (B,9,T,F)
        H_imag = torch.sum(
            self.v_imag[None, :, None, None, None] * m, dim=1
        )  # (B,9,T,F)

        M_real = rearrange(H_real, "b (m n) t f -> b m n t f", m=3)  # (B,3,3,T,F)
        M_imag = rearrange(H_imag, "b (m n) t f -> b m n t f", m=3)  # (B,3,3,T,F)

        # Unfold input STFT
        x_perm = x.permute(0, 3, 2, 1).contiguous()  # (B,2,T,F)
        x_unfold = self.unfold(x_perm)
        x_unfold = rearrange(
            x_unfold, "b (c m n) (t f) -> b c m n t f", m=3, n=3, f=x.shape[1]
        )

        # Complex multiplication via real-valued arithmetic
        x_enh_real = torch.sum(
            M_real * x_unfold[:, 0] - M_imag * x_unfold[:, 1], dim=(1, 2)
        )  # (B,T,F)
        x_enh_imag = torch.sum(
            M_real * x_unfold[:, 1] + M_imag * x_unfold[:, 0], dim=(1, 2)
        )  # (B,T,F)

        x_enh = (
            torch.stack([x_enh_real, x_enh_imag], dim=3)
            .transpose(1, 2)
            .contiguous()
        )  # (B,F,T,2)
        return x_enh
