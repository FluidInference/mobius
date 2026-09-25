import math

import torch
import torch.nn as nn
from einops import rearrange


# Architecture versions (mirrors `localvqe.version` in GGUF metadata, see
# ggml/localvqe_model.h):
#   1 = post-conv BatchNorm2d + ELU (legacy v1).
#   2 = pre-norm CausalGroupNorm + ReLU6 with skip_norm on the decoder
#       skip path and an internal norm in SubpixelConv2d (v1.1, matches
#       localvqe-v1.1-1.3M.pt).
#   3 = identical to v2 structurally, SiLU activation in place of ReLU6
#       (v1.2 onward, matches localvqe-v1.2-1.3M.pt).


def causal_pad(kernel_size):
    """Compute ZeroPad2d args for causal convolution.

    Time: pad KH-1 on top (past only).
    Freq: symmetric-ish, preserves size at stride=1, halves at stride=2.
    """
    kh, kw = kernel_size
    pad_left = (kw - 1) // 2
    pad_right = kw - 1 - pad_left
    return (pad_left, pad_right, kh - 1, 0)


class CausalGroupNorm(nn.Module):
    """GroupNorm-like norm that is causal across the time axis.

    nn.GroupNorm(C, C) on a (B, C, T, F) tensor reduces across (C, T, F)
    per sample, making the normalization at frame t depend on frames
    > t — fatal for streaming inference. CausalGroupNorm reduces across
    (C, F) per (B, T) instead, so each frame's stats depend only on
    that frame. Equivalent to nn.LayerNorm([C, F]) when F is fixed,
    but F changes across encoder/decoder stages so we compute the
    reduction explicitly. Matches the GGML graph's
    build_causal_groupnorm.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"CausalGroupNorm expects 4-D input, got ndim={x.ndim}")
        mean = x.mean(dim=(1, 3), keepdim=True)
        var = x.var(dim=(1, 3), keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class FE(nn.Module):
    """Power-law feature extraction.

    Compresses magnitude via power-law, preserving phase direction.
    Output has same real/imag structure but with compressed magnitudes.
    """

    def __init__(self, c=0.3):
        super().__init__()
        self.c = c

    def forward(self, x):
        """x: (B,F,T,2) -> (B,2,T,F)"""
        x_mag = torch.sqrt(x[..., [0]] ** 2 + x[..., [1]] ** 2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1 - self.c) + 1e-12)
        return x_c.permute(0, 3, 2, 1).contiguous()


def _v2_activation(arch_version: int) -> nn.Module:
    """ReLU6 for arch_version=2 (v1.1), SiLU for arch_version=3 (v1.2)."""
    if arch_version == 2:
        return nn.ReLU6()
    if arch_version == 3:
        return nn.SiLU()
    raise ValueError(f"Unsupported arch_version={arch_version}")


class ResidualBlock(nn.Module):
    """v1: pad → conv → BN → ELU + skip.
    v2/v3: norm → pad → conv → act + skip (pre-norm; act=ReLU6 for v2, SiLU for v3)."""

    def __init__(self, channels, kernel_size=(4, 3), arch_version=2):
        super().__init__()
        self.arch_version = arch_version
        self.pad = nn.ZeroPad2d(causal_pad(kernel_size))
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size)
        if arch_version == 1:
            self.bn = nn.BatchNorm2d(channels)
            self.elu = nn.ELU()
        else:
            self.norm = CausalGroupNorm(channels)
            self.act = _v2_activation(arch_version)

    def forward(self, x):
        if self.arch_version == 1:
            return self.elu(self.bn(self.conv(self.pad(x)))) + x
        return self.act(self.conv(self.pad(self.norm(x)))) + x


class EncoderBlock(nn.Module):
    """v1: pad → conv (stride=(1,2)) → BN → ELU → ResidualBlock.
    v2/v3: norm → pad → conv (stride=(1,2)) → act → ResidualBlock."""

    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), stride=(1, 2),
                 arch_version=2):
        super().__init__()
        self.arch_version = arch_version
        self.pad = nn.ZeroPad2d(causal_pad(kernel_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        if arch_version == 1:
            self.bn = nn.BatchNorm2d(out_channels)
            self.elu = nn.ELU()
        else:
            self.norm = CausalGroupNorm(in_channels)
            self.act = _v2_activation(arch_version)
        self.resblock = ResidualBlock(out_channels, kernel_size=kernel_size,
                                      arch_version=arch_version)

    def forward(self, x):
        if self.arch_version == 1:
            return self.resblock(self.elu(self.bn(self.conv(self.pad(x)))))
        return self.resblock(self.act(self.conv(self.pad(self.norm(x)))))


class S4DBottleneck(nn.Module):
    """Diagonal State Space Model (S4D) bottleneck.

    Complex diagonal SSM with polar parameterization for guaranteed stability.
    A = r * exp(j*theta) where r = exp(-softplus(rate)) < 1.
    Each state channel is independently prunable since A is diagonal.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input/output projections
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_size)

        # Diagonal A in polar form: A = r * exp(j*theta)
        # r = exp(-softplus(log_rate)) guarantees r in (0, 1)
        self.A_log_rate = nn.Parameter(torch.empty(hidden_size))
        self.A_theta = nn.Parameter(torch.empty(hidden_size))

        # Per-channel input coupling B = B_real + j * B_imag
        self.B_real = nn.Parameter(torch.empty(hidden_size))
        self.B_imag = nn.Parameter(torch.empty(hidden_size))

        # Per-channel output coupling C = C_real + j * C_imag
        self.C_real = nn.Parameter(torch.empty(hidden_size))
        self.C_imag = nn.Parameter(torch.empty(hidden_size))

        # Skip connection
        self.D = nn.Parameter(torch.empty(input_size))

        self._init_parameters()

    def _init_parameters(self):
        # log_rate: uniform so softplus(log_rate) covers ~0.001 to ~7
        # => r = exp(-softplus) covers ~0.999 (slow decay) to ~0.001 (fast decay)
        nn.init.uniform_(self.A_log_rate, -6.0, 2.0)
        # theta: uniform frequency coverage [0, pi]
        with torch.no_grad():
            self.A_theta.copy_(torch.linspace(0, math.pi, self.hidden_size))
        # B, C: small random
        nn.init.normal_(self.B_real, std=0.01)
        nn.init.normal_(self.B_imag, std=0.01)
        nn.init.normal_(self.C_real, std=0.01)
        nn.init.normal_(self.C_imag, std=0.01)
        # D: identity skip
        nn.init.ones_(self.D)

    def forward(self, x):
        """x: (B,C,T,F) -> (B,C,T,F)"""
        B_dim, C, T, F = x.shape
        u = rearrange(x, "b c t f -> b t (c f)")  # (B, T, H)

        v = self.input_proj(u)  # (B, T, N)

        # Polar -> Cartesian: A = r * (cos(theta) + j*sin(theta))
        # Clamp softplus output to ensure minimum decay rate (prevents r → 1 instability)
        sp = torch.nn.functional.softplus(self.A_log_rate).clamp(min=0.01)
        r = torch.exp(-sp)  # (N,) in (0, ~0.99)
        a_real = r * torch.cos(self.A_theta)  # (N,)
        a_imag = r * torch.sin(self.A_theta)  # (N,)

        # Run recurrence in float32 for numerical stability under AMP
        # Pre-allocate output buffer to avoid per-step allocations
        v_f = v.float()  # (B, T, N) — convert once
        h_real = torch.zeros(B_dim, self.hidden_size, dtype=torch.float32, device=v.device)
        h_imag = torch.zeros(B_dim, self.hidden_size, dtype=torch.float32, device=v.device)
        y = torch.empty(B_dim, T, self.hidden_size, dtype=torch.float32, device=v.device)
        a_real_f = a_real.float()
        a_imag_f = a_imag.float()
        B_real_f = self.B_real.float()
        B_imag_f = self.B_imag.float()
        C_real_f = self.C_real.float()
        C_imag_f = self.C_imag.float()

        for t in range(T):
            v_t = v_f[:, t, :]  # (B, N) — no per-step conversion
            h_real_new = a_real_f * h_real - a_imag_f * h_imag + B_real_f * v_t
            h_imag_new = a_real_f * h_imag + a_imag_f * h_real + B_imag_f * v_t
            h_real = h_real_new
            h_imag = h_imag_new
            y[:, t, :] = C_real_f * h_real - C_imag_f * h_imag

        y = y.to(u.dtype)  # back to original dtype for projections
        out = self.output_proj(y) + self.D * u  # (B, T, H)
        return rearrange(out, "b t (c f) -> b c t f", c=C)


class SubpixelConv2d(nn.Module):
    """v1: pad → conv → reshape (×2 freq).
    v2/v3: norm → pad → conv → reshape (×2 freq) (pre-norm)."""

    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), arch_version=2):
        super().__init__()
        self.arch_version = arch_version
        if arch_version in (2, 3):
            self.norm = CausalGroupNorm(in_channels)
        elif arch_version != 1:
            raise ValueError(f"Unsupported arch_version={arch_version}")
        self.pad = nn.ZeroPad2d(causal_pad(kernel_size))
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size)

    def forward(self, x):
        if self.arch_version in (2, 3):
            x = self.norm(x)
        y = self.conv(self.pad(x))
        y = rearrange(y, "b (r c) t f -> b c t (r f)", r=2)
        return y


class DecoderBlock(nn.Module):
    """v1: skip_conv(x_en) + x → resblock → deconv → BN+ELU (if not last).
    v2/v3: skip_norm → skip_conv → +x → resblock → deconv → act (if not last).
    Act is ReLU6 for v2 and SiLU for v3."""

    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), is_last=False,
                 enc_channels=None, arch_version=2):
        super().__init__()
        self.arch_version = arch_version
        if enc_channels is None:
            enc_channels = in_channels
        if arch_version in (2, 3):
            self.skip_norm = CausalGroupNorm(enc_channels)
        elif arch_version != 1:
            raise ValueError(f"Unsupported arch_version={arch_version}")
        self.skip_conv = nn.Conv2d(enc_channels, in_channels, 1)
        self.resblock = ResidualBlock(in_channels, kernel_size=kernel_size,
                                      arch_version=arch_version)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size,
                                     arch_version=arch_version)
        self.is_last = is_last
        if not is_last:
            if arch_version == 1:
                self.bn = nn.BatchNorm2d(out_channels)
                self.elu = nn.ELU()
            else:
                self.act = _v2_activation(arch_version)

    def forward(self, x, x_en):
        if self.arch_version in (2, 3):
            x_en = self.skip_norm(x_en)
        y = x + self.skip_conv(x_en)
        y = self.deconv(self.resblock(y))
        if not self.is_last:
            if self.arch_version == 1:
                y = self.elu(self.bn(y))
            else:
                y = self.act(y)
        return y
