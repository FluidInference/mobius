import torch
import torch.nn as nn

from .align import AlignBlock
from .blocks import (
    FE,
    DecoderBlock,
    EncoderBlock,
    S4DBottleneck,
)
from .ccm import CCM


class DCTEncoder(nn.Module):
    """Orthogonal DCT-II analysis filterbank (Conv1d, 512 filters, stride 256).

    Output is reshaped to (B, F, T, 2) so the downstream conv stack sees the
    same "two channels per frequency bin" layout as a complex STFT, but every
    downstream op is real-valued.
    """

    def __init__(self, n_freqs=256, kernel_size=512, stride=256):
        super().__init__()
        n_filters = kernel_size  # complete basis: 512 filters for 512-sample frames
        self.n_freqs = n_freqs
        self.conv = nn.Conv1d(1, n_filters, kernel_size, stride=stride, bias=False)
        self._init_dct(n_filters, kernel_size)

    def _init_dct(self, n_filters, kernel_size):
        """Initialize with orthogonal DCT-II basis."""
        import math as _math
        k = torch.arange(n_filters).unsqueeze(1).float()
        n = torch.arange(kernel_size).unsqueeze(0).float()
        basis = torch.cos(_math.pi * (2 * n + 1) * k / (2 * kernel_size))
        basis[0] *= 1.0 / (kernel_size ** 0.5)
        basis[1:] *= (2.0 / kernel_size) ** 0.5
        with torch.no_grad():
            self.conv.weight.copy_(basis.unsqueeze(1))

    def forward(self, x):
        """x: (B, N) waveform → (B, F, T, 2) STFT-compatible format"""
        pad = self.conv.kernel_size[0] // 2
        x_padded = torch.nn.functional.pad(x, (pad, pad))
        out = self.conv(x_padded.unsqueeze(1))  # (B, 512, T)
        # Reshape to (B, 256, T, 2) — treat pairs of filters as "real/imag"
        B, _N, T = out.shape
        return out.reshape(B, self.n_freqs, 2, T).permute(0, 1, 3, 2)  # (B, F, T, 2)


class DCTDecoder(nn.Module):
    """DCT synthesis filterbank: Linear projection back to 512 samples + overlap-add."""

    def __init__(self, n_freqs=256, kernel_size=512, stride=256):
        super().__init__()
        n_filters = kernel_size  # complete basis
        self.n_freqs = n_freqs
        self.linear = nn.Linear(n_filters, kernel_size, bias=False)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = kernel_size // 2  # must match encoder padding
        self.register_buffer('_overlap_count', None)

    def _init_from_encoder(self, encoder):
        """Initialize synthesis basis from encoder weights (transposed)."""
        with torch.no_grad():
            self.linear.weight.copy_(encoder.conv.weight.squeeze(1).T)

    def forward(self, x, length=None):
        """x: (B, F, T, 2) → (B, N) waveform"""
        # (B, F, T, 2) → (B, 512, T)
        B, F, T, _2 = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, F * 2, T)

        # Project each frame back to time domain
        frames = self.linear(x.transpose(1, 2))  # (B, T, kernel_size)

        out_len = (T - 1) * self.stride + self.kernel_size
        # fold expects (B, C*kernel, T) — treat each sample in the frame as a channel
        output = torch.nn.functional.fold(
            frames.transpose(1, 2),  # (B, kernel_size, T)
            output_size=(1, out_len),
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
        ).squeeze(2).squeeze(1)  # (B, out_len)

        # Normalize by overlap count (computed once, cached as buffer)
        if self._overlap_count is None or self._overlap_count.shape[-1] != out_len:
            ones = torch.ones(1, self.kernel_size, T, device=x.device)
            self._overlap_count = torch.nn.functional.fold(
                ones,
                output_size=(1, out_len),
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
            ).squeeze().clamp(min=1)
        output = output / self._overlap_count

        # Remove encoder padding
        output = output[:, self.pad:]

        if length is not None:
            if output.shape[-1] < length:
                output = torch.nn.functional.pad(output, (0, length - output.shape[-1]))
            else:
                output = output[:, :length]
        return output



def compute_freq_progression(n_freqs, kernel_size, n_stages=5):
    """Compute frequency bin count at each encoder stage.

    Each encoder block halves frequency via stride=2 with causal padding.
    """
    _kh, kw = kernel_size
    pad_left = (kw - 1) // 2
    pad_right = kw - 1 - pad_left
    f = n_freqs
    freqs = [f]
    for _ in range(n_stages):
        f = (f + pad_left + pad_right - kw) // 2 + 1
        freqs.append(f)
    return freqs


class LocalVQE(nn.Module):
    """LocalVQE: joint AEC / NS / dereverberation.

    Component order: DCT analysis → mic encoder (5 blocks) + far-end encoder
    (2 blocks) → AlignBlock (soft delay cross-attention) → fused encoder
    block → S4D bottleneck → decoder (5 sub-pixel blocks) → mask head → CCM
    (3×3 complex convolving mask, real-valued) → DCT synthesis + overlap-add.
    See configs/default.yaml for channel widths of the published checkpoint.
    """

    def __init__(
        self,
        mic_channels=None,
        far_channels=None,
        align_hidden=32,
        dmax=64,
        power_law_c=0.3,
        n_freqs=256,
        kernel_size=(4, 4),
        bottleneck_hidden=162,
        arch_version=3,
    ):
        super().__init__()
        if mic_channels is None:
            mic_channels = [2, 32, 40, 40, 40, 40]
        if far_channels is None:
            far_channels = [2, 32, 40]

        self.n_freqs = n_freqs
        self.arch_version = arch_version
        ks = tuple(kernel_size)
        av = arch_version

        self.encoder = DCTEncoder(n_freqs=n_freqs, kernel_size=512, stride=256)
        self.decoder = DCTDecoder(n_freqs=n_freqs, kernel_size=512, stride=256)
        self.decoder._init_from_encoder(self.encoder)

        # Feature extraction
        self.fe_mic = FE(c=power_law_c)
        self.fe_ref = FE(c=power_law_c)

        # Mic encoder blocks 1-2
        self.mic_enc1 = EncoderBlock(mic_channels[0], mic_channels[1], kernel_size=ks, arch_version=av)
        self.mic_enc2 = EncoderBlock(mic_channels[1], mic_channels[2], kernel_size=ks, arch_version=av)

        # Far-end encoder blocks 1-2
        self.far_enc1 = EncoderBlock(far_channels[0], far_channels[1], kernel_size=ks, arch_version=av)
        self.far_enc2 = EncoderBlock(far_channels[1], far_channels[2], kernel_size=ks, arch_version=av)

        # Alignment
        self.align = AlignBlock(
            in_channels=mic_channels[2],  # 128
            hidden_channels=align_hidden,
            dmax=dmax,
        )

        # Mic encoder blocks 3-5 (block 3 takes concat: 128+128=256)
        self.mic_enc3 = EncoderBlock(mic_channels[2] * 2, mic_channels[3], kernel_size=ks, arch_version=av)
        self.mic_enc4 = EncoderBlock(mic_channels[3], mic_channels[4], kernel_size=ks, arch_version=av)
        self.mic_enc5 = EncoderBlock(mic_channels[4], mic_channels[5], kernel_size=ks, arch_version=av)

        # Bottleneck: channels * freq_bins at deepest encoder stage
        freqs = compute_freq_progression(n_freqs, ks)
        bn_input = mic_channels[5] * freqs[5]
        bn_hidden = bottleneck_hidden if bottleneck_hidden > 0 else bn_input // 2
        self.bottleneck = S4DBottleneck(bn_input, bn_hidden)

        # Decoder blocks (mirror encoder)
        self.dec5 = DecoderBlock(mic_channels[5], mic_channels[4], kernel_size=ks, arch_version=av)
        self.dec4 = DecoderBlock(mic_channels[4], mic_channels[3], kernel_size=ks, arch_version=av)
        self.dec3 = DecoderBlock(mic_channels[3], mic_channels[2], kernel_size=ks, arch_version=av)
        self.dec2 = DecoderBlock(mic_channels[2], mic_channels[1], kernel_size=ks, arch_version=av)
        self.dec1 = DecoderBlock(mic_channels[1], 27, kernel_size=ks, is_last=True, arch_version=av)
        self.mask = CCM()
        self._init_ccm_identity()

    def _init_ccm_identity(self):
        """Initialize dec1 deconv bias so the CCM mask starts as identity (passthrough).

        The 27-ch mask is reshaped as (3 basis, 9 kernel).  Basis vectors
        v_real=[1,-0.5,-0.5] and v_imag=[0,√3/2,-√3/2] sum to zero, so
        default init (similar values across the 3 groups) produces near-zero
        mask magnitude.  Fix: set the current-frame kernel element of the
        first basis (r=0, v_real=1, v_imag=0) to 1, giving H_real[center]=1.

        CCM uses causal ZeroPad2d([1,1,2,0]) with 3×3 kernel:
          m=0: t-2, m=1: t-1, m=2: t (current frame)
          n=0: f-1, n=1: f (current freq), n=2: f+1
        So current (t, f) = kernel index 2*3+1 = 7, NOT 4.

        SubpixelConv2d stores 54 channels (27×2 for sub-pixel shuffle).
        Output channel c comes from conv channels c (even freq) and c+27
        (odd freq), so we set bias[7] = bias[34] = 1.
        """
        conv = self.dec1.deconv.conv
        with torch.no_grad():
            conv.bias.zero_()
            conv.bias[7] = 1.0   # r=0, current (t,f), even freq bins
            conv.bias[34] = 1.0  # r=0, current (t,f), odd freq bins

    def forward(self, mic_wav, ref_wav, return_delay=False):
        """
        mic_wav: (B, N) waveform
        ref_wav: (B, N) waveform

        Returns:
            enhanced: DCT-encoded domain output
            delay_dist: (B, T, dmax) — delay distribution (if return_delay=True)
        """
        # DCT encoder: (B, N) waveform → (B, F, T, 2) encoded
        mic_enc = self.encoder(mic_wav)
        ref_enc = self.encoder(ref_wav)

        # Feature extraction
        mic_fe = self.fe_mic(mic_enc)
        ref_fe = self.fe_ref(ref_enc)

        # Mic encoder 1-2
        mic_e1 = self.mic_enc1(mic_fe)  # (B, 64, T, 129)
        mic_e2 = self.mic_enc2(mic_e1)  # (B, 128, T, 65)

        # Far-end encoder 1-2
        far_e1 = self.far_enc1(ref_fe)  # (B, 32, T, 129)
        far_e2 = self.far_enc2(far_e1)  # (B, 128, T, 65)

        # Alignment
        align_result = self.align(mic_e2, far_e2, return_delay=return_delay)
        if return_delay:
            aligned_far, delay_dist = align_result
        else:
            aligned_far = align_result

        # Concat mic + aligned far-end
        concat = torch.cat([mic_e2, aligned_far], dim=1)  # (B, 256, T, 65)

        # Mic encoder 3-5
        mic_e3 = self.mic_enc3(concat)  # (B, 128, T, 33)
        mic_e4 = self.mic_enc4(mic_e3)  # (B, 128, T, 17)
        mic_e5 = self.mic_enc5(mic_e4)  # (B, 128, T, 9)

        # Bottleneck
        bn = self.bottleneck(mic_e5)  # (B, 128, T, 9)

        # Decoder with skip connections (trim freq to match encoder)
        d5 = self.dec5(bn, mic_e5)[..., : mic_e4.shape[-1]]
        d4 = self.dec4(d5, mic_e4)[..., : mic_e3.shape[-1]]
        d3 = self.dec3(d4, mic_e3)[..., : mic_e2.shape[-1]]
        d2 = self.dec2(d3, mic_e2)[..., : mic_e1.shape[-1]]
        d1 = self.dec1(d2, mic_e1)[..., : mic_fe.shape[-1]]

        # Apply CCM mask
        enhanced = self.mask(d1, mic_enc)

        if return_delay:
            return enhanced, delay_dist, d1
        return enhanced

    @classmethod
    def from_config(cls, cfg):
        """Create model from a Config object."""
        return cls(
            mic_channels=cfg.model.mic_channels,
            far_channels=cfg.model.far_channels,
            align_hidden=cfg.model.align_hidden,
            dmax=cfg.model.dmax,
            power_law_c=cfg.model.power_law_c,
            n_freqs=cfg.audio.n_freqs,
            kernel_size=cfg.model.kernel_size,
            bottleneck_hidden=cfg.model.bottleneck_hidden,
            arch_version=getattr(cfg.model, "arch_version", 2),
        )
