"""
CoreML-compatible version of CosyVoice3 generator.

Key changes from original:
1. Uses custom CoreMLISTFT instead of torch.istft
2. Integrates patched SineGen2 from generator_patched.py
3. All operations verified CoreML-compatible
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add cosyvoice repo to path
REPO_PATH = Path(__file__).parent / "cosyvoice_repo"
sys.path.insert(0, str(REPO_PATH))

# Import base components
from cosyvoice.hifigan.generator import (
    ResBlock,
    init_weights,
    CausalConv1d,
    CausalConv1dUpsample,
    CausalConv1dDownSample,
)
from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
from torch.nn.utils import weight_norm

# Import our patched modules
from generator_patched import SourceModuleHnNSF as PatchedSourceModuleHnNSF
from istft_coreml import CoreMLISTFT


class CausalHiFTGeneratorCoreML(nn.Module):
    """
    CoreML-compatible version of CausalHiFTGenerator.

    Changes from original:
    - Replaces torch.istft with CoreMLISTFT
    - Uses patched SourceModuleHnNSF (fixes torch.multiply)
    - Simplified _istft method to use custom implementation
    """

    def __init__(
            self,
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=24000,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            conv_pre_look_right=4,
            f0_predictor=None,
    ):
        super().__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Use patched SourceModuleHnNSF
        self.m_source = PatchedSourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
            sinegen_type='2',  # Force SineGen2 for sampling_rate=24000
            causal=True
        )
        self.upsample_rates = upsample_rates
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = weight_norm(
            CausalConv1d(in_channels, base_channels, conv_pre_look_right + 1, 1, causal_type='right')
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    CausalConv1dUpsample(
                        base_channels // (2**i),
                        base_channels // (2**(i + 1)),
                        k,
                        u,
                    )
                )
            )

        # Downsampling layers for source
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(
                    CausalConv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1, causal_type='left')
                )
            else:
                self.source_downs.append(
                    CausalConv1dDownSample(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u)
                )

            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d, causal=True)
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d, causal=True))

        # LayerNorm to stabilize ResBlocks outputs (prevents exponential amplification)
        self.resblock_norms = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            self.resblock_norms.append(nn.LayerNorm(ch))

        self.conv_post = weight_norm(CausalConv1d(ch, istft_params["n_fft"] + 2, 7, 1, causal_type='left'))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))

        # Custom CoreML-compatible ISTFT
        # Renamed to avoid naming conflict with torch.istft during conversion
        self.custom_istft = CoreMLISTFT(
            n_fft=istft_params["n_fft"],
            hop_length=istft_params["hop_len"],
            win_length=istft_params["n_fft"],
            window='hann',
            center=False
        )

        self.conv_pre_look_right = conv_pre_look_right
        self.f0_predictor = f0_predictor

    def _stft(self, x):
        """STFT for source processing."""
        # Note: This still uses torch.stft which IS supported by CoreML
        window = torch.hann_window(self.istft_params["n_fft"]).to(x.device)
        spec = torch.stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=window,
            return_complex=True
        )
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def decode(self, x, s=torch.zeros(1, 1, 0), finalize=True):
        """
        Decode mel features to audio.

        Args:
            x: Mel features [B, 80, T]
            s: Source signal [B, 1, samples]
            finalize: Whether this is the final chunk

        Returns:
            audio: Waveform [B, samples]
        """
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))

        if finalize is True:
            x = self.conv_pre(x)
        else:
            x = self.conv_pre(x[:, :, :-self.conv_pre_look_right], x[:, :, -self.conv_pre_look_right:])
            s_stft_real = s_stft_real[:, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
            s_stft_imag = s_stft_imag[:, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]

        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # Fusion with source
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

            # Apply LayerNorm to prevent exponential amplification
            # LayerNorm expects [B, T, C], we have [B, C, T]
            x = self.resblock_norms[i](x.transpose(1, 2)).transpose(1, 2)

        x = F.leaky_relu(x)
        x = self.conv_post(x)

        # Split into magnitude and phase
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])

        # Use custom ISTFT (CoreML-compatible)
        x = self.custom_istft(magnitude, phase)

        if finalize is False:
            x = x[:, :-int(np.prod(self.upsample_rates) * self.istft_params['hop_len'])]

        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def inference(self, speech_feat, finalize=True):
        """
        Inference from mel spectrogram to audio.

        Args:
            speech_feat: Mel spectrogram [B, 80, T]
            finalize: Whether this is the final chunk

        Returns:
            generated_speech: Audio waveform [B, samples]
            s: Source signal [B, 1, samples]
        """
        # Mel -> F0
        self.f0_predictor.to(torch.float64)
        f0 = self.f0_predictor(speech_feat.to(torch.float64), finalize=finalize).to(speech_feat)

        # F0 -> Source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, 1, samples]
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)

        # Decode to audio
        if finalize is True:
            generated_speech = self.decode(x=speech_feat, s=s, finalize=finalize)
        else:
            generated_speech = self.decode(x=speech_feat[:, :, :-self.f0_predictor.condnet[0].causal_padding], s=s, finalize=finalize)

        return generated_speech, s
