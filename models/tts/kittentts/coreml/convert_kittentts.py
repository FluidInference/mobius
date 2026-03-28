#!/usr/bin/env python3
"""
Convert KittenTTS Nano ONNX model to CoreML (.mlpackage).

KittenTTS is a distilled Kokoro/StyleTTS2 model (15M params, INT8 quantized ONNX).
This script:
  1. Extracts and dequantizes weights from the ONNX model
  2. Reconstructs the PyTorch model architecture (matching Kokoro's structure)
  3. Loads the dequantized weights
  4. Traces and converts to CoreML via coremltools

Requirements (Python 3.10):
  pip install torch coremltools onnx numpy huggingface_hub

Usage:
  python convert_kittentts.py --extract-only   # Extract dequantized weights
  python convert_kittentts.py --seconds 5      # Full CoreML conversion
"""

import argparse
import math
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: Extract and dequantize ONNX weights
# ---------------------------------------------------------------------------

def extract_onnx_weights(model_path: str) -> dict[str, np.ndarray]:
    """Extract all initializer tensors from ONNX model, dequantizing INT8 weights."""
    model = onnx.load(model_path)

    raw = {}
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        raw[init.name] = arr

    weights = {}

    for name, arr in raw.items():
        if name.startswith("kmodel."):
            weights[name] = arr.astype(np.float32) if arr.dtype != np.float32 else arr

    # Handle quantized weights: look for _quantized suffix
    quantized_keys = [k for k in raw.keys() if "_quantized" in k]
    for qk in quantized_keys:
        base = qk.replace("_quantized", "")
        scale_key = base + "_scale"
        zp_key = base + "_zero_point"

        q_data = raw[qk]
        if scale_key in raw:
            scale = raw[scale_key]
            zp = raw.get(zp_key, np.zeros_like(scale, dtype=q_data.dtype))
            q_float = q_data.astype(np.float32)
            s_float = scale.astype(np.float32)
            z_float = zp.astype(np.float32)
            if s_float.ndim == 0:
                pass
            elif s_float.ndim < q_float.ndim:
                shape = [1] * q_float.ndim
                shape[0] = s_float.shape[0]
                s_float = s_float.reshape(shape)
                z_float = z_float.reshape(shape)
            dequantized = (q_float - z_float) * s_float
            clean_name = qk.replace("_quantized", "")
            weights[clean_name] = dequantized
        else:
            weights[qk] = q_data.astype(np.float32)

    # Collect ONNX anonymous weights (onnx::MatMul_*, onnx::LSTM_*, etc.)
    for name, arr in raw.items():
        if name.startswith("onnx::") and name not in weights:
            if arr.dtype == np.float16:
                weights[name] = arr.astype(np.float32)
            elif arr.dtype != np.float32:
                weights[name] = arr.astype(np.float32)
            else:
                weights[name] = arr

    return weights


def extract_onnx_lstm_weights(model_path: str) -> list[dict]:
    """Extract LSTM weight mappings from ONNX graph in order.

    Returns list of dicts with keys: name, W_key, R_key, B_key
    where W/R/B_key are the initializer names for the DynamicQuantizeLSTM inputs.
    """
    model = onnx.load(model_path)
    lstm_info = []
    for node in model.graph.node:
        if node.op_type == "DynamicQuantizeLSTM":
            inputs = list(node.input)
            # DynamicQuantizeLSTM inputs: X, W, R, B, sequence_lens, initial_h, ...
            lstm_info.append({
                "name": node.name,
                "W_key": inputs[1],  # W (quantized)
                "R_key": inputs[2],  # R (quantized)
                "B_key": inputs[3],  # B (float)
            })
    return lstm_info


# ---------------------------------------------------------------------------
# Step 2: PyTorch model architecture (matching KittenTTS Nano dims)
# ---------------------------------------------------------------------------

class AdaIN1d(nn.Module):
    """Adaptive Instance Normalization (1D)."""
    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        gamma, beta = torch.chunk(h, 2, dim=-1)
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)
        return (1 + gamma) * self.norm(x) + beta


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization used in predictor text encoder."""
    def __init__(self, d_model: int, style_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(style_dim, d_model * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        gamma, beta = torch.chunk(h, 2, dim=-1)
        return (1 + gamma) * self.norm(x) + beta


class ResBlock1D(nn.Module):
    """Residual block with AdaIN for F0/N predictor (matches Kokoro AdainResBlk1d)."""
    def __init__(self, in_ch: int, out_ch: int, style_dim: int, upsample: bool = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = AdaIN1d(in_ch, style_dim)
        self.norm2 = AdaIN1d(out_ch, style_dim)
        self.upsample = upsample
        if upsample:
            self.pool = nn.ConvTranspose1d(in_ch, in_ch, 3, stride=2,
                                            groups=in_ch, padding=1, output_padding=1)
        self.conv1x1 = None
        if in_ch != out_ch:
            self.conv1x1 = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.conv1x1 is not None:
            x = self.conv1x1(x)
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        residual = self._shortcut(x)
        out = self.norm1(x, s)
        out = F.leaky_relu(out, 0.2)
        if self.upsample:
            out = self.pool(out)
        out = self.conv1(out)
        out = self.norm2(out, s)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        return (out + residual) * torch.rsqrt(torch.tensor(2.0, dtype=torch.float32))


class DecodeBlock(nn.Module):
    """Decoder block with AdaIN (matches Kokoro AdainResBlk1d)."""
    def __init__(self, in_ch: int, out_ch: int, style_dim: int, upsample: bool = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = AdaIN1d(in_ch, style_dim)
        self.norm2 = AdaIN1d(out_ch, style_dim)
        self.upsample = upsample
        if upsample:
            self.pool = nn.ConvTranspose1d(in_ch, in_ch, 3, stride=2,
                                            groups=in_ch, padding=1, output_padding=1)
        self.conv1x1 = None
        if in_ch != out_ch:
            self.conv1x1 = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.conv1x1 is not None:
            x = self.conv1x1(x)
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        residual = self._shortcut(x)
        out = self.norm1(x, s)
        out = F.leaky_relu(out, 0.2)
        if self.upsample:
            out = self.pool(out)
        out = self.conv1(out)
        out = self.norm2(out, s)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        return (out + residual) * torch.rsqrt(torch.tensor(2.0, dtype=torch.float32))


def snake_activation(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation: x + (1/alpha) * sin²(alpha * x)."""
    return x + (1.0 / alpha) * torch.sin(alpha * x).pow(2)


class NoiseResBlock(nn.Module):
    """Residual block used in noise path of generator, with AdaIN + Snake activation."""
    def __init__(self, channels: int, style_dim: int, kernel_size: int = 3,
                 dilations: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.adain1 = nn.ModuleList()
        self.adain2 = nn.ModuleList()
        self.alpha1 = nn.ParameterList()
        self.alpha2 = nn.ParameterList()
        for d in dilations:
            self.convs1.append(nn.Conv1d(channels, channels, kernel_size,
                                         dilation=d, padding=(kernel_size * d - d) // 2))
            self.convs2.append(nn.Conv1d(channels, channels, kernel_size,
                                         dilation=1, padding=kernel_size // 2))
            self.adain1.append(AdaIN1d(channels, style_dim))
            self.adain2.append(AdaIN1d(channels, style_dim))
            self.alpha1.append(nn.Parameter(torch.ones(1, channels, 1)))
            self.alpha2.append(nn.Parameter(torch.ones(1, channels, 1)))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for c1, c2, a1, a2, al1, al2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            xt = a1(x, s)
            xt = snake_activation(xt, al1)
            xt = c1(xt)
            xt2 = a2(xt, s)
            xt2 = snake_activation(xt2, al2)
            xt2 = c2(xt2)
            x = xt2 + x
        return x


class AdaINResBlock1(nn.Module):
    """HiFi-GAN style residual block with AdaIN and Snake activation."""
    def __init__(self, channels: int, kernel_size: int, dilations: tuple, style_dim: int):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.adain1 = nn.ModuleList()
        self.adain2 = nn.ModuleList()
        self.alpha1 = nn.ParameterList()
        self.alpha2 = nn.ParameterList()
        for d in dilations:
            self.convs1.append(nn.Conv1d(channels, channels, kernel_size, dilation=d,
                                         padding=(kernel_size * d - d) // 2))
            self.convs2.append(nn.Conv1d(channels, channels, kernel_size, dilation=1,
                                         padding=(kernel_size - 1) // 2))
            self.adain1.append(AdaIN1d(channels, style_dim))
            self.adain2.append(AdaIN1d(channels, style_dim))
            self.alpha1.append(nn.Parameter(torch.ones(1, channels, 1)))
            self.alpha2.append(nn.Parameter(torch.ones(1, channels, 1)))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for c1, c2, a1, a2, al1, al2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            xt = a1(x, s)
            xt = snake_activation(xt, al1)
            xt = c1(xt)
            xt2 = a2(xt, s)
            xt2 = snake_activation(xt2, al2)
            xt2 = c2(xt2)
            x = xt2 + x
        return x


class ConvSTFT(nn.Module):
    """Conv1d-based STFT/iSTFT for CoreML compatibility (no torch.stft)."""
    def __init__(self, n_fft: int = 20, hop_length: int = 5, win_length: int = 20):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_bins = n_fft // 2 + 1  # 11

        # Forward DFT weights: [n_bins, 1, n_fft]
        # These are initialized from ONNX weights (pre-computed DFT basis * window)
        self.weight_forward_real = nn.Parameter(torch.zeros(self.n_bins, 1, n_fft))
        self.weight_forward_imag = nn.Parameter(torch.zeros(self.n_bins, 1, n_fft))
        # Inverse DFT weights: [n_bins, 1, n_fft]
        self.weight_backward_real = nn.Parameter(torch.zeros(self.n_bins, 1, n_fft))
        self.weight_backward_imag = nn.Parameter(torch.zeros(self.n_bins, 1, n_fft))

    def transform(self, x: torch.Tensor):
        """Forward STFT: x [B, T] -> (mag [B, n_bins, frames], phase [B, n_bins, frames])"""
        x = x.unsqueeze(1)  # [B, 1, T]
        real = F.conv1d(x, self.weight_forward_real, stride=self.hop_length)
        imag = F.conv1d(x, self.weight_forward_imag, stride=self.hop_length)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-9)
        phase = torch.atan2(imag, real)
        return mag, phase

    def inverse(self, mag: torch.Tensor, phase: torch.Tensor):
        """Inverse STFT: (mag, phase) -> audio [B, 1, T]"""
        real = mag * torch.cos(phase)  # [B, n_bins, frames]
        imag = mag * torch.sin(phase)
        # Inverse via transposed conv (overlap-add)
        x_real = F.conv_transpose1d(real, self.weight_backward_real, stride=self.hop_length)
        x_imag = F.conv_transpose1d(imag, self.weight_backward_imag, stride=self.hop_length)
        return x_real - x_imag  # [B, 1, T]


class SineGenDeterministic(nn.Module):
    """Deterministic sine generation for source-filter vocoder."""
    def __init__(self, sampling_rate: int = 24000, harmonic_num: int = 8):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1  # 9
        self.sine_amp = 0.1
        self.noise_std = 0.003
        self.voiced_threshold = 0.0

    def forward(self, f0: torch.Tensor, random_phases: torch.Tensor,
                source_noise: torch.Tensor) -> torch.Tensor:
        """f0: [B, T, 1], random_phases: [B, 9], source_noise: [B, T, 9]"""
        uv = torch.sigmoid((f0 - self.voiced_threshold) * 0.5)  # [B, T, 1]
        harmonic_nums = torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype)
        fn = f0 * harmonic_nums.view(1, 1, -1)  # [B, T, 9]
        rad_values = fn / self.sampling_rate
        rad_values[:, 0, :] = rad_values[:, 0, :] + random_phases.squeeze(1)

        # Compute phase with wrapping to prevent fp32 precision loss in CoreML.
        # Without wrapping, cumsum grows to thousands of cycles for high harmonics,
        # and small fp32 differences between CoreML and PyTorch compound into
        # audible phase drift (harmonic 9 correlation drops to 0.79).
        # Fix: reshape into [B, n_frames, samples_per_frame, 9], cumsum within
        # each frame (max 300 steps), then add inter-frame carry.
        T = rad_values.shape[1]
        SPF = 300  # samples per F0 frame (upsample factor)
        n_frames = T // SPF
        tail = T - n_frames * SPF

        # Main frames: [B, n_frames, SPF, 9]
        main = rad_values[:, :n_frames * SPF, :].reshape(1, n_frames, SPF, self.dim)
        # Per-frame cumsum (only 300 steps, max value ~22.5 for harmonic 9)
        frame_cumsum = torch.cumsum(main, dim=2)  # [B, n_frames, SPF, 9]
        # Per-frame total phase increment
        frame_totals = frame_cumsum[:, :, -1:, :]  # [B, n_frames, 1, 9]
        # Inter-frame carry: cumsum of frame totals, shifted by 1
        inter_carry = torch.cumsum(frame_totals, dim=1)  # [B, n_frames, 1, 9]
        inter_carry = F.pad(inter_carry[:, :-1, :, :], (0, 0, 0, 0, 1, 0))  # shift right
        # Wrap the inter-frame carry to keep values small
        inter_carry = inter_carry - torch.floor(inter_carry)
        # Combine
        phase_accum_main = frame_cumsum + inter_carry  # [B, n_frames, SPF, 9]
        phase_accum_main = phase_accum_main.reshape(1, n_frames * SPF, self.dim)

        if tail > 0:
            tail_data = rad_values[:, n_frames * SPF:, :]
            tail_cumsum = torch.cumsum(tail_data, dim=1)
            last_carry = inter_carry[:, -1:, :, :].squeeze(2) + frame_totals[:, -1:, :, :].squeeze(2)
            last_carry = last_carry - torch.floor(last_carry)
            tail_cumsum = tail_cumsum + last_carry
            phase_accum = torch.cat([phase_accum_main, tail_cumsum], dim=1)
        else:
            phase_accum = phase_accum_main

        phase_wrapped = (phase_accum - torch.floor(phase_accum)) * 2 * math.pi
        sine_waves = torch.sin(phase_wrapped) * self.sine_amp
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * source_noise[:, :f0.shape[1], :]
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(nn.Module):
    """Harmonic-plus-noise source module."""
    def __init__(self, sampling_rate: int = 24000, harmonic_num: int = 8):
        super().__init__()
        self.l_sin_gen = SineGenDeterministic(sampling_rate, harmonic_num)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor, random_phases: torch.Tensor,
                source_noise: torch.Tensor) -> torch.Tensor:
        sine_wavs = self.l_sin_gen(f0, random_phases, source_noise)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class Generator(nn.Module):
    """ISTFTNet generator for KittenTTS Nano."""
    def __init__(self, style_dim: int = 128):
        super().__init__()
        self.num_upsamples = 2
        self.num_kernels = 2
        self.post_n_fft = 20

        # Upsample layers
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(256, 128, 20, stride=10, padding=5),
            nn.ConvTranspose1d(128, 64, 12, stride=6, padding=3),
        ])

        # Noise convolutions (from harmonic source)
        self.noise_convs = nn.ModuleList([
            nn.Conv1d(22, 128, 12, stride=6, padding=3),  # [22, 128, 12]
            nn.Conv1d(22, 64, 1),   # [22, 64, 1]
        ])

        # Noise residual blocks
        self.noise_res = nn.ModuleList([
            NoiseResBlock(128, style_dim, kernel_size=7, dilations=(1, 3, 5)),
            NoiseResBlock(64, style_dim, kernel_size=11, dilations=(1, 3, 5)),
        ])

        # Main path resblocks (2 per upsample stage, kernel=3, dilations=(1,3,5))
        self.resblocks = nn.ModuleList([
            AdaINResBlock1(128, 3, (1, 3, 5), style_dim),  # stage 0, block 0
            AdaINResBlock1(128, 3, (1, 3, 5), style_dim),  # stage 0, block 1
            AdaINResBlock1(64, 3, (1, 3, 5), style_dim),   # stage 1, block 0
            AdaINResBlock1(64, 3, (1, 3, 5), style_dim),   # stage 1, block 1
        ])

        # Post-processing
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.conv_post = nn.Conv1d(64, 22, 7, padding=3)  # post_n_fft + 2 = 22

        # STFT for source-filter
        self.stft = ConvSTFT(n_fft=20, hop_length=5, win_length=20)

        # F0 upsample (frame -> sample rate)
        self.f0_upsamp = nn.Upsample(scale_factor=300)

        # Source module
        self.m_source = SourceModuleHnNSF(sampling_rate=24000, harmonic_num=8)

    def forward(self, x: torch.Tensor, s: torch.Tensor, f0: torch.Tensor,
                random_phases: torch.Tensor, source_noise: torch.Tensor) -> torch.Tensor:
        # f0: [B, T_frames]
        f0_up = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_samples, 1]
        har_source = self.m_source(f0_up, random_phases, source_noise)  # [B, T_samples, 1]
        har_source = har_source.transpose(1, 2).squeeze(1)  # [B, T_samples]
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)  # [B, 22, frames]

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            # Align lengths
            if x_source.shape[2] != x.shape[2]:
                if x_source.shape[2] < x.shape[2]:
                    x_source = F.pad(x_source, (0, x.shape[2] - x_source.shape[2]))
                else:
                    x_source = x_source[:, :, :x.shape[2]]
            x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)

        # Split into magnitude and phase
        x_mag = x[:, :self.post_n_fft // 2 + 1, :]
        x_mag = torch.clamp(x_mag, min=-10, max=10)
        spec = torch.exp(x_mag)
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        audio = self.stft.inverse(spec, phase)
        return audio


class ALBERTEmbeddings(nn.Module):
    """ALBERT embeddings (word + position + token_type + LayerNorm)."""
    def __init__(self, vocab_size: int = 178, embed_dim: int = 128,
                 max_position: int = 512, type_vocab_size: int = 2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_position, embed_dim)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embed_dim)
        self.LayerNorm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)
        token_type_ids = torch.zeros_like(input_ids)

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class ALBERTAttention(nn.Module):
    """ALBERT multi-head self-attention."""
    def __init__(self, hidden_size: int = 768, num_heads: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        projected = self.dense(context)
        output = self.LayerNorm(projected + hidden_states)
        return output


class ALBERTLayer(nn.Module):
    """Single ALBERT layer (shared weights, repeated N times)."""
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, ffn_size: int = 2048):
        super().__init__()
        self.attention = ALBERTAttention(hidden_size, num_heads)
        self.ffn = nn.Linear(hidden_size, ffn_size)
        self.ffn_output = nn.Linear(ffn_size, hidden_size)
        self.full_layer_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        ffn_output = F.gelu(self.ffn(attention_output))
        ffn_output = self.ffn_output(ffn_output)
        output = self.full_layer_layer_norm(ffn_output + attention_output)
        return output


class ALBERTEncoder(nn.Module):
    """ALBERT encoder with shared layer weights."""
    def __init__(self, embed_dim: int = 128, hidden_size: int = 768,
                 num_heads: int = 12, ffn_size: int = 2048,
                 num_hidden_layers: int = 12):
        super().__init__()
        self.embedding_hidden_mapping_in = nn.Linear(embed_dim, hidden_size)
        self.albert_layer = ALBERTLayer(hidden_size, num_heads, ffn_size)
        self.num_hidden_layers = num_hidden_layers

    def forward(self, embeddings: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding_hidden_mapping_in(embeddings)
        for _ in range(self.num_hidden_layers):
            hidden_states = self.albert_layer(hidden_states, attention_mask)
        return hidden_states


class ALBERTModel(nn.Module):
    """Full ALBERT model for KittenTTS."""
    def __init__(self):
        super().__init__()
        self.embeddings = ALBERTEmbeddings(
            vocab_size=178, embed_dim=128, max_position=512, type_vocab_size=2
        )
        self.encoder = ALBERTEncoder(
            embed_dim=128, hidden_size=768, num_heads=12,
            ffn_size=2048, num_hidden_layers=12
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(input_ids)
        # Create extended attention mask [B, 1, 1, seq_len]
        extended_mask = attention_mask[:, None, None, :].to(dtype=embeddings.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0
        output = self.encoder(embeddings, extended_mask)
        return output


class TextEncoder(nn.Module):
    """Text encoder: embedding -> CNN (Conv+LayerNorm+LeakyReLU) -> LSTM -> proj."""
    def __init__(self, vocab_size: int = 178, embed_dim: int = 128,
                 cnn_channels: int = 128, cnn_layers: int = 6,
                 lstm_hidden: int = 64, proj_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cnn = nn.ModuleList()
        for _ in range(cnn_layers):
            # ONNX graph: Conv -> Transpose -> LayerNorm -> Transpose -> LeakyReLU
            self.cnn.append(nn.Sequential(
                nn.Conv1d(cnn_channels, cnn_channels, 5, padding=2),
                nn.LayerNorm(cnn_channels),
                nn.LeakyReLU(negative_slope=0.01),
            ))
        self.lstm = nn.LSTM(
            cnn_channels, lstm_hidden, num_layers=1,
            bidirectional=True, batch_first=True,
        )
        self.text_proj = nn.Linear(lstm_hidden * 2, proj_dim)

    def forward(self, input_ids: torch.Tensor, input_lengths: torch.Tensor,
                text_mask_bool: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # [B, T, 128]
        x = x.transpose(1, 2)  # [B, 128, T]
        m = text_mask_bool.unsqueeze(1)  # [B, 1, T]
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            conv = c[0]
            ln = c[1]
            lrelu = c[2]
            x = conv(x)  # [B, C, T]
            # LayerNorm expects [B, T, C], so transpose
            x = ln(x.transpose(1, 2)).transpose(1, 2)  # [B, C, T]
            x = lrelu(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, 128]
        batch_size = x.shape[0]
        h0 = torch.zeros(2, batch_size, self.lstm.hidden_size, dtype=x.dtype, device=x.device)
        c0 = torch.zeros(2, batch_size, self.lstm.hidden_size, dtype=x.dtype, device=x.device)
        x, _ = self.lstm(x, (h0, c0))  # [B, T, 128]
        x = x.transpose(-1, -2)  # [B, 128, T]
        x.masked_fill_(m, 0.0)
        return x


class PredictorTextEncoder(nn.Module):
    """Predictor text encoder: alternating LSTM + AdaLayerNorm layers."""
    def __init__(self, d_model: int = 128, style_dim: int = 128,
                 lstm_hidden: int = 64, num_blocks: int = 6):
        super().__init__()
        self.d_model = d_model
        self.sty_dim = style_dim
        self.lstm_hidden = lstm_hidden

        self.lstms = nn.ModuleList()
        for i in range(num_blocks):
            # LSTM: input = d_model + style_dim = 256, hidden = 64, bidirectional -> output = 128
            input_size = d_model + style_dim
            self.lstms.append(
                nn.LSTM(input_size, lstm_hidden, num_layers=1,
                        bidirectional=True, batch_first=True)
            )
            # AdaLayerNorm
            self.lstms.append(AdaLayerNorm(d_model, style_dim))

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                input_lengths: torch.Tensor,
                text_mask_bool: torch.Tensor) -> torch.Tensor:
        # x: [B, d_model, T] from d_en
        masks = text_mask_bool  # [B, T]
        x = x.permute(2, 0, 1)  # [T, B, d_model]
        s = style.expand(x.shape[0], x.shape[1], -1)  # [T, B, style_dim]

        # Concat style
        x = torch.cat([x, s], axis=-1)  # [T, B, d_model + style_dim]
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)  # [B, T, d_model + style_dim]
        x = x.transpose(-1, -2)  # [B, d_model + style_dim, T]

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                # Re-concat style
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)  # [B, T, C]
                batch_size = x.shape[0]
                h0 = torch.zeros(2, batch_size, self.lstm_hidden, dtype=x.dtype, device=x.device)
                c0 = torch.zeros(2, batch_size, self.lstm_hidden, dtype=x.dtype, device=x.device)
                x, _ = block(x, (h0, c0))
                x = x.transpose(-1, -2)  # [B, C, T]
                # Pad if needed
                if x.shape[-1] < masks.shape[-1]:
                    x_pad = torch.zeros([x.shape[0], x.shape[1], masks.shape[-1]],
                                        device=x.device, dtype=x.dtype)
                    x_pad[:, :, :x.shape[-1]] = x
                    x = x_pad

        return x.transpose(-1, -2)  # [B, T, d_model]


class F0NPredictor(nn.Module):
    """F0 or N (noise amplitude) predictor."""
    def __init__(self, style_dim: int = 128):
        super().__init__()
        # 3 ResBlocks: [128->128], [128->64, upsample], [64->64]
        self.blocks = nn.ModuleList([
            ResBlock1D(128, 128, style_dim, upsample=False),
            ResBlock1D(128, 64, style_dim, upsample=True),
            ResBlock1D(64, 64, style_dim, upsample=False),
        ])
        self.proj = nn.Conv1d(64, 1, 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, s)
        return self.proj(x).squeeze(1)  # [B, T]


class Predictor(nn.Module):
    """Duration, F0, and N predictor."""
    def __init__(self, d_model: int = 128, style_dim: int = 128):
        super().__init__()
        self.text_encoder = PredictorTextEncoder(d_model, style_dim)
        self.lstm = nn.LSTM(d_model + style_dim, 64, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.duration_proj = nn.Linear(d_model, 50)
        # Shared LSTM for F0/N: input = d_model + style_dim, hidden = 64, bidir -> 128
        self.shared = nn.LSTM(d_model + style_dim, 64, num_layers=1,
                              bidirectional=True, batch_first=True)
        self.F0 = F0NPredictor(style_dim)
        self.N = F0NPredictor(style_dim)

    def F0Ntrain(self, en: torch.Tensor, style: torch.Tensor):
        """Predict F0 and N from encoded features.
        en: [B, d_model + style_dim, frames]
        """
        # Shared LSTM processes en first
        batch_size = en.shape[0]
        h0 = torch.zeros(2, batch_size, 64, dtype=en.dtype, device=en.device)
        c0 = torch.zeros(2, batch_size, 64, dtype=en.dtype, device=en.device)
        x, _ = self.shared(en.transpose(-1, -2), (h0, c0))  # [B, frames, 128]
        x = x.transpose(-1, -2)  # [B, 128, frames]
        F0 = self.F0(x, style)
        N = self.N(x, style)
        return F0, N


class Decoder(nn.Module):
    """Decoder: encode + decode blocks + generator."""
    def __init__(self, style_dim: int = 128):
        super().__init__()
        # encode: input = 512 (text_enc) + 1 (F0) + 1 (N) = 514
        self.encode = DecodeBlock(514, 256, style_dim)
        # asr_res: project 512 -> 64
        self.asr_res = nn.Conv1d(512, 64, 1)
        # F0/N conv: stride=2 to downsample from 2x upsampled F0/N predictions
        self.F0_conv = nn.Conv1d(1, 1, 3, stride=2, padding=1)
        self.N_conv = nn.Conv1d(1, 1, 3, stride=2, padding=1)
        # 4 decode blocks: input = 256 + 64 + 1 + 1 = 322, output = 256
        # last block has pool=True
        self.decode = nn.ModuleList([
            DecodeBlock(322, 256, style_dim),
            DecodeBlock(322, 256, style_dim),
            DecodeBlock(322, 256, style_dim),
            DecodeBlock(322, 256, style_dim, upsample=True),
        ])
        self.generator = Generator(style_dim)


# ---------------------------------------------------------------------------
# Step 3: Complete model wrapper for tracing
# ---------------------------------------------------------------------------

class KittenTTSComplete(nn.Module):
    """Complete KittenTTS model with fixed shapes for CoreML tracing."""
    def __init__(self, fixed_total_frames: int = None, samples_per_frame: int = 600):
        super().__init__()
        self.style_dim = 128

        # BERT
        self.bert = ALBERTModel()
        self.bert_encoder = nn.Linear(768, 128)

        # Text encoder
        self.text_encoder = TextEncoder()

        # Predictor
        self.predictor = Predictor()

        # Decoder
        self.decoder = Decoder()

        self.fixed_total_frames = fixed_total_frames
        self.samples_per_frame = samples_per_frame

    def create_alignment_matrix(self, pred_dur, device, max_frames):
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
        pred_dur = torch.round(pred_dur).clamp(min=0)
        pred_dur_int = pred_dur.to(torch.int32)
        cum = torch.cumsum(pred_dur_int, dim=0)
        starts = F.pad(cum[:-1], (1, 0), value=0)
        max_frames = int(max_frames)
        frame_grid = torch.arange(max_frames, device=device, dtype=torch.int32).unsqueeze(0)
        starts = starts.to(torch.int32).unsqueeze(1)
        ends = cum.to(torch.int32).unsqueeze(1)
        mask = (frame_grid >= starts) & (frame_grid < ends)
        total = torch.clamp(cum[-1], max=max_frames).to(torch.int32)
        valid_cols = (frame_grid < total.unsqueeze(0))
        mask = mask & valid_cols
        return mask.to(torch.float32).unsqueeze(0)

    def forward(self, input_ids, ref_s, random_phases, attention_mask, source_noise):
        speed = 1
        batch_size, L = input_ids.shape
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.int32)
        text_mask_bool = (attention_mask == 0)
        input_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)

        # BERT forward
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_output).transpose(-1, -2)  # [B, 128, T]

        # Style split: first 128 = decoder style, last 128 = predictor style
        style = ref_s[:, 128:]  # predictor style
        ref_s_style = ref_s[:, :128]  # decoder style

        # Predictor text encoder
        d = self.predictor.text_encoder(d_en, style, input_lengths, text_mask_bool)

        # Duration prediction
        # d already includes style from the last AdaLayerNorm block in text_encoder
        # so it's [B, T, d_model + style_dim] = [B, T, 256] matching predictor.lstm input_size
        lstm_layers = 2  # bidirectional
        h0 = torch.zeros(lstm_layers, batch_size, 64, dtype=d.dtype, device=d.device)
        c0 = torch.zeros(lstm_layers, batch_size, 64, dtype=d.dtype, device=d.device)
        x, _ = self.predictor.lstm(d, (h0, c0))
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1)
        valid = attention_mask.to(dtype=pred_dur.dtype)
        pred_dur = pred_dur * valid

        total_frames = pred_dur.sum(dim=1)
        audio_length_samples = (total_frames * self.samples_per_frame).to(torch.int32)

        if self.fixed_total_frames is not None:
            total_frames_int = self.fixed_total_frames
        else:
            total_frames_int = int(torch.clamp(total_frames[0], min=1).item())

        # Alignment matrix
        pred_aln_trg = self.create_alignment_matrix(
            pred_dur[0], device=input_ids.device, max_frames=total_frames_int
        )

        # Expand encoded features to frame level
        en = d.transpose(-1, -2) @ pred_aln_trg  # [B, 128, frames]

        # F0 and N prediction
        F0_pred, N_pred = self.predictor.F0Ntrain(en, style)

        # Text encoder forward
        t_en = self.text_encoder(input_ids, input_lengths, text_mask_bool)
        # text_proj
        t_en = self.text_encoder.text_proj(t_en.transpose(-1, -2)).transpose(-1, -2)  # [B, 512, T]
        asr = t_en @ pred_aln_trg  # [B, 512, frames]

        # Decoder
        F0_processed = self.decoder.F0_conv(F0_pred.unsqueeze(1))
        N_processed = self.decoder.N_conv(N_pred.unsqueeze(1))

        x = torch.cat([asr, F0_processed, N_processed], dim=1)  # [B, 514, frames]
        x_encoded = self.decoder.encode(x, ref_s_style)

        asr_res = self.decoder.asr_res(asr)  # [B, 64, frames]

        x_current = x_encoded
        concat_residuals = True
        for decode_block in self.decoder.decode:
            if concat_residuals:
                x_input = torch.cat([x_current, asr_res, F0_processed, N_processed], dim=1)
            else:
                x_input = x_current
            x_current = decode_block(x_input, ref_s_style)
            if decode_block.upsample:
                concat_residuals = False

        # Generator
        audio = self.decoder.generator(
            x_current, ref_s_style, F0_pred, random_phases, source_noise
        )

        # Zero out samples past the predicted length
        sample_idx = torch.arange(audio.shape[-1], device=audio.device)
        mask = (sample_idx < audio_length_samples.unsqueeze(-1)).unsqueeze(0).to(audio.dtype)
        audio = audio * mask

        return audio, audio_length_samples, pred_dur


# ---------------------------------------------------------------------------
# Step 4: Weight loading
# ---------------------------------------------------------------------------

def load_weights_from_onnx(model: KittenTTSComplete, weights: dict,
                           lstm_info: list, verbose: bool = True):
    """Load dequantized ONNX weights into the PyTorch model."""

    def clean(name):
        """Remove kmodel. prefix from weight name."""
        if name.startswith("kmodel."):
            return name[len("kmodel."):]
        return name

    sd = model.state_dict()
    loaded = set()
    missing = []

    def _set(key, value):
        if key in sd:
            t = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            if sd[key].shape != t.shape:
                if verbose:
                    print(f"  SHAPE MISMATCH: {key}: expected {sd[key].shape}, got {t.shape}")
                return False
            sd[key] = t
            loaded.add(key)
            return True
        else:
            missing.append(key)
            return False

    # --- Named kmodel weights ---
    for wname, warr in weights.items():
        cname = clean(wname)
        # Skip quantized/scale/zero_point duplicates
        if "_quantized" in cname or "_scale" in cname or "_zero_point" in cname:
            continue
        # Skip onnx:: keys (handled separately)
        if cname.startswith("onnx::"):
            continue
        # Map weight names to model state dict keys
        mapped = map_weight_name(cname)
        if mapped:
            arr = warr
            # ONNX stores Linear weights as [in, out] but PyTorch uses [out, in]
            # This affects all fc.weight in AdaIN1d, AdaLayerNorm, etc.
            if mapped.endswith(".fc.weight") and arr.ndim == 2:
                arr = arr.T
            _set(mapped, arr)

    # --- ONNX anonymous weights ---
    # BERT attention weights (fp16 stored as fp32)
    onnx_bert_map = {
        # ONNX MatMul: x @ W where W is [in_features, out_features]
        # PyTorch Linear stores weight as [out_features, in_features]
        # So we need to transpose: W.T
        "onnx::MatMul_7607": "bert.encoder.albert_layer.attention.query.weight",
        "onnx::MatMul_7610": "bert.encoder.albert_layer.attention.key.weight",
        "onnx::MatMul_7613": "bert.encoder.albert_layer.attention.value.weight",
        "onnx::MatMul_7617": "bert.encoder.albert_layer.attention.dense.weight",
        "onnx::MatMul_7618": "bert.encoder.albert_layer.ffn.weight",
        "onnx::MatMul_7619": "bert.encoder.albert_layer.ffn_output.weight",
    }
    for onnx_key, model_key in onnx_bert_map.items():
        if onnx_key in weights:
            w = weights[onnx_key].copy()
            # ONNX MatMul: x @ W, W is [in, out]. PyTorch: weight is [out, in]. Need W.T.
            _set(model_key, w.T)

    # embedding_hidden_mapping_in (Linear(128, 768))
    # ONNX MatMulInteger uses onnx::MatMul_7606 [128, 768] (quantized, dequantized).
    # PyTorch Linear(128, 768) weight shape is [768, 128] = W.T
    if "onnx::MatMul_7606" in weights:
        w = weights["onnx::MatMul_7606"].copy()  # [128, 768]
        _set("bert.encoder.embedding_hidden_mapping_in.weight", w.T)  # [768, 128]

    # bert_encoder (Linear(768, 128))
    # ONNX MatMul uses onnx::MatMul_7763 [768, 128] (fp16).
    # PyTorch Linear(768, 128) weight shape is [128, 768] = W.T
    if "onnx::MatMul_7763" in weights:
        w = weights["onnx::MatMul_7763"].copy()  # [768, 128]
        _set("bert_encoder.weight", w.T)  # [128, 768]

    # text_encoder.text_proj
    if "onnx::MatMul_7598" in weights:
        # ONNX: [128, 512] -> PyTorch Linear(128, 512) weight is [512, 128]
        _set("text_encoder.text_proj.weight", weights["onnx::MatMul_7598"].T)

    # predictor.duration_proj
    if "onnx::MatMul_8154" in weights:
        # ONNX: [128, 50] -> PyTorch Linear(128, 50) weight is [50, 128]
        _set("predictor.duration_proj.weight", weights["onnx::MatMul_8154"].T)

    # generator.m_source.l_linear
    if "onnx::MatMul_8321" in weights:
        # ONNX: [9, 1] -> PyTorch Linear(9, 1) weight is [1, 9]
        _set("decoder.generator.m_source.l_linear.weight", weights["onnx::MatMul_8321"].T)

    # --- LSTM weights ---
    # ONNX DynamicQuantizeLSTM format (transposed vs standard):
    # W: [num_dirs, input_size, 4*hidden_size]
    # R: [num_dirs, hidden_size, 4*hidden_size]
    # B: [num_dirs, 8*hidden_size]
    #
    # PyTorch LSTM stores:
    # weight_ih_l{i}: [4*hidden_size, input_size]
    # weight_hh_l{i}: [4*hidden_size, hidden_size]
    # bias_ih_l{i}: [4*hidden_size]
    # bias_hh_l{i}: [4*hidden_size]
    # For bidirectional, _reverse variants

    # Map LSTM nodes to model paths
    lstm_model_paths = [
        "text_encoder.lstm",                    # /text_encoder/lstm/LSTM_quant
        "predictor.text_encoder.lstms.0",       # /text_encoder/lstms.0/LSTM_quant
        "predictor.text_encoder.lstms.2",       # /text_encoder/lstms.2/LSTM_quant
        "predictor.text_encoder.lstms.4",       # /text_encoder/lstms.4/LSTM_quant
        "predictor.text_encoder.lstms.6",       # /text_encoder/lstms.6/LSTM_quant
        "predictor.text_encoder.lstms.8",       # /text_encoder/lstms.8/LSTM_quant
        "predictor.text_encoder.lstms.10",      # /text_encoder/lstms.10/LSTM_quant
        "predictor.lstm",                       # /lstm/LSTM_quant
        "predictor.shared",                      # /shared/LSTM_quant - F0/N shared LSTM
    ]

    for i, (info, model_path) in enumerate(zip(lstm_info, lstm_model_paths)):
        if model_path is None:
            continue

        W_key = info["W_key"]
        R_key = info["R_key"]
        B_key = info["B_key"]

        # Get the dequantized weights (prefer clean name without _quantized suffix)
        W_base = W_key.replace("_quantized", "")
        W = weights.get(W_base, weights.get(W_key, None))

        R_base = R_key.replace("_quantized", "")
        R = weights.get(R_base, weights.get(R_key, None))
        B = weights.get(B_key, None)

        if W is None or R is None or B is None:
            if verbose:
                print(f"  WARNING: Missing LSTM weights for {model_path}")
            continue

        # W: [2, input_size, 4*hidden] -> transpose each direction
        # R: [2, hidden_size, 4*hidden] -> transpose each direction
        # B: [2, 8*hidden] -> split into ih and hh biases
        #
        # ONNX LSTM gate order: [i, o, f, c]
        # PyTorch LSTM gate order: [i, f, g, o]
        # Need to reorder: ONNX[0,1,2,3] -> PyTorch[0,2,3,1]

        hidden_size = B.shape[1] // 8  # 64

        def _reorder_gates(tensor, h):
            """Reorder LSTM gates from ONNX [i,o,f,c] to PyTorch [i,f,g,o]."""
            # tensor has gate dim as one of its axes with size 4*h
            # Split into 4 gates of size h, then reorder
            i_gate = tensor[..., 0*h:1*h]
            o_gate = tensor[..., 1*h:2*h]
            f_gate = tensor[..., 2*h:3*h]
            c_gate = tensor[..., 3*h:4*h]
            return np.concatenate([i_gate, f_gate, c_gate, o_gate], axis=-1)

        for d in range(2):  # forward and reverse
            suffix = f"_l0" if d == 0 else f"_l0_reverse"

            # W[d]: [input, 4*h] -> reorder gates -> transpose -> PyTorch: [4*h, input]
            W_reordered = _reorder_gates(W[d], hidden_size)
            _set(f"{model_path}.weight_ih{suffix}", W_reordered.T)
            # R[d]: [h, 4*h] -> reorder gates -> transpose -> PyTorch: [4*h, h]
            R_reordered = _reorder_gates(R[d], hidden_size)
            _set(f"{model_path}.weight_hh{suffix}", R_reordered.T)
            # B[d]: [8*h] -> reorder gates for both ih and hh biases
            b = B[d]
            b_ih = _reorder_gates(b[:4 * hidden_size].reshape(1, -1), hidden_size).flatten()
            b_hh = _reorder_gates(b[4 * hidden_size:].reshape(1, -1), hidden_size).flatten()
            _set(f"{model_path}.bias_ih{suffix}", b_ih)
            _set(f"{model_path}.bias_hh{suffix}", b_hh)

    # Load state dict
    model.load_state_dict(sd, strict=False)

    if verbose:
        total = len(sd)
        print(f"  Loaded {len(loaded)}/{total} parameters")
        not_loaded = set(sd.keys()) - loaded
        if not_loaded:
            print(f"  Not loaded ({len(not_loaded)}):")
            for k in sorted(not_loaded):
                print(f"    {k}")


def map_weight_name(cname: str) -> str | None:
    """Map ONNX weight name (cleaned) to PyTorch state dict key."""
    # Direct mappings that match 1:1
    # Most weights map directly since we structured the model to match

    # BatchNorm in text_encoder uses gamma/beta instead of weight/bias
    if "text_encoder.cnn" in cname and cname.endswith(".gamma"):
        return cname.replace(".gamma", ".weight")
    if "text_encoder.cnn" in cname and cname.endswith(".beta"):
        return cname.replace(".beta", ".bias")

    # BERT embeddings
    if cname.startswith("bert."):
        # ALBERT encoder layer is at index [0][0] in ONNX but we use albert_layer
        cname = cname.replace(
            "bert.encoder.albert_layer_groups.0.albert_layers.0",
            "bert.encoder.albert_layer"
        )
        return cname

    # bert_encoder (Linear(768, 128)) - just bias, weight is from onnx::MatMul
    if cname == "bert_encoder.bias":
        return "bert_encoder.bias"

    # Predictor
    if cname.startswith("predictor."):
        # predictor.duration_proj.linear_layer.bias -> predictor.duration_proj.bias
        if cname == "predictor.duration_proj.linear_layer.bias":
            return "predictor.duration_proj.bias"
        # F0_proj / N_proj stored with underscore in ONNX but inside F0NPredictor
        # predictor.F0_proj.weight -> predictor.F0.proj.weight
        if cname.startswith("predictor.F0_proj"):
            return cname.replace("predictor.F0_proj", "predictor.F0.proj")
        if cname.startswith("predictor.N_proj"):
            return cname.replace("predictor.N_proj", "predictor.N.proj")
        # F0/N blocks: predictor.F0.{0,1,2}.* -> predictor.F0.blocks.{0,1,2}.*
        if ".F0." in cname or ".N." in cname:
            parts = cname.split(".")
            if len(parts) >= 3 and parts[1] in ("F0", "N"):
                try:
                    idx = int(parts[2])
                    rest = ".".join(parts[3:])
                    return f"predictor.{parts[1]}.blocks.{idx}.{rest}"
                except ValueError:
                    pass
        return cname

    # Decoder
    if cname.startswith("decoder.decoder."):
        # Remove extra "decoder." prefix
        inner = cname[len("decoder.decoder."):]
        # asr_res.0.weight -> asr_res.weight (Sequential index removal)
        if inner.startswith("asr_res.0."):
            inner = inner.replace("asr_res.0.", "asr_res.")
        return f"decoder.{inner}"

    # Text encoder
    if cname.startswith("text_encoder."):
        return cname

    return cname


# ---------------------------------------------------------------------------
# Step 5: Main conversion
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert KittenTTS ONNX to CoreML")
    parser.add_argument("--model", default=None,
                        help="Path to KittenTTS ONNX model (auto-downloads if not specified)")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="Target max audio duration in seconds")
    parser.add_argument("--output", default="kitten_tts_nano.mlpackage",
                        help="Output mlpackage path")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract and save dequantized weights, skip CoreML conversion")
    parser.add_argument("--verify-only", action="store_true",
                        help="Build PyTorch model and verify weight loading, skip CoreML conversion")
    args = parser.parse_args()

    # Get model path
    if args.model is None:
        from huggingface_hub import hf_hub_download
        args.model = hf_hub_download("KittenML/kitten-tts-nano-0.1", "kitten_tts_nano_v0_1.onnx")
        print(f"Using model: {args.model}")

    print(f"Model size: {os.path.getsize(args.model) / 1e6:.1f} MB")

    # Extract weights
    print("\n[1/5] Extracting and dequantizing ONNX weights...")
    weights = extract_onnx_weights(args.model)
    lstm_info = extract_onnx_lstm_weights(args.model)

    kmodel_weights = {k: v for k, v in weights.items() if k.startswith("kmodel.")}
    print(f"  Named model weights: {len(kmodel_weights)}")
    print(f"  ONNX anonymous weights: {sum(1 for k in weights if k.startswith('onnx::'))}")
    print(f"  LSTM nodes: {len(lstm_info)}")

    total_params = sum(np.prod(v.shape) for v in kmodel_weights.values())
    print(f"  Total parameters: {total_params:,.0f}")

    if args.extract_only:
        out_path = args.output.replace(".mlpackage", "_weights.npz")
        np.savez_compressed(out_path, **kmodel_weights)
        print(f"\n  Saved dequantized weights to: {out_path}")

        pt_path = args.output.replace(".mlpackage", "_weights.pt")
        state_dict = {k.replace("kmodel.", ""): torch.from_numpy(v) for k, v in kmodel_weights.items()}
        torch.save(state_dict, pt_path)
        print(f"  Saved PyTorch state dict to: {pt_path}")
        return

    # Compute fixed frame count
    # F0 predictor upsamples 2x, then f0_upsamp does 300x -> 600 samples per frame
    TARGET_FRAMES = int(args.seconds * 24000 / 600)
    TARGET_AUDIO_SAMPLES = TARGET_FRAMES * 600
    MAX_TOKENS = int(args.seconds * 14)  # ~14 tokens per second estimate

    # Build model
    print(f"\n[2/5] Building PyTorch model (max_tokens={MAX_TOKENS}, target_frames={TARGET_FRAMES})...")
    model = KittenTTSComplete(fixed_total_frames=TARGET_FRAMES)
    model.eval()

    # Load weights
    print("\n[3/5] Loading dequantized weights...")
    load_weights_from_onnx(model, weights, lstm_info, verbose=True)

    if args.verify_only:
        print("\n  Verification complete. Model built and weights loaded.")
        return

    # Trace
    print(f"\n[4/5] Tracing model...")
    input_ids = torch.zeros(1, MAX_TOKENS, dtype=torch.long)
    num_real_tokens = min(50, MAX_TOKENS)
    for i in range(num_real_tokens):
        input_ids[0, i] = (i % 177) + 1

    ref_s = torch.randn(1, 256)
    random_phases = torch.randn(1, 9)
    attention_mask = torch.zeros(1, MAX_TOKENS, dtype=torch.int32)
    attention_mask[0, :num_real_tokens] = 1

    L_UP = TARGET_AUDIO_SAMPLES
    source_noise = torch.randn(1, L_UP, 9)

    example_inputs = (input_ids, ref_s, random_phases, attention_mask, source_noise)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)
    print("  Trace complete.")

    # Convert to CoreML
    print("\n[5/5] Converting to CoreML...")
    import coremltools as ct

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
            ct.TensorType(name="ref_s", shape=ref_s.shape, dtype=np.float32),
            ct.TensorType(name="random_phases", shape=random_phases.shape, dtype=np.float32),
            ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int32),
            ct.TensorType(name="source_noise", shape=source_noise.shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="audio", dtype=np.float32),
            ct.TensorType(name="audio_length_samples", dtype=np.int32),
            ct.TensorType(name="pred_dur", dtype=np.float32),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.save(args.output)
    print(f"\n{'=' * 60}")
    print(f"Saved: {args.output}")
    print(f"Inputs: {[i.name for i in mlmodel.get_spec().description.input]}")
    print(f"Outputs: {[o.name for o in mlmodel.get_spec().description.output]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
