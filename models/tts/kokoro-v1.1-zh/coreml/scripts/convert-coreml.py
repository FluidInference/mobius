"""Convert Kokoro-82M-v1.1-zh (Mandarin) to CoreML — fp16+int8pal preset (7 mlpackages).

Adapted from https://github.com/laishere/kokoro-coreml (MIT) by switching the
upstream checkpoint from `hexgrad/Kokoro-82M` (English, n_token=177) to
`hexgrad/Kokoro-82M-v1.1-zh` (Mandarin, n_token=178, Bopomofo + tone-digit
vocab). Architecture (ALBERT, predictor, text_encoder, decoder/generator,
iSTFT) is identical to v1.0 — only the embedding vocab size and weights
change, so the 7-stage trace + RangeDim shape bounds stay byte-for-byte
compatible.

Output: KokoroAlbert, KokoroPostAlbert, KokoroAlignment, KokoroProsody,
KokoroNoise, KokoroVocoder, KokoroTail (plus ref.wav / test.wav for parity).

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python convert-coreml.py \
        --output-dir build/kokoro-v1.1-zh [--max-frames 2000] [--stages all]
"""
import argparse
import math
import pathlib

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kokoro import KModel
from kokoro.custom_stft import CustomSTFT
from kokoro.istftnet import AdainResBlk1d, AdaINResBlock1

# OUTDIR is assigned in main() from --output-dir; module-level placeholder for
# helpers that still reference it.
OUTDIR: pathlib.Path = pathlib.Path("output")

# ── Patch rsqrt for tracing ──
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
def _patched_resblk_forward(self, x, s):
    out = self._residual(x, s)
    out = (out + self._shortcut(x)) * _INV_SQRT2
    return out
AdainResBlk1d.forward = _patched_resblk_forward

# ── Patch Snake: cos identity sin²(αx) = (1 - cos(2αx))/2 (faster on ANE) ──
def _cos_resblock1_forward(self, x, s):
    for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
        xt = n1(x, s)
        cv = torch.cos(xt * (a1 * 2))
        xt = xt + (cv * (-0.5) + 0.5) * (1.0 / a1)
        xt = c1(xt)
        xt = n2(xt, s)
        cv = torch.cos(xt * (a2 * 2))
        xt = xt + (cv * (-0.5) + 0.5) * (1.0 / a2)
        xt = c2(xt)
        x = xt + x
    return x


class CoreMLVocoderDualOutput(nn.Module):
    """Vocoder with dual output: anchor (discarded) + x_pre (for fp32 tail).
    Uses cos Snake (faster on ANE). Audio output keeps ANE graph alive but is
    discarded in favor of x_pre → fp32 tail model for clean audio."""
    def __init__(self, decoder):
        super().__init__()
        self.encode = decoder.encode
        self.decode = decoder.decode
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.asr_res = decoder.asr_res
        gen = decoder.generator
        self.num_kernels = gen.num_kernels
        self.num_upsamples = gen.num_upsamples
        self.ups = gen.ups
        self.resblocks = gen.resblocks

    def forward(self, asr, F0_curve, N, x_source_0, x_source_1, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_feat = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_feat], dim=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N_feat], dim=1)
            x = block(x, s)
            if block.upsample_type != 'none':
                res = False
        noise_sources = [x_source_0, x_source_1]
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = torch.cat([x[:, :, 1:2], x], dim=2)
            x = x + noise_sources[i]
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        x_pre = F.leaky_relu(x)
        # Reduce over explicit dims so the MIL keeps rank >= 1 throughout.
        # mean() -> rank-0 -> unsqueeze lowers to a chain macOS 14's Espresso
        # runtime mangles into rank 2 ("Output rank has changed after
        # reshaping"), failing every prediction (FluidAudio#836).
        anchor = x_pre.mean(dim=(1, 2))
        return anchor, x_pre


class CoreMLTailModel(nn.Module):
    """FP32 tail: conv_post + exp + sin + iSTFT. Runs on ALL (GPU fp32 accum)."""
    def __init__(self, generator):
        super().__init__()
        self.post_n_fft = generator.post_n_fft
        self.conv_post = generator.conv_post
        self.stft = CoreMLCustomSTFT(
            CustomSTFT(filter_length=generator.post_n_fft,
                       hop_length=generator.stft.hop_length,
                       win_length=generator.post_n_fft))

    def forward(self, x):
        x = self.conv_post(x)
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)


def compute_shape_bounds(max_frames):
    """Derive all RangeDim bounds from max_frames (max acoustic frames T_a)."""
    return {
        'max_T_enc': 512,
        'max_T_a': max_frames,
        'max_T2': max_frames * 2,
        'max_ns0_T': max_frames * 20,
        'max_ns1_T': max_frames * 120 + 1,
        'max_x_pre_T': max_frames * 120 + 1,
        'max_audio_samples': max_frames * 600,
    }


class CoreMLCustomSTFT(nn.Module):
    """iSTFT using nn.ConvTranspose1d (traceable for CoreML)."""
    def __init__(self, original):
        super().__init__()
        self.center = original.center
        self.n_fft = original.n_fft
        self.hop_length = original.hop_length
        self.freq_bins = original.freq_bins
        self.deconv_real = nn.ConvTranspose1d(self.freq_bins, 1, self.n_fft,
                                              stride=self.hop_length, padding=0, bias=False)
        self.deconv_imag = nn.ConvTranspose1d(self.freq_bins, 1, self.n_fft,
                                              stride=self.hop_length, padding=0, bias=False)
        backward_real = original.weight_backward_real.clone()
        backward_imag = original.weight_backward_imag.clone()
        backward_real[1:-1] *= 2.0
        backward_imag[1:-1] *= 2.0
        self.deconv_real.weight = nn.Parameter(backward_real, requires_grad=False)
        self.deconv_imag.weight = nn.Parameter(backward_imag, requires_grad=False)

    def inverse(self, magnitude, phase):
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        waveform = self.deconv_real(real_part) - self.deconv_imag(imag_part)
        if self.center:
            pad_len = self.n_fft // 2
            waveform = waveform[..., pad_len:-pad_len]
        return waveform


class CoreMLProsodyF0N(nn.Module):
    """Prosody F0 + Noise prediction (runs after host-side alignment).
    Input en = d @ alignment, shape [1, 640, T_a]."""
    def __init__(self, predictor):
        super().__init__()
        self.shared = predictor.shared
        self.F0 = predictor.F0
        self.N = predictor.N
        self.F0_proj = predictor.F0_proj
        self.N_proj = predictor.N_proj

    def forward(self, en, s):
        x, _ = self.shared(en.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0).squeeze(1)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N).squeeze(1)
        return F0, N


class CoreMLAlbert(nn.Module):
    """Pure PyTorch ALBERT — no HuggingFace infrastructure.
    Reimplements forward pass using only traceable PyTorch ops."""
    def __init__(self, hf_albert):
        super().__init__()
        config = hf_albert.config
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.word_embeddings = hf_albert.embeddings.word_embeddings
        self.position_embeddings = hf_albert.embeddings.position_embeddings
        self.token_type_embeddings = hf_albert.embeddings.token_type_embeddings
        self.embedding_LayerNorm = hf_albert.embeddings.LayerNorm
        self.embedding_projection = hf_albert.encoder.embedding_hidden_mapping_in
        layer = hf_albert.encoder.albert_layer_groups[0].albert_layers[0]
        self.query = layer.attention.query
        self.key = layer.attention.key
        self.value = layer.attention.value
        self.attn_dense = layer.attention.dense
        self.attn_LayerNorm = layer.attention.LayerNorm
        self.ffn = layer.ffn
        self.ffn_output = layer.ffn_output
        self.full_layer_norm = layer.full_layer_layer_norm

    def forward(self, input_ids, attention_mask, token_type_ids):
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        embeddings = (self.word_embeddings(input_ids)
                      + self.position_embeddings(position_ids)
                      + self.token_type_embeddings(token_type_ids))
        embeddings = self.embedding_LayerNorm(embeddings)
        hidden = self.embedding_projection(embeddings)
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(hidden.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0
        for _ in range(self.num_hidden_layers):
            hidden = self._transformer_layer(hidden, extended_mask)
        return hidden

    def _transformer_layer(self, hidden, attention_mask):
        B, T, _ = hidden.shape
        Q = self.query(hidden).view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        K = self.key(hidden).view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        V = self.value(hidden).view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        attn_out = self.attn_dense(context)
        attn_out = self.attn_LayerNorm(attn_out + hidden)
        ffn_out = F.gelu(self.ffn(attn_out), approximate='tanh')
        ffn_out = self.ffn_output(ffn_out)
        hidden = self.full_layer_norm(ffn_out + attn_out)
        return hidden


class CoreMLTextEncoder(nn.Module):
    """TextEncoder without pack_padded_sequence."""
    def __init__(self, original):
        super().__init__()
        self.embedding = original.embedding
        self.cnn = original.cnn
        self.lstm = original.lstm

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m_unsq = m.unsqueeze(1)
        x = x.masked_fill(m_unsq, 0.0)
        for c in self.cnn:
            x = c(x)
            x = x.masked_fill(m_unsq, 0.0)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(-1, -2)
        x = x.masked_fill(m_unsq, 0.0)
        return x


class CoreMLDurationEncoder(nn.Module):
    """DurationEncoder with unrolled LSTM/AdaLayerNorm loop, no pack_padded_sequence."""
    def __init__(self, original):
        super().__init__()
        self.lstms = nn.ModuleList()
        self.norms = nn.ModuleList()
        for block in original.lstms:
            if isinstance(block, nn.LSTM):
                self.lstms.append(block)
            else:
                self.norms.append(block)
        self.dropout = original.dropout
        self.d_model = original.d_model
        self.sty_dim = original.sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        seq_len = x.shape[0]
        s = style.unsqueeze(0).expand(seq_len, -1, -1)
        x = torch.cat([x, s], dim=-1)
        x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for i in range(len(self.lstms)):
            x = x.transpose(-1, -2)
            x, _ = self.lstms[i](x)
            x = F.dropout(x, p=self.dropout, training=False)
            x = x.transpose(-1, -2)
            if i < len(self.norms):
                x = self.norms[i](x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
                x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
        return x.transpose(-1, -2)


class CoreMLAlbertStandalone(nn.Module):
    """Standalone ALBERT model.
    Input: input_ids [1, T], attention_mask [1, T]
    Output: bert_dur [1, T, 768]"""
    def __init__(self, kmodel):
        super().__init__()
        self.bert = CoreMLAlbert(kmodel.bert)

    def forward(self, input_ids, attention_mask):
        token_type_ids = torch.zeros_like(input_ids)
        return self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


class CoreMLPostAlbert(nn.Module):
    """Post-ALBERT encoder: bert_dur → duration, d, t_en.
    Input: bert_dur [1, T, 768], input_ids [1, T], style_s [1, 128], speed [1], attention_mask [1, T]
    Output: duration [1, T], d [1, T, 640], t_en [1, 512, T]"""
    def __init__(self, kmodel):
        super().__init__()
        self.bert_encoder = kmodel.bert_encoder
        self.dur_encoder = CoreMLDurationEncoder(kmodel.predictor.text_encoder)
        self.dur_lstm = kmodel.predictor.lstm
        self.duration_proj = kmodel.predictor.duration_proj
        self.text_encoder = CoreMLTextEncoder(kmodel.text_encoder)

    def forward(self, bert_dur, input_ids, style_s, speed, attention_mask):
        input_lengths = attention_mask.sum(dim=-1).to(torch.long)
        text_mask = (attention_mask == 0)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        d = self.dur_encoder(d_en, style_s, input_lengths, text_mask)
        x, _ = self.dur_lstm(d)
        dur_raw = self.duration_proj(x)
        duration = torch.sigmoid(dur_raw).sum(dim=-1) / speed
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        return duration, d + 0, t_en + 0


class CoreMLAlignmentStandalone(nn.Module):
    """Standalone alignment: pred_dur + d + t_en → en, asr (dynamic T_a output).
    Uses cumsum + broadcasting comparison — no repeat_interleave."""
    def __init__(self, max_frames=600):
        super().__init__()
        self.max_frames = max_frames

    def forward(self, pred_dur, d, t_en):
        d = d.float()
        t_en = t_en.float()
        dur = pred_dur.float()
        cum_dur = torch.cumsum(dur, dim=-1)
        starts = cum_dur - dur
        frames = torch.arange(self.max_frames, device=d.device).float().unsqueeze(0).unsqueeze(0)
        alignment = ((frames >= starts.unsqueeze(-1)) & (frames < cum_dur.unsqueeze(-1))).float()
        en = d.transpose(-1, -2) @ alignment
        asr = t_en @ alignment
        T_a = cum_dur[:, -1:].to(torch.int32).squeeze()
        en = en[:, :, :T_a]
        asr = asr[:, :, :T_a]
        return en, asr


class CoreMLSineGenV2(nn.Module):
    """SineGen with avg_pool1d downsample + interpolate upsample for CoreML."""
    def __init__(self, original):
        super().__init__()
        self.sine_amp = original.sine_amp
        self.noise_std = original.noise_std
        self.harmonic_num = original.harmonic_num
        self.dim = original.dim
        self.sampling_rate = original.sampling_rate
        self.voiced_threshold = original.voiced_threshold
        self.upsample_scale = original.upsample_scale

    def forward(self, f0):
        harmonics = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype)
        fn = f0 * harmonics.view(1, 1, -1)
        rad_values = fn / self.sampling_rate
        rv = rad_values.transpose(1, 2)
        rv_down = F.avg_pool1d(rv, kernel_size=self.upsample_scale, stride=self.upsample_scale)
        rad_values_down = rv_down.transpose(1, 2)
        phase = torch.cumsum(rad_values_down, dim=1) * (2 * math.pi)
        ph = phase.transpose(1, 2) * self.upsample_scale
        ph_up = F.interpolate(ph, scale_factor=float(self.upsample_scale), mode="linear", align_corners=False)
        phase = ph_up.transpose(1, 2)
        sines = torch.sin(phase) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * 0.01  # Deterministic noise for tracing
        sine_waves = sines * uv + noise
        return sine_waves, uv, noise


class CoreMLSourceModuleV2(nn.Module):
    """SourceModule using SineGenV2 for CoreML tracing."""
    def __init__(self, original):
        super().__init__()
        self.sine_amp = original.sine_amp
        self.l_sin_gen = CoreMLSineGenV2(original.l_sin_gen)
        self.l_linear = original.l_linear
        self.l_tanh = original.l_tanh

    def forward(self, x):
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.zeros_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class CoreMLForwardSTFT(nn.Module):
    """Forward STFT using nn.Conv1d (requires CustomSTFT from disable_complex=True)."""
    def __init__(self, original_stft):
        super().__init__()
        self.center = original_stft.center
        self.n_fft = original_stft.n_fft
        self.hop_length = original_stft.hop_length
        self.freq_bins = original_stft.freq_bins
        self.pad_mode = original_stft.pad_mode
        self.conv_real = nn.Conv1d(1, self.freq_bins, self.n_fft,
                                    stride=self.hop_length, padding=0, bias=False)
        self.conv_imag = nn.Conv1d(1, self.freq_bins, self.n_fft,
                                    stride=self.hop_length, padding=0, bias=False)
        self.conv_real.weight = nn.Parameter(original_stft.weight_forward_real, requires_grad=False)
        self.conv_imag.weight = nn.Parameter(original_stft.weight_forward_imag, requires_grad=False)

    def transform(self, waveform):
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
        x = waveform.unsqueeze(1)
        real_out = self.conv_real(x)
        imag_out = self.conv_imag(x)
        # `torch.stft` on a real signal forces DC and Nyquist bins to purely
        # real (imag exactly 0) and `torch.angle` returns +π for real<0.
        # Our conv1d-based STFT computes imag from `-sin(πn) * window`, which
        # is mathematically 0 at Nyquist but evaluates to ~1e-15 in fp32.
        # `atan2(tiny-negative, negative)` returns ~-π, producing a 2π phase
        # error that propagates as broad-spectrum noise through noise_convs.
        #
        # Two-stage fix:
        #   1. Clip imag values below 1e-7 to exactly zero. Far above the
        #      ~1e-15 floor that DC/Nyquist conv produces, far below any
        #      legitimate spectral imag value (≥ 1e-3 for typical audio).
        #   2. After atan2, force the (imag==0, real<0) branch to +π — the
        #      same correction kokoro/custom_stft.py:135-138 documents for
        #      the ONNX path. PyTorch's atan2 returns +π in this case;
        #      CoreML's returns 0.
        eps = 1e-5
        imag_clipped = torch.where(imag_out.abs() < eps,
                                   torch.zeros_like(imag_out), imag_out)
        magnitude = torch.sqrt(real_out ** 2 + imag_clipped ** 2 + 1e-14)
        phase = torch.atan2(imag_clipped, real_out)
        correction_mask = (imag_clipped == 0) & (real_out < 0)
        phase = torch.where(correction_mask,
                            torch.full_like(phase, math.pi), phase)
        return magnitude, phase


class CoreMLFullNoiseModel(nn.Module):
    """Full noise pipeline: F0 + style → x_sources.
    Includes SineGen + STFT + noise_convs + noise_res."""
    def __init__(self, generator):
        super().__init__()
        self.f0_upsamp = generator.f0_upsamp
        self.m_source = CoreMLSourceModuleV2(generator.m_source)
        fwd_stft = CustomSTFT(
            filter_length=generator.stft.filter_length,
            hop_length=generator.stft.hop_length,
            win_length=generator.stft.win_length,
        )
        self.stft = CoreMLForwardSTFT(fwd_stft)
        self.noise_convs = generator.noise_convs
        self.noise_res = generator.noise_res

    def forward(self, F0_curve, style_timbre):
        f0 = self.f0_upsamp(F0_curve[:, None]).transpose(1, 2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        x_source_0 = self.noise_convs[0](har)
        x_source_0 = self.noise_res[0](x_source_0, style_timbre)
        x_source_1 = self.noise_convs[1](har)
        x_source_1 = self.noise_res[1](x_source_1, style_timbre)
        return x_source_0, x_source_1


def precompute_noise_sources(generator, F0_curve, style_timbre):
    """Compute noise sources on host (SineGen + noise_convs + noise_res in fp32)."""
    with torch.no_grad():
        f0_up = generator.f0_upsamp(F0_curve[:, None]).transpose(1, 2)
        har_source, _, _ = generator.m_source(f0_up)
        har_source_flat = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator.stft.transform(har_source_flat)
        har = torch.cat([har_spec, har_phase], dim=1)

        sources = []
        for i in range(generator.num_upsamples):
            x_source = generator.noise_convs[i](har)
            x_source = generator.noise_res[i](x_source, style_timbre)
            sources.append(x_source)
        return sources


def _remove_weight_norm(model):
    """Remove weight_norm parametrizations for tracing."""
    for module in model.modules():
        if hasattr(module, 'parametrizations'):
            for name in list(module.parametrizations.keys()):
                try:
                    torch.nn.utils.parametrize.remove_parametrizations(module, name, leave_parametrized=True)
                except Exception:
                    pass


def mel_corr(a, b, sr=24000, n_fft=1024, n_mels=80, hop=256):
    from scipy.signal import stft as scipy_stft
    fmin, fmax = 0.0, sr / 2.0
    mel_lo = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_hi = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_lo, mel_hi, n_mels + 2)
    freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lo, mid, hi = freqs[i], freqs[i + 1], freqs[i + 2]
        up = (fft_freqs - lo) / max(mid - lo, 1e-10)
        down = (hi - fft_freqs) / max(hi - mid, 1e-10)
        fb[i] = np.maximum(0, np.minimum(up, down))
    def mel_spec(x):
        _, _, Zxx = scipy_stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        S = np.abs(Zxx) ** 2
        return np.log1p(fb @ S)
    A, B = mel_spec(a), mel_spec(b)
    return np.corrcoef(A.flatten(), B.flatten())[0, 1]


STAGE_NAMES = ['albert', 'post_albert', 'alignment', 'prosody', 'noise', 'vocoder', 'tail']


def main():
    parser = argparse.ArgumentParser(description='Convert Kokoro TTS to CoreML (fp16+int8pal preset)')
    parser.add_argument('--output-dir', type=pathlib.Path, required=True,
                        help='Directory to write the 7 mlpackages and ref/test WAVs into')
    parser.add_argument('--max-frames', type=int, default=2000, help='Max T_a frames for RangeDim (also static buffer cap in Alignment trace)')
    parser.add_argument('--stages', nargs='+', choices=STAGE_NAMES + ['all'], default=['all'],
                        help=f'Which stages to convert. Choices: all | {" ".join(STAGE_NAMES)}. '
                             f'Skipped stages reuse existing mlpackages on disk for the E2E chain.')
    parser.add_argument('--no-palettize', nargs='*', choices=STAGE_NAMES + ['all'], default=[],
                        help='Stages to convert WITHOUT int8 kmeans palettization '
                             '(emit pure fp16/fp32 weights). Useful for isolating quantization '
                             'noise. Pass "all" to disable palettization for every selected stage.')
    args = parser.parse_args()
    selected = set(STAGE_NAMES) if 'all' in args.stages else set(args.stages)
    no_pal = set(STAGE_NAMES) if 'all' in args.no_palettize else set(args.no_palettize)

    global OUTDIR
    OUTDIR = args.output_dir
    OUTDIR.mkdir(parents=True, exist_ok=True)

    import time as _time

    # ── Load model (v1.1-zh checkpoint) ──
    print('[1] Loading KModel (hexgrad/Kokoro-82M-v1.1-zh)...')
    model = KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh')
    model.eval()
    # Sanity-check: v1.1-zh vocab must contain Bopomofo (ㄅ-ㄩ, U+3105-U+312F).
    # Actual vocab size is 171 entries (IPA + Bopomofo + tone digits + a few
    # Hanzi/punctuation), distinct from v1.0's 177-entry English IPA vocab.
    bopomofo_keys = [k for k in model.vocab if any('\u3105' <= c <= '\u312f' for c in k)]
    assert bopomofo_keys, (
        f'Expected v1.1-zh vocab to contain Bopomofo (ㄅㄆㄇ…), but found none. '
        f'Got {len(model.vocab)} entries: {list(model.vocab.keys())[:20]}. '
        f'Check that `kokoro` package is recent enough to recognize '
        f'Kokoro-82M-v1.1-zh config.json.'
    )
    print(f'    vocab: {len(model.vocab)} entries, {len(bopomofo_keys)} Bopomofo')

    # ── Generate Mandarin test inputs via misaki[zh] G2P ──
    # KPipeline with lang_code='z' uses misaki.zh → pypinyin → Bopomofo+digit
    # tone phonemes; the resulting `ps` string is what we feed to the trace.
    print('[2] Generating Mandarin test inputs...')
    from kokoro.pipeline import KPipeline
    pipe = KPipeline(lang_code='z', repo_id='hexgrad/Kokoro-82M-v1.1-zh', model=model)
    trace_voice = 'zf_001'
    voice_pack = pipe.load_voice(trace_voice)

    text_zh = '你好世界，今天天气很好。'
    phonemes = None
    for _gs, ps, _tks in pipe(text_zh, voice=trace_voice):
        phonemes = ps
        break
    if not phonemes:
        raise RuntimeError(f'misaki[zh] G2P produced empty phonemes for {text_zh!r}')
    print(f'    text={text_zh!r}')
    print(f'    phonemes={phonemes!r} (len={len(phonemes)})')

    input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    ref_s = voice_pack[len(phonemes) - 1]

    with torch.no_grad():
        input_lengths = torch.LongTensor([input_ids.shape[1]])
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        style_timbre = ref_s[:, :128]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        T_a = pred_dur.sum().item()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln = torch.zeros(input_ids.shape[1], T_a)
        pred_aln[indices, torch.arange(T_a)] = 1.0
        pred_aln = pred_aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln

    print(f'    T_a={T_a}')

    # Teacher reference
    with torch.no_grad():
        ref_audio = model.decoder(asr, F0_pred, N_pred, style_timbre)
    ref_flat = ref_audio.squeeze().numpy()
    real_audio_len = len(ref_flat)

    max_frames = args.max_frames

    # Precompute noise sources before weight_norm removal
    generator = model.decoder.generator
    noise_sources = precompute_noise_sources(generator, F0_pred, style_timbre)

    # Remove weight_norm and dropout on all submodules
    _remove_weight_norm(model.decoder)
    _remove_weight_norm(model.predictor)
    _remove_weight_norm(model.text_encoder)
    for parent_mod in [model.predictor, model.text_encoder]:
        for m_name, m_mod in list(parent_mod.named_modules()):
            if isinstance(m_mod, nn.Dropout):
                parts = m_name.rsplit('.', 1)
                parent = parent_mod.get_submodule(parts[0]) if len(parts) > 1 else parent_mod
                setattr(parent, parts[-1], nn.Identity())

    attention_mask = (~text_mask).int()
    speed = torch.ones(1)

    # Derive all shape bounds from max_frames
    bounds = compute_shape_bounds(max_frames)
    T_enc = ct.RangeDim(lower_bound=2, upper_bound=bounds['max_T_enc'], default=input_ids.shape[1])
    T_a_dim = ct.RangeDim(lower_bound=1, upper_bound=bounds['max_T_a'], default=T_a)
    T2_dim = ct.RangeDim(lower_bound=2, upper_bound=bounds['max_T2'], default=F0_pred.shape[1])
    T_ns0 = ct.RangeDim(lower_bound=1, upper_bound=bounds['max_ns0_T'], default=noise_sources[0].shape[2])
    T_ns1 = ct.RangeDim(lower_bound=1, upper_bound=bounds['max_ns1_T'], default=noise_sources[1].shape[2])
    T_pre = ct.RangeDim(lower_bound=100, upper_bound=bounds['max_x_pre_T'], default=T_a * 120 + 1)
    print(f'    Shape bounds: T_a≤{bounds["max_T_a"]}, ns0_T≤{bounds["max_ns0_T"]}, '
          f'ns1_T≤{bounds["max_ns1_T"]}, x_pre_T≤{bounds["max_x_pre_T"]}')

    CU_LIST = [('CPU_ONLY', ct.ComputeUnit.CPU_ONLY),
                ('CPU_AND_NE', ct.ComputeUnit.CPU_AND_NE),
                ('CPU_AND_GPU', ct.ComputeUnit.CPU_AND_GPU),
                ('ALL', ct.ComputeUnit.ALL)]

    def bench(path, feed, n_runs=10):
        result = None
        for cu_name, cu in CU_LIST:
            try:
                ml = ct.models.MLModel(str(path), compute_units=cu)
                for _ in range(3): ml.predict(feed)
                t0 = _time.perf_counter()
                for _ in range(n_runs): out = ml.predict(feed)
                ms = (_time.perf_counter() - t0) / n_runs * 1000
                print(f'    {cu_name:15s}  {ms:5.1f}ms')
                result = out
            except Exception as e:
                print(f'    {cu_name:15s}  FAILED: {str(e)[:60]}')
        return result

    pal_config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=8))

    # ═══ 1/7: ALBERT (fp16+int8pal) ═══
    albert_path = OUTDIR / 'KokoroAlbert.mlpackage'
    albert_feed = {"input_ids": input_ids.numpy().astype(np.int32),
                   "attention_mask": attention_mask.numpy().astype(np.int32)}
    if 'albert' in selected:
        print('\n[1/7] ALBERT (fp32compute+fp16io)...')
        albert = CoreMLAlbertStandalone(model)
        albert.eval()
        with torch.no_grad():
            traced = torch.jit.trace(albert, (input_ids, attention_mask), strict=False)
        ml = ct.convert(traced,
            inputs=[ct.TensorType(name="input_ids", shape=(1, T_enc), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, T_enc), dtype=np.int32)],
            outputs=[ct.TensorType(name="bert_dur")],
            convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.ALL)
        if 'albert' not in no_pal:
            ml = cto.palettize_weights(ml, pal_config)
        else:
            print('    SKIP palettize (--no-palettize albert)')
        ml.save(str(albert_path))
        bench(albert_path, albert_feed)
    else:
        print(f'\n[1/7] ALBERT — SKIP (reusing {albert_path.name})')

    # ═══ 2/7: PostAlbert (fp16+int8pal) ═══
    post_path = OUTDIR / 'KokoroPostAlbert.mlpackage'
    post_feed = {"bert_dur": bert_dur.numpy().astype(np.float16),
                 "input_ids": input_ids.numpy().astype(np.int32),
                 "style_s": s.numpy().astype(np.float16),
                 "speed": np.array([1.0], dtype=np.float16),
                 "attention_mask": attention_mask.numpy().astype(np.int32)}
    if 'post_albert' in selected:
        print('\n[2/7] PostAlbert (fp32 pure)...')
        post = CoreMLPostAlbert(model)
        post.eval()
        with torch.no_grad():
            traced = torch.jit.trace(post, (bert_dur, input_ids, s, speed, attention_mask), strict=False)
        ml = ct.convert(traced,
            inputs=[ct.TensorType(name="bert_dur", shape=(1, T_enc, 768), dtype=np.float32),
                    ct.TensorType(name="input_ids", shape=(1, T_enc), dtype=np.int32),
                    ct.TensorType(name="style_s", shape=(1, 128), dtype=np.float32),
                    ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
                    ct.TensorType(name="attention_mask", shape=(1, T_enc), dtype=np.int32)],
            outputs=[ct.TensorType(name="duration"), ct.TensorType(name="d"), ct.TensorType(name="t_en")],
            convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.ALL)
        if 'post_albert' not in no_pal:
            ml = cto.palettize_weights(ml, pal_config)
        else:
            print('    SKIP palettize (--no-palettize post_albert)')
        ml.save(str(post_path))
        bench(post_path, post_feed)
    else:
        print(f'\n[2/7] PostAlbert — SKIP (reusing {post_path.name})')

    # ═══ 3/7: Alignment (fp16+int8pal) ═══
    align_path = OUTDIR / 'KokoroAlignment.mlpackage'
    pred_dur_int = torch.round(duration).clamp(min=1).to(torch.int32)
    align_feed = {"pred_dur": pred_dur_int.numpy().astype(np.int32),
                  "d": d.numpy().astype(np.float16),
                  "t_en": t_en.numpy().astype(np.float16)}
    if 'alignment' in selected:
        print('\n[3/7] Alignment (fp32 pure)...')
        align = CoreMLAlignmentStandalone(max_frames=max_frames)
        align.eval()
        with torch.no_grad():
            traced = torch.jit.trace(align, (pred_dur_int, d, t_en), strict=False)
        ml = ct.convert(traced,
            inputs=[ct.TensorType(name="pred_dur", shape=(1, T_enc), dtype=np.int32),
                    ct.TensorType(name="d", shape=(1, T_enc, 640), dtype=np.float32),
                    ct.TensorType(name="t_en", shape=(1, 512, T_enc), dtype=np.float32)],
            outputs=[ct.TensorType(name="en"), ct.TensorType(name="asr")],
            convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.ALL)
        if 'alignment' not in no_pal:
            ml = cto.palettize_weights(ml, pal_config)
        else:
            print('    SKIP palettize (--no-palettize alignment)')
        ml.save(str(align_path))
        bench(align_path, align_feed)
    else:
        print(f'\n[3/7] Alignment — SKIP (reusing {align_path.name})')

    # ═══ 4/7: Prosody (fp16+int8pal) ═══
    pros_path = OUTDIR / 'KokoroProsody.mlpackage'
    pros_feed = {"en": en.numpy().astype(np.float16),
                 "style_s": s.numpy().astype(np.float16)}
    if 'prosody' in selected:
        print('\n[4/7] Prosody (fp32 pure)...')
        prosody = CoreMLProsodyF0N(model.predictor)
        prosody.eval()
        with torch.no_grad():
            traced = torch.jit.trace(prosody, (en, s), strict=False)
        ml = ct.convert(traced,
            inputs=[ct.TensorType(name="en", shape=(1, 640, T_a_dim), dtype=np.float32),
                    ct.TensorType(name="style_s", shape=(1, 128), dtype=np.float32)],
            outputs=[ct.TensorType(name="F0"), ct.TensorType(name="N")],
            convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.ALL)
        if 'prosody' not in no_pal:
            ml = cto.palettize_weights(ml, pal_config)
        else:
            print('    SKIP palettize (--no-palettize prosody)')
        ml.save(str(pros_path))
        bench(pros_path, pros_feed)
    else:
        print(f'\n[4/7] Prosody — SKIP (reusing {pros_path.name})')

    # ═══ 5/7: Noise (fp32+int8pal — sin/cumsum precision-sensitive) ═══
    noise_path = OUTDIR / 'KokoroNoise.mlpackage'
    noise_feed = {"F0_curve": F0_pred.numpy().astype(np.float32),
                  "style_timbre": style_timbre.numpy().astype(np.float32)}
    if 'noise' in selected:
        print('\n[5/7] Noise (fp32+int8pal)...')
        noise_model = CoreMLFullNoiseModel(generator)
        noise_model.eval()
        with torch.no_grad():
            traced = torch.jit.trace(noise_model, (F0_pred, style_timbre), strict=False)
        ml = ct.convert(traced,
            inputs=[ct.TensorType(name="F0_curve", shape=(1, T2_dim), dtype=np.float32),
                    ct.TensorType(name="style_timbre", shape=(1, 128), dtype=np.float32)],
            outputs=[ct.TensorType(name="x_source_0"), ct.TensorType(name="x_source_1")],
            convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.ALL)
        if 'noise' not in no_pal:
            ml = cto.palettize_weights(ml, pal_config)
        else:
            print('    SKIP palettize (--no-palettize noise) — pure fp32 weights')
        ml.save(str(noise_path))
        bench(noise_path, noise_feed)
    else:
        print(f'\n[5/7] Noise — SKIP (reusing {noise_path.name})')

    # ═══ 6/7: Vocoder (cos fp16+int8pal, dual output with minimal anchor) ═══
    voc_path = OUTDIR / 'KokoroVocoder.mlpackage'
    voc_feed = {"asr": asr.numpy().astype(np.float16),
                "F0_curve": F0_pred.numpy().astype(np.float16),
                "N_pred": N_pred.numpy().astype(np.float16),
                "x_source_0": noise_sources[0].numpy().astype(np.float16),
                "x_source_1": noise_sources[1].numpy().astype(np.float16),
                "style_timbre": style_timbre.numpy().astype(np.float16)}
    if 'vocoder' in selected:
        print('\n[6/7] Vocoder (cos fp32 pure, dual output)...')
        AdaINResBlock1.forward = _cos_resblock1_forward
        vocoder = CoreMLVocoderDualOutput(model.decoder)
        vocoder.eval()
        with torch.no_grad():
            traced = torch.jit.trace(vocoder, (asr, F0_pred, N_pred,
                     noise_sources[0], noise_sources[1], style_timbre), strict=False)
        ns0_C = noise_sources[0].shape[1]
        ns1_C = noise_sources[1].shape[1]
        ml = ct.convert(traced,
            inputs=[ct.TensorType(name="asr", shape=(1, 512, T_a_dim), dtype=np.float32),
                    ct.TensorType(name="F0_curve", shape=(1, T2_dim), dtype=np.float32),
                    ct.TensorType(name="N_pred", shape=(1, T2_dim), dtype=np.float32),
                    ct.TensorType(name="x_source_0", shape=(1, ns0_C, T_ns0), dtype=np.float32),
                    ct.TensorType(name="x_source_1", shape=(1, ns1_C, T_ns1), dtype=np.float32),
                    ct.TensorType(name="style_timbre", shape=(1, 128), dtype=np.float32)],
            outputs=[ct.TensorType(name="anchor"), ct.TensorType(name="x_pre")],
            convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.ALL)
        if 'vocoder' not in no_pal:
            print('    Palettizing...')
            ml = cto.palettize_weights(ml, pal_config)
        else:
            print('    SKIP palettize (--no-palettize vocoder)')
        ml.save(str(voc_path))
        bench(voc_path, voc_feed)
    else:
        print(f'\n[6/7] Vocoder — SKIP (reusing {voc_path.name})')

    # ═══ 7/7: Tail (fp32 conv_post + exp + sin + iSTFT) ═══
    tail_path = OUTDIR / 'KokoroTail.mlpackage'
    if 'tail' in selected:
        print('\n[7/7] Tail (fp32)...')
        tail_model = CoreMLTailModel(generator)
        tail_model.eval()
        with torch.no_grad():
            x_pre_dummy = torch.randn(1, 128, 1000)
            traced_tail = torch.jit.trace(tail_model, x_pre_dummy, strict=False)
        ml_tail = ct.convert(traced_tail,
            inputs=[ct.TensorType(name="x_pre", shape=(1, 128, T_pre), dtype=np.float32)],
            outputs=[ct.TensorType(name="audio")],
            convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.ALL)
        ml_tail.save(str(tail_path))
        voc_out = ct.models.MLModel(str(voc_path), compute_units=ct.ComputeUnit.CPU_AND_NE).predict(voc_feed)
        tail_feed = {"x_pre": np.array(voc_out["x_pre"]).astype(np.float32)}
        bench(tail_path, tail_feed)
    else:
        print(f'\n[7/7] Tail — SKIP (reusing {tail_path.name})')

    # ═══ E2E chained validation ═══
    print('\n[E2E] Chained validation (7 models, dual output + fp32 tail)...')
    m_albert = ct.models.MLModel(str(albert_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    m_post = ct.models.MLModel(str(post_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    m_align = ct.models.MLModel(str(align_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    m_pros = ct.models.MLModel(str(pros_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    m_noise = ct.models.MLModel(str(noise_path), compute_units=ct.ComputeUnit.ALL)
    m_voc = ct.models.MLModel(str(voc_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    m_tail = ct.models.MLModel(str(tail_path), compute_units=ct.ComputeUnit.ALL)

    def run_chain():
        o1 = m_albert.predict(albert_feed)
        o2 = m_post.predict({
            "bert_dur": np.array(o1["bert_dur"]).astype(np.float16),
            "input_ids": input_ids.numpy().astype(np.int32),
            "style_s": s.numpy().astype(np.float16),
            "speed": np.array([1.0], dtype=np.float16),
            "attention_mask": attention_mask.numpy().astype(np.int32)})
        dur = np.array(o2["duration"]).flatten()
        pd = np.round(dur).clip(min=1).astype(np.int32).reshape(1, -1)
        o3 = m_align.predict({
            "pred_dur": pd,
            "d": np.array(o2["d"]).astype(np.float16),
            "t_en": np.array(o2["t_en"]).astype(np.float16)})
        o4 = m_pros.predict({
            "en": np.array(o3["en"]).astype(np.float16),
            "style_s": s.numpy().astype(np.float16)})
        o5 = m_noise.predict({
            "F0_curve": np.array(o4["F0"]).astype(np.float32),
            "style_timbre": style_timbre.numpy().astype(np.float32)})
        o6 = m_voc.predict({
            "asr": np.array(o3["asr"]).astype(np.float16),
            "F0_curve": np.array(o4["F0"]).astype(np.float16),
            "N_pred": np.array(o4["N"]).astype(np.float16),
            "x_source_0": np.array(o5["x_source_0"]).astype(np.float16),
            "x_source_1": np.array(o5["x_source_1"]).astype(np.float16),
            "style_timbre": style_timbre.numpy().astype(np.float16)})
        return m_tail.predict({
            "x_pre": np.array(o6["x_pre"]).astype(np.float32)})

    for _ in range(2): run_chain()  # warmup
    o_final = run_chain()
    e2e_flat = o_final["audio"].flatten()[:real_audio_len]
    corr_e2e = np.corrcoef(ref_flat, e2e_flat)[0, 1]
    mc_e2e = mel_corr(e2e_flat, ref_flat)

    t0 = _time.perf_counter()
    for _ in range(5): run_chain()
    e2e_ms = (_time.perf_counter() - t0) / 5 * 1000
    print(f'    corr={corr_e2e:.6f}, mel_corr={mc_e2e:.6f}, chain={e2e_ms:.1f}ms')

    try:
        import soundfile as sf
        sf.write(str(OUTDIR / 'test.wav'), e2e_flat, 24000)
        sf.write(str(OUTDIR / 'ref.wav'), ref_flat, 24000)
    except ImportError:
        pass

    print(f'\n{"="*60}')
    print(f'E2E models saved to {OUTDIR}/ (7 models: cos+int8pal vocoder + fp32 tail):')
    for f in sorted(OUTDIR.glob('*.mlpackage')):
        print(f'  {f.name}')
    print(f'E2E chain: corr={corr_e2e:.6f}, mel_corr={mc_e2e:.6f}, {e2e_ms:.1f}ms')


if __name__ == '__main__':
    main()
