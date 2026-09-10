"""CoreML-traceable wrappers for Chatterbox S3Gen (flow + HiFT vocoder).

``FlowCoreML`` replicates CausalMaskedDiffWithXvec.inference (finalize=True)
with a static token bucket N (prompt + generated, densely packed then
right-padded) producing M = 2N mel frames, running the full 10-step CFG Euler
loop in-graph. Noise ``z`` is a host input so parity is deterministic.

``HiFTCoreML`` replicates HiFTGenerator.inference with:
  * matmul STFT/iSTFT (torch.stft unsupported by coremltools)
  * SineGen's random phase + additive noise as host inputs
  * weight_norm folded to plain tensors

Both bind the loaded upstream modules directly — no weight copies.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stft_coreml import ISTFT, STFT
from src.weight_norm_fold import fold_weight_norm


class FlowCoreML(nn.Module):
    def __init__(self, flow: nn.Module, n_total_tokens: int, n_timesteps: int = 10):
        super().__init__()
        self.flow = flow
        self.N = n_total_tokens
        self.ratio = flow.token_mel_ratio                 # 2
        self.M = n_total_tokens * self.ratio
        self.n_timesteps = n_timesteps
        self.cfg_rate = float(flow.decoder.inference_cfg_rate)  # 0.7

        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
        if flow.decoder.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        self.register_buffer("t_span", t_span, persistent=False)
        self.register_buffer("arange_n", torch.arange(n_total_tokens, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("arange_m", torch.arange(self.M, dtype=torch.float32),
                             persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,        # (1, N) int32 — prompt ++ generated, right-padded
        token_len: torch.Tensor,     # (1,) int32 — number of valid tokens
        prompt_len: torch.Tensor,    # (1,) int32 — prompt token count
        prompt_feat: torch.Tensor,   # (1, M, 80) — prompt mel, zero-padded past 2*prompt_len
        embedding: torch.Tensor,     # (1, 192) — CAMPPlus x-vector
        z: torch.Tensor,             # (1, 80, M) — CFM initial noise
    ) -> torch.Tensor:
        """Returns mel (1, 80, M); valid frames are [2*prompt_len, 2*token_len)."""
        flow = self.flow
        tl = token_len.to(torch.float32).view(1)

        emb = F.normalize(embedding, dim=1)
        emb = flow.spk_embed_affine_layer(emb)            # (1, 80)

        tok_mask = (self.arange_n < tl).view(1, -1, 1).to(embedding.dtype)
        tok = flow.input_embedding(torch.clamp(tokens.to(torch.int64), min=0)) * tok_mask

        # Replicate UpsampleConformerEncoder.forward with one fix: re-zero the
        # padded positions after each embed stage. embed's Linear+LayerNorm
        # maps zero rows to a nonzero bias vector; the right-looking
        # pre_lookahead conv then reads that junk at the tail of the valid
        # region, which is why a padded bucket diverged from the exact-size
        # run (bit-exact once re-zeroed — see convert-s3gen.py parity).
        enc = flow.encoder
        masks_bool = (self.arange_n < tl).view(1, 1, -1) > 0          # (1, 1, N)
        xs, pos_emb, masks_e = enc.embed(tok, masks_bool)
        xs = xs * tok_mask
        xs = enc.pre_lookahead_layer(xs)
        xs = enc.forward_layers(xs, masks_e, pos_emb, masks_e)

        xs = xs.transpose(1, 2)
        xs, _ = enc.up_layer(xs, token_len.to(torch.int64).view(1))
        xs = xs.transpose(1, 2)
        mel_mask_col = (self.arange_m < 2.0 * tl).view(1, -1, 1).to(embedding.dtype)
        masks_up = (self.arange_m < 2.0 * tl).view(1, 1, -1) > 0      # (1, 1, M)
        xs, pos_emb_up, masks_u = enc.up_embed(xs, masks_up)
        xs = xs * mel_mask_col
        xs = enc.forward_up_layers(xs, masks_u, pos_emb_up, masks_u)
        h = enc.after_norm(xs)                            # (1, M, 512)
        h = flow.encoder_proj(h)                          # (1, M, 80)

        # conds: prompt mel prefix, zeros elsewhere (prompt_feat pre-padded by host)
        cond = prompt_feat.transpose(1, 2)                # (1, 80, M)
        mu = h.transpose(1, 2)                            # (1, 80, M)

        mel_mask = (self.arange_m < 2.0 * tl).view(1, 1, -1).to(embedding.dtype)

        x = z
        zero_mu = torch.zeros_like(mu)
        zero_spk = torch.zeros_like(emb)
        zero_cond = torch.zeros_like(cond)
        mask_in = torch.cat([mel_mask, mel_mask], dim=0)  # (2, 1, M)

        for step in range(self.n_timesteps):
            t = self.t_span[step]
            dt = self.t_span[step + 1] - t
            x_in = torch.cat([x, x], dim=0)
            mu_in = torch.cat([mu, zero_mu], dim=0)
            spk_in = torch.cat([emb, zero_spk], dim=0)
            cond_in = torch.cat([cond, zero_cond], dim=0)
            t_in = t.view(1).repeat(2)
            dxdt = flow.decoder.estimator(x_in, mask_in, mu_in, t_in, spk_in, cond_in)
            d_cond, d_uncond = torch.split(dxdt, [1, 1], dim=0)
            dxdt = (1.0 + self.cfg_rate) * d_cond - self.cfg_rate * d_uncond
            x = x + dt * dxdt

        return x


class SineGenCoreML(nn.Module):
    """Deterministic SineGen: random phase + noise become inputs."""

    def __init__(self, sine_gen: nn.Module):
        super().__init__()
        self.sine_amp = sine_gen.sine_amp
        self.noise_std = sine_gen.noise_std
        self.harmonic_num = sine_gen.harmonic_num
        self.voiced_threshold = sine_gen.voiced_threshold
        harmonics = torch.arange(1, self.harmonic_num + 2, dtype=torch.float32)
        self.register_buffer("harm_over_sr", (harmonics / sine_gen.sampling_rate).view(1, -1, 1),
                             persistent=False)

    def forward(self, f0, phase_vec, noise):
        # f0: (1, 1, L); phase_vec: (1, H+1, 1) (row 0 must be 0); noise: (1, H+1, L)
        F_mat = f0 * self.harm_over_sr                      # (1, H+1, L)
        theta = 2.0 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1.0)
        sine_waves = self.sine_amp * torch.sin(theta + phase_vec)
        uv = (f0 > self.voiced_threshold).to(f0.dtype)      # (1, 1, L)
        noise_amp = uv * self.noise_std + (1.0 - uv) * self.sine_amp / 3.0
        return sine_waves * uv + noise_amp * noise          # (1, H+1, L)


class HiFTCoreML(nn.Module):
    def __init__(self, gen: nn.Module):
        super().__init__()
        fold_weight_norm(gen)
        self.gen = gen
        self.sine_gen = SineGenCoreML(gen.m_source.l_sin_gen)

        n_fft = gen.istft_params["n_fft"]
        hop = gen.istft_params["hop_len"]
        self.stft = STFT(n_fft, hop, window="hann")
        self.istft = ISTFT(n_fft, hop, window="hann")
        self.n_fft = n_fft
        self.upsample_total = int(gen.f0_upsamp.scale_factor)  # 480 samples/mel frame

    def forward(self, mel: torch.Tensor, phase_vec: torch.Tensor, noise: torch.Tensor):
        """mel: (1, 80, T) → audio (1, T*480).

        phase_vec (1, 9, 1) and noise (1, 9, T*480) reproduce SineGen's
        randomness host-side (phase_vec[0,0,0] must be 0).
        """
        gen = self.gen
        f0 = gen.f0_predictor(mel)                         # (1, T)
        s_f0 = gen.f0_upsamp(f0[:, None])                  # (1, 1, L)
        sine_waves = self.sine_gen(s_f0, phase_vec, noise) # (1, 9, L)
        sine_merge = gen.m_source.l_tanh(
            gen.m_source.l_linear(sine_waves.transpose(1, 2)))  # (1, L, 1)
        s = sine_merge.transpose(1, 2)                     # (1, 1, L)

        s_real, s_imag = self.stft(s.squeeze(1))
        s_stft = torch.cat([s_real, s_imag], dim=1)

        x = gen.conv_pre(mel)
        for i in range(gen.num_upsamples):
            x = F.leaky_relu(x, gen.lrelu_slope)
            x = gen.ups[i](x)
            if i == gen.num_upsamples - 1:
                x = gen.reflection_pad(x)
            si = gen.source_downs[i](s_stft)
            si = gen.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(gen.num_kernels):
                rb = gen.resblocks[i * gen.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / gen.num_kernels

        x = F.leaky_relu(x)
        x = gen.conv_post(x)
        n_bins = self.n_fft // 2 + 1
        magnitude = torch.exp(x[:, :n_bins, :])
        phase = torch.sin(x[:, n_bins:, :])
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        audio = self.istft(real, imag)                     # (1, L)
        return torch.clamp(audio, -gen.audio_limit, gen.audio_limit)
