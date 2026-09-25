"""Streaming (explicit-state) re-formulation of LocalVQE for Core ML export.

The upstream ``LocalVQE`` module is fully causal but written as a whole-clip
graph (causal zero padding on every time-axis conv, an unrolled S4D
recurrence, and a fold-based overlap-add).  ``StreamingLocalVQE`` wraps a
loaded upstream model and exposes a pure function

    (mic_hop, ref_hop, *states) -> (enhanced_hop, *new_states)

that processes ``frames`` consecutive 256-sample hops per call and carries
every piece of causal context (conv time histories, AlignBlock delay
windows, S4D hidden state, CCM frame history, DCT input history and the
synthesis overlap-add tail) as explicit tensors.  Feeding the outputs back
as the next call's inputs reproduces the whole-clip forward pass exactly
(verified in ``verify.py``).

Output timing: the analysis frame that finishes at the end of hop ``k``
covers samples ``[256k-256, 256k+256)`` so the synthesis of that frame
completes output samples ``[256k-512, 256k-256)``.  The emitted hop
therefore lags the input by one hop (256 samples = 16 ms).  A caller that
wants sample-aligned whole-clip output drops the first 256 emitted samples
and feeds one hop of zeros at the end.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .upstream.model import LocalVQE

HOP = 256
N_FFT = 512


def _conv_time_step(
    conv: nn.Conv2d, x: torch.Tensor, hist: torch.Tensor, pad_lr: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal conv over time with an explicit history of ``kh-1`` frames.

    ``x``/``hist`` are (1, C, T, F) / (1, C, kh-1, F); the history replaces
    the upstream top zero padding.  Returns (conv output, new history).
    """
    kh = conv.kernel_size[0]
    xs = torch.cat([hist, x], dim=2)
    new_hist = xs[:, :, xs.shape[2] - (kh - 1):, :]
    y = conv(F.pad(xs, (pad_lr[0], pad_lr[1], 0, 0)))
    return y, new_hist


class StreamingLocalVQE(nn.Module):
    """Explicit-state wrapper around a loaded upstream ``LocalVQE`` (arch v2/v3)."""

    def __init__(self, model: LocalVQE, frames: int = 1, ola_scale: float = 1.0):
        super().__init__()
        if model.arch_version not in (2, 3):
            raise ValueError(f"arch_version {model.arch_version} not supported (need 2 or 3)")
        self.m = model
        self.frames = frames
        self.ola_scale = ola_scale
        kh, kw = model.mic_enc1.conv.kernel_size
        self.kh = kh
        self.pad_lr = ((kw - 1) // 2, kw - 1 - (kw - 1) // 2)

        # S4D constants (polar -> cartesian), baked once.
        bn = model.bottleneck
        sp = F.softplus(bn.A_log_rate.detach()).clamp(min=0.01)
        r = torch.exp(-sp)
        self.register_buffer("a_real", (r * torch.cos(bn.A_theta.detach())).float())
        self.register_buffer("a_imag", (r * torch.sin(bn.A_theta.detach())).float())

        self.state_spec = self._build_state_spec()

    # ------------------------------------------------------------------ specs
    def _freqs(self) -> List[int]:
        from .upstream.model import compute_freq_progression

        return compute_freq_progression(self.m.n_freqs, self.m.mic_enc1.conv.kernel_size)

    def _build_state_spec(self) -> List[Tuple[str, Tuple[int, ...]]]:
        m = self.m
        h = self.kh - 1
        fq = self._freqs()
        spec: List[Tuple[str, Tuple[int, ...]]] = []

        def enc(name: str, block, f_in: int, f_out: int):
            spec.append((f"{name}_conv", (1, block.conv.in_channels, h, f_in)))
            spec.append((f"{name}_res", (1, block.resblock.conv.in_channels, h, f_out)))

        def dec(name: str, block, f_in: int):
            spec.append((f"{name}_res", (1, block.resblock.conv.in_channels, h, f_in)))
            spec.append((f"{name}_deconv", (1, block.deconv.conv.in_channels, h, f_in)))

        spec.append(("mic_pcm", (1, HOP)))
        spec.append(("ref_pcm", (1, HOP)))
        enc("mic_enc1", m.mic_enc1, fq[0], fq[1])
        enc("mic_enc2", m.mic_enc2, fq[1], fq[2])
        enc("far_enc1", m.far_enc1, fq[0], fq[1])
        enc("far_enc2", m.far_enc2, fq[1], fq[2])
        al = m.align
        spec.append(("align_k", (1, al.hidden_channels, al.dmax - 1, fq[2])))
        spec.append(("align_ref", (1, al.in_channels, al.dmax - 1, fq[2])))
        spec.append(("align_smooth", (1, al.hidden_channels, 4, al.dmax)))
        enc("mic_enc3", m.mic_enc3, fq[2], fq[3])
        enc("mic_enc4", m.mic_enc4, fq[3], fq[4])
        enc("mic_enc5", m.mic_enc5, fq[4], fq[5])
        spec.append(("s4d_real", (1, m.bottleneck.hidden_size)))
        spec.append(("s4d_imag", (1, m.bottleneck.hidden_size)))
        dec("dec5", m.dec5, fq[5])
        dec("dec4", m.dec4, fq[4])
        dec("dec3", m.dec3, fq[3])
        dec("dec2", m.dec2, fq[2])
        dec("dec1", m.dec1, fq[1])
        spec.append(("ccm", (1, 2, 2, m.n_freqs)))
        spec.append(("ola", (1, HOP)))
        return spec

    def zero_states(self) -> List[torch.Tensor]:
        return [torch.zeros(shape, dtype=torch.float32) for _, shape in self.state_spec]

    # ---------------------------------------------------------------- blocks
    def _encoder(self, block, x, hist_conv, hist_res):
        n = block.norm(x)
        y, hist_conv = _conv_time_step(block.conv, n, hist_conv, self.pad_lr)
        y = block.act(y)
        rb = block.resblock
        n2 = rb.norm(y)
        z, hist_res = _conv_time_step(rb.conv, n2, hist_res, self.pad_lr)
        return rb.act(z) + y, hist_conv, hist_res

    def _decoder(self, block, x, x_en, hist_res, hist_deconv):
        x_en = block.skip_norm(x_en)
        y = x + block.skip_conv(x_en)
        rb = block.resblock
        n = rb.norm(y)
        z, hist_res = _conv_time_step(rb.conv, n, hist_res, self.pad_lr)
        y = rb.act(z) + y
        dc = block.deconv
        n2 = dc.norm(y)
        w, hist_deconv = _conv_time_step(dc.conv, n2, hist_deconv, self.pad_lr)
        w = rearrange(w, "b (r c) t f -> b c t (r f)", r=2)
        if not block.is_last:
            w = block.act(w)
        return w, hist_res, hist_deconv

    def _align(self, x_mic, x_ref, hist_k, hist_ref, hist_smooth):
        al = self.m.align
        T = x_mic.shape[2]
        dmax = al.dmax
        Fq = x_ref.shape[3]
        Q = al.pconv_mic(x_mic)  # (1,H,T,F)
        K = al.pconv_ref(x_ref)  # (1,H,T,F)
        k_full = torch.cat([hist_k, K], dim=2)  # (1,H,T+dmax-1,F)
        new_hist_k = k_full[:, :, k_full.shape[2] - (dmax - 1):, :]
        # Ku[:, :, t, d, :] = k_full[:, :, t + d, :]  (d = dmax-1 is the current frame)
        ku = torch.stack([k_full[:, :, d:d + T, :] for d in range(dmax)], dim=3)  # (1,H,T,dmax,F)
        V = torch.sum(Q.unsqueeze(3) * ku, dim=-1) / (Fq**0.5)  # (1,H,T,dmax)
        v_full = torch.cat([hist_smooth, V], dim=2)  # (1,H,T+4,dmax)
        new_hist_smooth = v_full[:, :, v_full.shape[2] - 4:, :]
        Vs = al.conv[1](F.pad(v_full, (1, 1, 0, 0)))  # (1,1,T,dmax)
        A = torch.softmax(Vs / al.temperature, dim=-1)  # (1,1,T,dmax)
        r_full = torch.cat([hist_ref, x_ref], dim=2)  # (1,C,T+dmax-1,F)
        new_hist_ref = r_full[:, :, r_full.shape[2] - (dmax - 1):, :]
        ru = torch.stack([r_full[:, :, d:d + T, :] for d in range(dmax)], dim=3)  # (1,C,T,dmax,F)
        aligned = torch.sum(ru * A.unsqueeze(-1), dim=3)  # (1,C,T,F)
        return aligned, new_hist_k, new_hist_ref, new_hist_smooth

    def _s4d(self, x, h_real, h_imag):
        bn = self.m.bottleneck
        C = x.shape[1]
        u = rearrange(x, "b c t f -> b t (c f)")
        v = bn.input_proj(u)  # (1,T,N)
        ys = []
        for t in range(v.shape[1]):
            v_t = v[:, t, :]
            h_real_new = self.a_real * h_real - self.a_imag * h_imag + bn.B_real * v_t
            h_imag_new = self.a_real * h_imag + self.a_imag * h_real + bn.B_imag * v_t
            h_real, h_imag = h_real_new, h_imag_new
            ys.append(bn.C_real * h_real - bn.C_imag * h_imag)
        y = torch.stack(ys, dim=1)
        out = bn.output_proj(y) + bn.D * u
        return rearrange(out, "b t (c f) -> b c t f", c=C), h_real, h_imag

    def _ccm(self, mask, x_spec, hist):
        """mask (1,27,T,F), x_spec (1,2,T,F) channel-major spectrum, hist (1,2,2,F)."""
        ccm = self.m.mask
        T = mask.shape[2]
        m = rearrange(mask, "b (r c) t f -> b r c t f", r=3)
        H_real = torch.sum(ccm.v_real[None, :, None, None, None] * m, dim=1)  # (1,9,T,F)
        H_imag = torch.sum(ccm.v_imag[None, :, None, None, None] * m, dim=1)
        xs = torch.cat([hist, x_spec], dim=2)  # (1,2,T+2,F)
        new_hist = xs[:, :, xs.shape[2] - 2:, :]
        xp = F.pad(xs, (1, 1, 0, 0))  # (1,2,T+2,F+2)
        Fq = x_spec.shape[3]
        taps = [xp[:, :, mm:mm + T, nn:nn + Fq] for mm in range(3) for nn in range(3)]
        xu = torch.stack(taps, dim=2)  # (1,2,9,T,F)
        xr, xi = xu[:, 0], xu[:, 1]  # (1,9,T,F)
        enh_real = torch.sum(H_real * xr - H_imag * xi, dim=1)  # (1,T,F)
        enh_imag = torch.sum(H_real * xi + H_imag * xr, dim=1)
        return torch.stack([enh_real, enh_imag], dim=1), new_hist  # (1,2,T,F)

    # --------------------------------------------------------------- forward
    def _analysis(self, pcm_hop, hist):
        """pcm_hop (1, HOP*T), hist (1, HOP) -> spectrum (1,2,T,F) channel-major, new hist."""
        x = torch.cat([hist, pcm_hop], dim=1)  # (1, HOP*(T+1))
        new_hist = x[:, x.shape[1] - HOP:]
        out = self.m.encoder.conv(x.unsqueeze(1))  # (1, 512, T)
        spec = out.reshape(1, self.m.n_freqs, 2, -1).permute(0, 2, 3, 1)  # (1,2,T,F)
        return spec, new_hist

    def _fe(self, spec):
        """Power-law compression on a channel-major (1,2,T,F) spectrum."""
        c = self.m.fe_mic.c
        mag = torch.sqrt(spec[:, 0:1] ** 2 + spec[:, 1:2] ** 2 + 1e-12)
        return spec / (mag.pow(1 - c) + 1e-12)

    def _synthesis(self, enh, ola):
        """enh (1,2,T,F) -> pcm (1, HOP*T), new ola tail (1, HOP)."""
        T = enh.shape[2]
        x = enh.permute(0, 2, 3, 1).reshape(1, T, -1)  # (1,T,2F) index f*2+c
        frames = self.m.decoder.linear(x)  # (1,T,512)
        a = frames[:, :, :HOP]
        b = frames[:, :, HOP:]
        prev = torch.cat([ola.unsqueeze(1), b[:, : T - 1, :]], dim=1) if T > 1 else ola.unsqueeze(1)
        out = (prev + a) * self.ola_scale
        return out.reshape(1, HOP * T), b[:, T - 1, :]

    def forward(self, mic, ref, *states):
        st: Dict[str, torch.Tensor] = {name: s for (name, _), s in zip(self.state_spec, states)}
        m = self.m

        mic_spec, st["mic_pcm"] = self._analysis(mic, st["mic_pcm"])
        ref_spec, st["ref_pcm"] = self._analysis(ref, st["ref_pcm"])
        mic_fe = self._fe(mic_spec)
        ref_fe = self._fe(ref_spec)

        e1, st["mic_enc1_conv"], st["mic_enc1_res"] = self._encoder(
            m.mic_enc1, mic_fe, st["mic_enc1_conv"], st["mic_enc1_res"])
        e2, st["mic_enc2_conv"], st["mic_enc2_res"] = self._encoder(
            m.mic_enc2, e1, st["mic_enc2_conv"], st["mic_enc2_res"])
        f1, st["far_enc1_conv"], st["far_enc1_res"] = self._encoder(
            m.far_enc1, ref_fe, st["far_enc1_conv"], st["far_enc1_res"])
        f2, st["far_enc2_conv"], st["far_enc2_res"] = self._encoder(
            m.far_enc2, f1, st["far_enc2_conv"], st["far_enc2_res"])

        aligned, st["align_k"], st["align_ref"], st["align_smooth"] = self._align(
            e2, f2, st["align_k"], st["align_ref"], st["align_smooth"])
        concat = torch.cat([e2, aligned], dim=1)

        e3, st["mic_enc3_conv"], st["mic_enc3_res"] = self._encoder(
            m.mic_enc3, concat, st["mic_enc3_conv"], st["mic_enc3_res"])
        e4, st["mic_enc4_conv"], st["mic_enc4_res"] = self._encoder(
            m.mic_enc4, e3, st["mic_enc4_conv"], st["mic_enc4_res"])
        e5, st["mic_enc5_conv"], st["mic_enc5_res"] = self._encoder(
            m.mic_enc5, e4, st["mic_enc5_conv"], st["mic_enc5_res"])

        bn, st["s4d_real"], st["s4d_imag"] = self._s4d(e5, st["s4d_real"], st["s4d_imag"])

        d5, st["dec5_res"], st["dec5_deconv"] = self._decoder(m.dec5, bn, e5, st["dec5_res"], st["dec5_deconv"])
        d5 = d5[..., : e4.shape[-1]]
        d4, st["dec4_res"], st["dec4_deconv"] = self._decoder(m.dec4, d5, e4, st["dec4_res"], st["dec4_deconv"])
        d4 = d4[..., : e3.shape[-1]]
        d3, st["dec3_res"], st["dec3_deconv"] = self._decoder(m.dec3, d4, e3, st["dec3_res"], st["dec3_deconv"])
        d3 = d3[..., : e2.shape[-1]]
        d2, st["dec2_res"], st["dec2_deconv"] = self._decoder(m.dec2, d3, e2, st["dec2_res"], st["dec2_deconv"])
        d2 = d2[..., : e1.shape[-1]]
        d1, st["dec1_res"], st["dec1_deconv"] = self._decoder(m.dec1, d2, e1, st["dec1_res"], st["dec1_deconv"])
        d1 = d1[..., : mic_fe.shape[-1]]

        enh, st["ccm"] = self._ccm(d1, mic_spec, st["ccm"])
        pcm, st["ola"] = self._synthesis(enh, st["ola"])

        return (pcm, *[st[name] for name, _ in self.state_spec])


def run_streaming(
    sm: StreamingLocalVQE, mic: torch.Tensor, ref: torch.Tensor, step_fn=None
) -> torch.Tensor:
    """Drive a streaming model over whole (1, N) clips; returns sample-aligned (1, N).

    ``step_fn(mic_hop, ref_hop, states) -> (out, states)`` defaults to the
    PyTorch module; ``convert-coreml.py`` passes a Core ML-backed step.
    """
    n = mic.shape[1]
    hop_len = HOP * sm.frames
    n_pad = (-n) % hop_len
    mic_p = F.pad(mic, (0, n_pad + hop_len))
    ref_p = F.pad(ref, (0, n_pad + hop_len))
    states = sm.zero_states()
    outs = []
    if step_fn is None:

        def step_fn(a, b, s):
            with torch.no_grad():
                res = sm(a, b, *s)
            return res[0], list(res[1:])

    for i in range(0, mic_p.shape[1], hop_len):
        out, states = step_fn(mic_p[:, i:i + hop_len], ref_p[:, i:i + hop_len], states)
        outs.append(out)
    y = torch.cat(outs, dim=1)
    return y[:, HOP:HOP + n]
