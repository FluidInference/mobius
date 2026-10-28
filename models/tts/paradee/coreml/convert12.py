"""Paradee-8M v1.2 (four voices) -> Core ML: the v1.0 two-model layout, plus a `voice` input on both models.

  ParadeeText      input_ids [1,T] int32 (0 pad at both ends, T<=512), voice [1] int32
                   -> duration [1,T] (pre-round), d [1,224,T], asr_tok [1,512,T]
  host             n = max(1, round(duration / speed)); en = d.repeat(n), asr = asr_tok.repeat(n)  (F = sum n)
                   noise = N(0,1) [1,1,600F]
  ParadeeAcoustic  en [1,224,F], asr [1,512,F], noise [1,1,600F], voice [1] int32
                   -> F0Ntrain -> decoder A trunk -> frame-rate source-filter head -> audio [1,600F] @ 24 kHz

`voice` is the row of each half's style table, in config.json order (af_heart, bf_isabella, am_puck, bm_fable).
The head's noise is an input (upstream draws it in-graph), so the graph is deterministic and checkable.

The head (vhead.SourceFilterHead2) is rewritten for Core ML:
- STFT / iSTFT: frames built from shifted hop blocks (n_fft = 4 hop, burst 2 hop) + one GEMM; inverse is GEMM +
  shifted overlap-add. No complex ops, no strided convs, no gathers.
- Harmonic phase: upstream accumulates cycles per frame in float64. Core ML has no float64, and an fp32 running sum
  reaches ~4e3 cycles, where one ulp (5e-4 cycles) is 0.03 cycles on harmonic 64. Here the per-frame advance, wrapped
  to [0, 1), is rounded to 1/2^17 cycle and summed in int32 (exact: 8000 frames * 2^17 < 2^31), then wrapped.
- Sine bank: sum_{k<=M} sin(k t) = sin(M t/2) sin((M+1) t/2) / sin(t/2), with M the frame's harmonic count, instead of
  a [64, samples] tensor (615 MB at 100 s). Near sin(t/2) = 0 it uses the first-order limit M(M+1)/2 * sin(t).

usage: uv run python convert12.py [--out build-v1.2] [--max-frames 4000]
"""
import argparse
import math
import pathlib

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ref12
from convert import CoreMLAlbert  # also patches AdainResBlk1d.forward (no rsqrt of an int tensor)
from ref12 import FHOP, SR

Q = 1 << 17  # harmonic phase fixed point: 1/Q cycle


def dft_bases(n):
    """export_paradee_v1_1.ConvSTFT bases as GEMM operands: fwd [n, 2K], inv [2K, n], w2 [n] (hann, periodic)."""
    w = torch.hann_window(n, dtype=torch.float64)
    k = torch.arange(n // 2 + 1, dtype=torch.float64)[:, None]
    ang = 2 * torch.pi * ((k * torch.arange(n, dtype=torch.float64)) % n) / n
    c = torch.full((n // 2 + 1, 1), 2.0, dtype=torch.float64)
    c[0] = c[-1] = 1
    fwd = (torch.cat([torch.cos(ang), -torch.sin(ang)]) * w).T
    inv = torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w
    return fwd.float(), inv.float(), (w**2).float()


def frames(x, n, hop):
    """[S] (S % hop == 0) -> torch.stft(center=True, reflect) frames [S/hop + 1, n], n = r * hop."""
    r = n // hop
    b = F.pad(x[None, None], (n // 2, n // 2), mode="reflect")[0, 0].reshape(-1, hop)  # [S/hop + r, hop]
    return torch.cat([b[j : (j - r + 1) if j < r - 1 else None] for j in range(r)], 1)


def overlap_add(fr, n, hop):
    """[nf, n] -> [(nf + r - 1) * hop] for n = r * hop, == conv_transpose1d(stride=hop). n is a Python int: a traced
    fr.shape would turn the pad widths into graph values."""
    r = n // hop
    q = fr.reshape(-1, r, hop)
    acc = F.pad(q[:, 0], (0, 0, 0, r - 1))
    for j in range(1, r):
        acc = acc + F.pad(q[:, j], (0, 0, j, r - 1 - j))
    return acc.reshape(-1)


class Head(nn.Module):
    """vhead.SourceFilterHead2.forward (with the burst path) for batch 1, Core ML friendly."""

    def __init__(self, h):
        super().__init__()
        assert h.burst and h.mod == 2 and h.up == 2 and h.n_fft == 4 * h.hop, (h.burst, h.mod, h.up, h.n_fft, h.hop)
        self.h = h
        for name, n in (("m", h.n_fft), ("b", 150)):
            fwd, inv, w2 = dft_bases(n)
            self.register_buffer(f"fwd_{name}", fwd)
            self.register_buffer(f"inv_{name}", inv)
            self.register_buffer(f"w2_{name}", w2[None])
        s = torch.arange(FHOP, dtype=torch.float64) + 0.5
        self.register_buffer("a_lo", (s[: FHOP // 2] / FHOP + 0.5).float())  # sample's position after the previous centre
        self.register_buffer("a_hi", (s[FHOP // 2 :] / FHOP - 0.5).float())  # ... after its own frame centre
        self.register_buffer("fr", torch.linspace(0, SR / 2, h.nb)[:, None])
        self.register_buffer("ones_hop", torch.ones(1, FHOP))
        self.register_buffer("ones_up", torch.ones(1, h.up))

    def stft(self, x, name, n, hop):
        y = frames(x, n, hop) @ getattr(self, f"fwd_{name}")  # [nf, 2K]
        k = n // 2 + 1
        return y[:, :k].T, y[:, k:].T

    def istft(self, re, im, name, n, hop):
        y = overlap_add(torch.cat([re, im], 0).T @ getattr(self, f"inv_{name}"), n, hop)
        env = overlap_add(torch.ones_like(re[:1]).T * getattr(self, f"w2_{name}"), n, hop)
        return (y / env.clamp_min(1e-8))[n // 2 : -(n // 2)]

    def sine_bank(self, f):
        """f [n] F0 per 80 fps frame -> e [S] (vhead.sine_bank, K=64, * 0.1), cyc [n, 300], uv [n]."""
        h = self.h
        r = f * (FHOP / SR)  # cycles per frame
        rq = torch.round((r - torch.floor(r)) * Q).to(torch.int32)
        cq = torch.cumsum(rq, 0)
        c = (cq - torch.div(cq, Q, rounding_mode="floor") * Q).to(torch.float32) * (1.0 / Q)  # wrapped running phase
        d = torch.cat([r[1:], torch.zeros_like(r[:1])])  # advance to the next frame centre
        c_prev = torch.cat([c[:1], c[:-1]])
        d_prev = torch.cat([torch.zeros_like(r[:1]), d[:-1]])
        cyc = torch.cat([c_prev[:, None] + self.a_lo * d_prev[:, None], c[:, None] + self.a_hi * d[:, None]], 1)
        uv = (f > 10).to(torch.float32)
        m = (torch.clamp(torch.ceil(11000.0 / f) - 1, 0.0, float(h.K)) * uv)[:, None]  # harmonics below 11 kHz, 0 if unvoiced
        half = torch.pi * cyc
        den = torch.sin(half)
        small = torch.abs(den) < 1e-4
        num = torch.sin(m * half) * torch.sin((m + 1) * half)
        e = torch.where(small, 0.5 * m * (m + 1) * torch.sin(2 * half), num / torch.where(small, torch.ones_like(den), den))
        return (e * 0.1).reshape(-1), cyc, uv

    def forward(self, x, f0, nz):
        """x [1, 128, n] trunk output, f0 [1, n], nz [S = 300 n] -> audio [S]."""
        h = self.h
        z = h.norm0(h.inp(x).transpose(1, 2)).transpose(1, 2)
        for b in h.blocks:
            z = b(z)
        hn = h.norm1(z.transpose(1, 2))  # [1, n, dim]
        o = h.out(hn).reshape(1, -1, h.C * h.G + 1)  # [1, 2n, C G + 1]
        o = torch.cat([o, o[:, -1:]], 1)[0]  # centred STFT has one more frame: [nf, C G + 1]
        g = (o[:, :-1].reshape(-1, h.C, h.G) @ h.M.T).permute(1, 2, 0)  # [C, nb, nf]
        gh, gn = torch.exp(g[0].clamp(max=6.0)), torch.exp(g[3].clamp(max=6.0))
        um = torch.sqrt(g[1] ** 2 + g[2] ** 2).clamp_min(1e-4)
        ur, ui = g[1] / um, g[2] / um

        f = f0[0]
        e, cyc, uv = self.sine_bank(f)
        uvs = (uv[:, None] * self.ones_hop).reshape(-1)
        cycs = cyc.reshape(-1)
        n_, hop = h.n_fft, h.hop
        Er, Ei = self.stft(e, "m", n_, hop)
        Nr, Ni = self.stft(nz, "m", n_, hop)
        j = 0
        for m in range(1, h.mod + 1):
            for fn in (torch.cos, torch.sin):
                mr, mi = self.stft(nz * uvs * fn(2 * torch.pi * m * cycs), "m", n_, hop)
                Nr = Nr + g[4 + j] * mr
                Ni = Ni + g[4 + j] * mi
                j += 1
        f0f = torch.cat([(f[:, None] * self.ones_up).reshape(-1), f[-1:]]).clamp_min(60.0)  # per STFT frame
        ang = 2 * torch.pi * (self.fr / f0f[None]) * (0.5 * torch.tanh(o[:, -1]))[None]
        ca, sa = torch.cos(ang), torch.sin(ang)
        Hr, Hi = gh * (ur * ca - ui * sa), gh * (ur * sa + ui * ca)
        y = self.istft(Er * Hr - Ei * Hi + Nr * gn, Er * Hi + Ei * Hr + Ni * gn, "m", n_, hop)
        gb = torch.exp((h.out_b(hn).reshape(1, -1, h.Gb) @ h.Mb.T).clamp(max=6.0))[0].T  # [76, 4n]
        gb = torch.cat([gb, gb[:, -1:]], 1)
        br, bi = self.stft(torch.flip(nz, [0]), "b", 150, 75)
        return y + self.istft(br * gb, bi * gb, "b", 150, 75)


class TextModel12(nn.Module):
    def __init__(self, ts, tab):
        super().__init__()
        self.ts = ts
        self.bert = CoreMLAlbert(ts.bert)
        self.register_buffer("tab", tab)

    def forward(self, ids, voice):
        ts = self.ts
        p = ts.predictor
        s = torch.index_select(self.tab, 0, voice.long())  # [1, 32]
        x = ts.bert_encoder(self.bert(ids)).transpose(-1, -2)
        sx = s[:, :, None] * torch.ones_like(x[:, :1])  # [1, 32, T]
        x = torch.cat([x, sx], 1)
        for b in p.text_encoder.lstms:
            if isinstance(b, nn.LSTM):
                x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
            else:
                x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
        dur = torch.sigmoid(p.duration_proj(p.lstm(x.transpose(-1, -2))[0])).sum(-1)
        e = ts.text_encoder
        h = e.embedding(ids).transpose(1, 2)
        for c in e.cnn:
            h = c(h)
        asr_tok = ts.asr_proj(e.lstm(h.transpose(1, 2))[0]).transpose(-1, -2)
        return dur, x, asr_tok


class AcousticModel12(nn.Module):
    def __init__(self, ts, vd, ts_tab, vd_tab):
        super().__init__()
        self.pred, self.vd, self.head = ts.predictor, vd, Head(vd.head)
        self.register_buffer("ts_tab", ts_tab)
        self.register_buffer("vd_tab", vd_tab)

    def forward(self, en, asr, noise, voice):
        st = torch.index_select(self.ts_tab, 0, voice.long())
        sd = torch.index_select(self.vd_tab, 0, voice.long())
        F0, N = self.pred.F0Ntrain(en, st)
        v = self.vd
        f0c = v.F0_conv(F0.unsqueeze(1))
        nc = v.N_conv(N.unsqueeze(1))
        x = v.encode(torch.cat([asr, f0c, nc], 1), sd)
        asr_res = v.asr_res(asr)
        for block in v.decode:  # only the last block upsamples, so every block gets the residual inputs
            x = block(torch.cat([x, asr_res, f0c, nc], 1), sd)
        return self.head(x, F0, noise[0, 0])[None]


def snr(ref, x):
    ref, x = np.asarray(ref, np.float64).ravel(), np.asarray(x, np.float64).ravel()
    return 10 * math.log10((ref**2).sum() / max(((ref - x) ** 2).sum(), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("build-v1.2"))
    ap.add_argument("--max-frames", type=int, default=4000, help="max aligned frames F (600F samples; 4000 = 100 s)")
    A = ap.parse_args()
    A.out.mkdir(parents=True, exist_ok=True)

    ts, vd, ts_tab, vd_tab = ref12.load12()
    tm, am = TextModel12(ts, ts_tab).eval(), AcousticModel12(ts, vd, ts_tab, vd_tab).eval()
    ps = "pəɹˈædi ɪz ɐ smˈɔl vˈYs ðæt ɹˈʌnz ˈɛniwˌɛɹ."
    ids = ref12.phonemes_to_ids(ps)
    refs = {}
    for v, voice in enumerate(ref12.VOICES):
        ref, mid = ref12.reference(ts, vd, ts_tab, vd_tab, ids, v, seed=v)
        vt = torch.tensor([v], dtype=torch.int32)
        with torch.no_grad():
            dur, dmat, asr_tok = tm(ids, vt)
            wav = am(mid["en"], mid["asr"], mid["noise"][None], vt)
        print(f"{voice:12s} text wrapper vs ref: dur {(dur[0] - mid['dur']).abs().max():.1e} "
              f"d {(dmat - mid['d'].transpose(-1, -2)).abs().max():.1e} asr {(asr_tok - mid['asr_tok']).abs().max():.1e}"
              f" | acoustic wrapper vs ref: SNR {snr(ref, wav):.1f} dB, max|diff| {(wav - ref).abs().max():.1e}")
        refs[v] = (ref, mid)

    v0 = 1  # trace with a British voice so nothing voice-specific gets baked in as a constant
    ref, mid = refs[v0]
    Fa = mid["aln"].shape[2]
    vt = torch.tensor([v0], dtype=torch.int32)
    T = ct.RangeDim(lower_bound=3, upper_bound=512, default=ids.shape[1])
    Fd = ct.RangeDim(lower_bound=1, upper_bound=A.max_frames, default=Fa)
    Sd = ct.RangeDim(lower_bound=2 * FHOP, upper_bound=A.max_frames * 2 * FHOP, default=Fa * 2 * FHOP)
    with torch.no_grad():
        t_tr = torch.jit.trace(tm, (ids.int(), vt), strict=False)
        a_tr = torch.jit.trace(am, (mid["en"], mid["asr"], mid["noise"][None], vt), strict=False)
    voice_in = ct.TensorType(name="voice", shape=(1,), dtype=np.int32)
    text = ct.convert(
        t_tr,
        inputs=[ct.TensorType(name="input_ids", shape=(1, T), dtype=np.int32), voice_in],
        outputs=[ct.TensorType(name="duration"), ct.TensorType(name="d"), ct.TensorType(name="asr_tok")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    text.short_description = ("Paradee-8M v1.2 text side: phoneme ids + voice (0 af_heart, 1 bf_isabella, 2 am_puck, "
                              "3 bm_fable) -> durations, prosody features, asr features")
    text.save(str(A.out / "ParadeeText.mlpackage"))
    acou = ct.convert(
        a_tr,
        inputs=[
            ct.TensorType(name="en", shape=(1, dmat.shape[1], Fd), dtype=np.float32),
            ct.TensorType(name="asr", shape=(1, 512, Fd), dtype=np.float32),
            ct.TensorType(name="noise", shape=(1, 1, Sd), dtype=np.float32),
            voice_in,
        ],
        outputs=[ct.TensorType(name="audio")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    acou.short_description = "Paradee-8M v1.2 acoustic side: aligned features + noise + voice -> 24 kHz audio"
    acou.save(str(A.out / "ParadeeAcoustic.mlpackage"))

    arrays = {"input_ids": ids.numpy().astype(np.int32)}
    for v, (ref, mid) in refs.items():
        arrays.update({f"duration_{v}": mid["dur"].numpy()[None], f"d_{v}": mid["d"].transpose(-1, -2).numpy(),
                       f"asr_tok_{v}": mid["asr_tok"].numpy(), f"en_{v}": mid["en"].numpy(), f"asr_{v}": mid["asr"].numpy(),
                       f"noise_{v}": mid["noise"][None].numpy(), f"audio_{v}": ref.numpy()})
    np.savez(A.out / "ref.npz", **arrays)
    print(f"saved {A.out}/ParadeeText.mlpackage, ParadeeAcoustic.mlpackage, ref.npz")


if __name__ == "__main__":
    main()
