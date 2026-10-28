"""Paradee-8M v1.0 -> Core ML (2 models + host-side duration expansion).

  ParadeeText.mlpackage      input_ids [1,T] int32 (0 pad at both ends, T<=512)
                             -> duration [1,T] (pre-round), d [1,224,T], asr_tok [1,512,T]
  host                       n = max(1, round(duration / speed)); en = d.repeat(n), asr = asr_tok.repeat(n)  (F = sum n)
  ParadeeAcoustic.mlpackage  en [1,224,F], asr [1,512,F], noise [1,1,600F] (N(0,1))
                             -> F0Ntrain -> decoder (iSTFTNet) -> phase-lock filter -> audio [1,600F] @ 24 kHz

The harmonic source's Gaussian noise is an input instead of an in-graph RNG op, so the graph is deterministic
and checkable against PyTorch. One channel suffices: kokoro adds noise per harmonic (9 ch) and then mixes the
harmonics with l_linear, so W @ (a * n) == a * ||W|| * z with z ~ N(0,1). kokoro's random initial phases are
dropped too: they only touch sample 0, which the 1/300 linear downsample never reads.

usage: uv run python convert.py [--out build] [--max-frames 4000]
"""
import argparse
import math
import pathlib
import time

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import paradee_ref as R
from kokoro.istftnet import AdainResBlk1d

UP = 300  # samples per F0 frame (prod(upsample_rates) * istft hop = 10 * 6 * 5)


def _resblk_forward(self, x, s):
    """kokoro AdainResBlk1d.forward without rsqrt(int tensor), which Core ML rejects."""
    return (self._residual(x, s) + self._shortcut(x)) * (1.0 / math.sqrt(2.0))


AdainResBlk1d.forward = _resblk_forward


class CoreMLAlbert(nn.Module):
    """Traceable ALBERT forward (no HF masking machinery). Single unpadded sentence -> no attention mask."""

    def __init__(self, hf):
        super().__init__()
        c = hf.config
        self.L, self.H, self.D = c.num_hidden_layers, c.num_attention_heads, c.hidden_size
        self.hd = self.D // self.H
        e = hf.embeddings
        self.word, self.pos, self.tok, self.eln = e.word_embeddings, e.position_embeddings, e.token_type_embeddings, e.LayerNorm
        self.proj = hf.encoder.embedding_hidden_mapping_in
        layer = hf.encoder.albert_layer_groups[0].albert_layers[0]
        a = layer.attention
        self.q, self.k, self.v, self.dense, self.aln = a.query, a.key, a.value, a.dense, a.LayerNorm
        self.ffn, self.ffn_out, self.fln = layer.ffn, layer.ffn_output, layer.full_layer_layer_norm
        assert c.hidden_act == "gelu_new", c.hidden_act

    def forward(self, ids):
        T = ids.shape[1]
        pos = torch.arange(T).unsqueeze(0)
        h = self.eln(self.word(ids) + self.pos(pos) + self.tok(torch.zeros_like(ids)))
        h = self.proj(h)
        for _ in range(self.L):
            q = self.q(h).view(1, T, self.H, self.hd).transpose(1, 2)
            k = self.k(h).view(1, T, self.H, self.hd).transpose(1, 2)
            v = self.v(h).view(1, T, self.H, self.hd).transpose(1, 2)
            att = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd), dim=-1)
            ctx = (att @ v).transpose(1, 2).reshape(1, T, self.D)
            a = self.aln(self.dense(ctx) + h)
            f = self.ffn_out(F.gelu(self.ffn(a), approximate="tanh"))
            h = self.fln(f + a)
        return h


class TextModel(nn.Module):
    def __init__(self, ts):
        super().__init__()
        self.ts = ts
        self.bert = CoreMLAlbert(ts.bert)

    def forward(self, ids):
        ts = self.ts
        p, s = ts.predictor, ts.style
        x = ts.bert_encoder(self.bert(ids)).transpose(-1, -2)
        T = ids.shape[1]
        sx = s[:, :, None].expand(-1, -1, T)
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


def source(gen, F0, z):
    """kokoro SourceModuleHnNSF(f0_upsamp(F0)) with explicit noise. F0 [1,2F], z [1,1,S] -> [1,S] (pre-STFT source)."""
    sg = gen.m_source.l_sin_gen
    k = torch.arange(1, sg.harmonic_num + 2, dtype=F0.dtype)
    rad = F0[:, :, None] * k / sg.sampling_rate
    rad = rad - torch.floor(rad)  # [1,2F,H] == kokoro's 1/300 downsample
    phase = torch.cumsum(rad, dim=1) * (2 * math.pi)
    phase = F.interpolate(phase.transpose(1, 2) * UP, scale_factor=float(UP), mode="linear")  # [1,H,S]
    f0_up = torch.repeat_interleave(F0, UP, dim=1)[:, None]  # [1,1,S]
    uv = (f0_up > sg.voiced_threshold).float()
    W = gen.m_source.l_linear.weight  # [1,H]
    harm = (W[:, :, None] * torch.sin(phase) * sg.sine_amp).sum(1, keepdim=True) * uv
    noise_amp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
    return torch.tanh(harm + gen.m_source.l_linear.bias + noise_amp * W.norm() * z)[:, 0]


def stft_inverse(st, mag, phase):
    """ConvSTFT.inverse with a shape-derived ones tensor (traceable for dynamic lengths)."""
    z = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], 1)
    y = F.conv_transpose1d(z, st.inv, stride=st.hop)[:, 0]
    env = F.conv_transpose1d(torch.ones_like(mag[:, :1]), st.w2, stride=st.hop)[:, 0]
    return (y / env.clamp_min(1e-8))[:, st.n // 2 : -(st.n // 2)]


def _frames(x):
    """[1,S] -> reflect-padded STFT frames [nf,N_FFT] (torch.stft center=True framing) via one gather."""
    xp = F.pad(x[:, None], (R.N_FFT // 2, R.N_FFT // 2), mode="reflect")[0, 0]
    nf = x.shape[1] // R.HOP + 1
    idx = torch.arange(nf)[:, None] * R.HOP + torch.arange(R.N_FFT)[None]
    return torch.index_select(xp, 0, idx.reshape(-1).long()).reshape(nf, R.N_FFT)


def _overlap_add(fr):
    """[nf,N_FFT] -> [(nf-1)*HOP + N_FFT], == conv_transpose1d(stride=HOP). N_FFT = 4 * HOP."""
    q = fr.reshape(-1, 4, R.HOP)
    acc = F.pad(q[:, 0], (0, 0, 0, 3)) + F.pad(q[:, 1], (0, 0, 1, 2)) + F.pad(q[:, 2], (0, 0, 2, 1)) + F.pad(q[:, 3], (0, 0, 3, 0))
    return acc.reshape(-1)


def lock(lk, s, src, F0):
    """paradee_ref.Lock.forward (s and src are both 600F long). The 1024-point band STFT/iSTFT run as gather + GEMM +
    overlap-add instead of 1024-tap strided convs, which the Core ML CPU backend executes ~40x slower."""
    fwd = lk.fwd[:, 0].T  # [N_FFT, 2K]
    K = lk.K
    S = _frames(s) @ fwd  # [nf, 2K]
    E = _frames(src) @ fwd
    Sr, Si, Er, Ei = S[:, :K].T, S[:, K:].T, E[:, :K].T, E[:, K:].T  # [K, nf]
    nf = S.shape[0]
    fidx = torch.arange(nf, dtype=torch.float32)
    j = torch.clamp(torch.floor(fidx * R.HOP / UP), max=F0.shape[1] - 1).long()
    v = (torch.index_select(F0[0], 0, j) > 60).float()[None]
    Em = torch.sqrt(Er**2 + Ei**2).clamp_min(1e-6)
    Rr, Ri = (Sr * Er + Si * Ei) / Em, (Si * Er - Sr * Ei) / Em
    Rr = F.conv1d(Rr[:, None], lk.sk, padding=lk.pad)[:, 0]
    Ri = F.conv1d(Ri[:, None], lk.sk, padding=lk.pad)[:, 0]
    Rm = torch.sqrt(Rr**2 + Ri**2).clamp_min(1e-12)
    Sm = torch.sqrt(Sr**2 + Si**2)
    ur, ui = (Er * Rr - Ei * Ri) / (Em * Rm), (Er * Ri + Ei * Rr) / (Em * Rm)
    Dr, Di = (Sm * ur - Sr) * v, (Sm * ui - Si) * v
    ola = _overlap_add(torch.cat([Dr, Di]).T @ lk.inv[:, 0])  # [2K,nf]^T @ [2K,N_FFT]
    env = _overlap_add(torch.ones_like(Dr[:1]).T * lk.w2[0])
    n = s.shape[1]
    return s + (ola / env.clamp_min(1e-8))[None, R.N_FFT // 2 : R.N_FFT // 2 + n]


class AcousticModel(nn.Module):
    def __init__(self, ts, ds):
        super().__init__()
        self.pred, self.s_text = ts.predictor, ts.style
        self.ds, self.s_dec = ds, ds.style
        self.lk = R.Lock()

    def forward(self, en, asr, z):
        F0, N = self.pred.F0Ntrain(en, self.s_text)
        d, s = self.ds, self.s_dec
        f0c = d.F0_conv(F0.unsqueeze(1))
        nc = d.N_conv(N.unsqueeze(1))
        x = d.encode(torch.cat([asr, f0c, nc], 1), s)
        asr_res = d.asr_res(asr)
        for block in d.decode:
            x = block(torch.cat([x, asr_res, f0c, nc], 1), s)  # only the last block upsamples
        g = d.generator
        src = source(g, F0, z)
        har = torch.cat(g.stft.transform(src), 1)
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            xs_ = g.noise_res[i](g.noise_convs[i](har), s)
            x = g.ups[i](x)
            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)
            x = x + xs_
            acc = None
            for j in range(g.num_kernels):
                r = g.resblocks[i * g.num_kernels + j](x, s)
                acc = r if acc is None else acc + r
            x = acc / g.num_kernels
        x = g.conv_post(F.leaky_relu(x))
        nb = g.post_n_fft // 2 + 1
        wav = stft_inverse(g.stft, torch.exp(x[:, :nb]), torch.sin(x[:, nb:]))
        return lock(self.lk, wav, src, F0)


def reference(ts, ds, ids, speed=1.0, seed=0):
    """Upstream pipeline (random source noise from kokoro's own SineGen), plus the matching single-channel z."""
    cap = {}
    sg = ds.generator.m_source.l_sin_gen
    orig = torch.randn_like

    def grab(x, *a, **k):
        y = orig(x, *a, **k)
        if x.shape[-1] == sg.dim and "n" not in cap:
            cap["n"] = y
        return y

    hook = ds.generator.m_source.register_forward_hook(lambda m, i, o: cap.__setitem__("src", o[0]))
    with torch.no_grad():
        d, dur, asr_tok = R.text_forward(ts, ids)
        aln = R.alignment(R.durations(dur, speed))
        F0, N = ts.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, ts.style)
        torch.manual_seed(seed)
        torch.randn_like = grab
        try:
            s = ds(asr_tok @ aln, F0, N).squeeze(1)
        finally:
            torch.randn_like = orig
        out = R.Lock()(s, cap["src"].squeeze(-1), F0)
    hook.remove()
    W = ds.generator.m_source.l_linear.weight
    z = (cap["n"] @ W.T / W.norm()).transpose(1, 2).detach()  # [1,1,S]
    return out, dict(d=d, dur=dur, asr_tok=asr_tok, aln=aln, en=d.transpose(-1, -2) @ aln, asr=asr_tok @ aln, z=z)


def phonemes_to_ids(vocab, ps):
    return torch.LongTensor([[0] + [vocab[c] for c in ps if c in vocab][:510] + [0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("build"))
    ap.add_argument("--max-frames", type=int, default=4000, help="max aligned frames F (600F samples; 4000 = 100 s)")
    A = ap.parse_args()
    A.out.mkdir(parents=True, exist_ok=True)

    import json

    vocab = json.load(open(R.HERE / "weights/config.json"))["vocab"]
    ts, ds = R.load_halves()
    ps = "pəɹˈædi ɪz ɐ smˈɔl vˈYs ðæt ɹˈʌnz ˈɛniwˌɛɹ."
    ids = phonemes_to_ids(vocab, ps)
    ref, mid = reference(ts, ds, ids)
    Fa = mid["aln"].shape[2]
    print(f"T={ids.shape[1]} F={Fa} samples={ref.shape[1]}")

    tm, am = TextModel(ts).eval(), AcousticModel(ts, ds).eval()
    with torch.no_grad():
        dur, dmat, asr_tok = tm(ids)
        print(f"text wrapper vs ref: dur {(dur[0] - mid['dur']).abs().max():.2e}  d {(dmat - mid['d'].transpose(-1, -2)).abs().max():.2e}"
              f"  asr {(asr_tok - mid['asr_tok']).abs().max():.2e}")
        wav = am(mid["en"], mid["asr"], mid["z"])
        print(f"acoustic wrapper vs ref: max|diff| {(wav - ref).abs().max():.2e} (peak {ref.abs().max():.3f})")

    T = ct.RangeDim(lower_bound=3, upper_bound=512, default=ids.shape[1])
    Fd = ct.RangeDim(lower_bound=1, upper_bound=A.max_frames, default=Fa)
    Sd = ct.RangeDim(lower_bound=UP * 2, upper_bound=A.max_frames * 2 * UP, default=Fa * 2 * UP)

    with torch.no_grad():
        t_tr = torch.jit.trace(tm, ids.int(), strict=False)
        a_tr = torch.jit.trace(am, (mid["en"], mid["asr"], mid["z"]), strict=False)
    text = ct.convert(
        t_tr,
        inputs=[ct.TensorType(name="input_ids", shape=(1, T), dtype=np.int32)],
        outputs=[ct.TensorType(name="duration"), ct.TensorType(name="d"), ct.TensorType(name="asr_tok")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    text.short_description = "Paradee-8M v1.0 text side: phoneme ids -> durations, prosody features, asr features"
    text.save(str(A.out / "ParadeeText.mlpackage"))
    acou = ct.convert(
        a_tr,
        inputs=[
            ct.TensorType(name="en", shape=(1, dmat.shape[1], Fd), dtype=np.float32),
            ct.TensorType(name="asr", shape=(1, 512, Fd), dtype=np.float32),
            ct.TensorType(name="noise", shape=(1, 1, Sd), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="audio")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    acou.short_description = "Paradee-8M v1.0 acoustic side: aligned features + source noise -> 24 kHz audio"
    acou.save(str(A.out / "ParadeeAcoustic.mlpackage"))

    np.savez(A.out / "ref.npz", input_ids=ids.numpy().astype(np.int32), duration=mid["dur"].numpy()[None],
             d=mid["d"].transpose(-1, -2).numpy(), asr_tok=mid["asr_tok"].numpy(), en=mid["en"].numpy(),
             asr=mid["asr"].numpy(), noise=mid["z"].numpy(), audio=ref.numpy())
    print(f"saved {A.out}/ParadeeText.mlpackage, ParadeeAcoustic.mlpackage, ref.npz")

if __name__ == "__main__":
    main()
