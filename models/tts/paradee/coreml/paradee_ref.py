"""PyTorch reference for Paradee-8M v1.0, built from the upstream training code (upstream/training/scripts)
exactly as export_paradee.py assembles it for the released ONNX graph."""
import pathlib
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream/training/scripts"))

from student import TextStudent  # noqa: E402
from student_decoder import load_student_decoder  # noqa: E402
from torch.nn.utils import remove_weight_norm  # noqa: E402
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations  # noqa: E402

N_FFT, HOP, SR = 1024, 256, 24000
FR = torch.arange(N_FFT // 2 + 1) * SR / N_FFT
BINS = torch.nonzero((FR >= 2000) & (FR < 8000)).squeeze(1)  # 2-8 kHz: bins 86..341
LOCK_L = 33


def strip(m):
    for x in m.modules():
        try:
            remove_weight_norm(x)
        except Exception:
            pass
        if is_parametrized(x, "weight"):
            try:
                remove_parametrizations(x, "weight", leave_parametrized=True)
            except Exception:
                pass


class Lock(nn.Module):
    """export_paradee.Lock, verbatim math."""

    def __init__(self, L=LOCK_L):
        super().__init__()
        w = torch.hann_window(N_FFT)
        n = torch.arange(N_FFT).float()
        k = BINS.float()[:, None]
        ang = 2 * torch.pi * k * n / N_FFT
        self.register_buffer("fwd", torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w])[:, None])
        self.register_buffer("inv", (torch.cat([torch.cos(ang), -torch.sin(ang)]) * (2 / N_FFT) * w)[:, None])
        self.register_buffer("w2", (w**2)[None, None])
        sk = torch.hann_window(L + 2)[1:-1]
        self.register_buffer("sk", (sk / sk.sum())[None, None])
        self.pad = L // 2
        self.K = len(BINS)

    def stft(self, x):
        y = F.conv1d(F.pad(x[:, None], (N_FFT // 2, N_FFT // 2), mode="reflect"), self.fwd, stride=HOP)[0]
        return y[: self.K], y[self.K :]

    def forward(self, s, src, f0):
        k = min(s.shape[1], src.shape[1])
        s = s[:, :k]
        src = src[:, :k]
        Sr, Si = self.stft(s)
        Er, Ei = self.stft(src)
        nf = Sr.shape[1]
        j = torch.clamp(torch.arange(nf) * HOP // 300, max=f0.shape[1] - 1)
        v = (f0[0, j] > 60).float()[None]
        Em = torch.sqrt(Er**2 + Ei**2).clamp_min(1e-6)
        Rr, Ri = (Sr * Er + Si * Ei) / Em, (Si * Er - Sr * Ei) / Em
        sm = lambda z: F.conv1d(z[:, None], self.sk, padding=self.pad)[:, 0]  # noqa: E731
        Rr, Ri = sm(Rr), sm(Ri)
        Rm = torch.sqrt(Rr**2 + Ri**2).clamp_min(1e-12)
        Sm = torch.sqrt(Sr**2 + Si**2)
        ur, ui = (Er * Rr - Ei * Ri) / (Em * Rm), (Er * Ri + Ei * Rr) / (Em * Rm)
        Dr, Di = (Sm * ur - Sr) * v, (Sm * ui - Si) * v
        ola = F.conv_transpose1d(torch.cat([Dr, Di])[None], self.inv, stride=HOP)[0]
        env = F.conv_transpose1d(torch.ones(1, 1, nf), self.w2, stride=HOP)[0]
        d = (ola / env.clamp_min(1e-8))[:, N_FFT // 2 : N_FFT // 2 + k]
        return s + d


class ConvSTFT(nn.Module):
    """export_paradee.ConvSTFT, verbatim math (matches torch.stft/istft the decoder was trained with)."""

    def __init__(self, n, hop):
        super().__init__()
        self.n, self.hop = n, hop
        w = torch.hann_window(n)
        k = torch.arange(n // 2 + 1).float()[:, None]
        ang = 2 * torch.pi * k * torch.arange(n).float() / n
        re, im = torch.cos(ang) * w, -torch.sin(ang) * w
        im[0] = 0
        im[-1] = 0
        self.register_buffer("fwd", torch.cat([re, im])[:, None])
        c = torch.full((n // 2 + 1, 1), 2.0)
        c[0] = c[-1] = 1
        self.register_buffer("inv", (torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w)[:, None])
        self.register_buffer("w2", (w**2)[None, None])

    def transform(self, x):
        y = F.conv1d(F.pad(x[:, None], (self.n // 2, self.n // 2), mode="reflect"), self.fwd, stride=self.hop)
        K = self.n // 2 + 1
        re, im = y[:, :K], y[:, K:]
        ph = torch.where((im == 0) & (re < 0), torch.full_like(re, torch.pi), torch.atan2(im, re))
        return torch.sqrt(re**2 + im**2), ph

    def inverse(self, mag, phase):
        z = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], 1)
        y = F.conv_transpose1d(z, self.inv, stride=self.hop)[:, 0]
        env = F.conv_transpose1d(torch.ones(1, 1, mag.shape[-1]), self.w2, stride=self.hop)[:, 0]
        return (y / env.clamp_min(1e-8))[:, self.n // 2 : -(self.n // 2)][:, None]


def load_halves(weights=HERE / "weights/pytorch"):
    ts = TextStudent("s+mlp").eval()
    ts.load_state_dict(torch.load(weights / "text_side.pt", map_location="cpu")["model"])
    ds = load_student_decoder("A", str(weights / "decoder.pt"), disable_complex=True)
    strip(ts)
    strip(ds)
    ds.generator.stft = ConvSTFT(ds.generator.stft.filter_length, ds.generator.stft.hop_length)
    return ts.eval(), ds.eval()


def text_forward(ts, ids):
    """export_paradee.Paradee.text: ids [1,T] -> d [1,T,C], dur [T] (pre-round), asr_tok [1,512,T]."""
    p = ts.predictor
    s = ts.style
    d_en = ts.bert_encoder(ts.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
    x = d_en
    T = ids.shape[1]
    sx = s[:, :, None].expand(-1, -1, T)
    x = torch.cat([x, sx], 1)
    for b in p.text_encoder.lstms:
        if isinstance(b, nn.LSTM):
            x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
        else:
            x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
    d = x.transpose(-1, -2)
    dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]
    e = ts.text_encoder
    h = e.embedding(ids).transpose(1, 2)
    for c in e.cnn:
        h = c(h)
    t_en = e.lstm(h.transpose(1, 2))[0]
    asr_tok = ts.asr_proj(t_en).transpose(-1, -2)
    return d, dur, asr_tok


def durations(dur, speed=1.0):
    return torch.clamp(torch.round(dur / speed), min=1).long()


def alignment(n):
    end = torch.cumsum(n, 0)
    start = end - n
    f = torch.arange(int(end[-1]))[None]
    return ((f >= start[:, None]) & (f < end[:, None])).float()[None]  # [1,T,F]
