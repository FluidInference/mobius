"""Find where PyTorch-vs-CoreML divergence begins by exposing intermediate outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hift_coreml import HiFTCoreML


def build_gen():
    from hyperpyyaml import load_hyperpyyaml
    here = Path(__file__).parent.parent
    with open(here / "cosyvoice3_dl" / "cosyvoice3.yaml", "r") as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    gen = cfg["hift"]
    sd = torch.load(str(here / "cosyvoice3_dl" / "hift.pt"), map_location="cpu", weights_only=False)
    gen.load_state_dict(sd, strict=False)
    gen.eval()
    return gen


class IntermediateExporter(nn.Module):
    """Wraps HiFTCoreML but returns pre-ISTFT real/imag instead of audio."""
    def __init__(self, w: HiFTCoreML):
        super().__init__()
        self.w = w

    def forward(self, mel):
        w = self.w
        f0 = w.gen.f0_predictor(mel, finalize=True)
        s = w.gen.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = w.gen.m_source(s)
        s = s.transpose(1, 2)
        s_real, s_imag = w.stft(s.squeeze(1))
        x = w.gen.conv_pre(mel)
        s_stft = torch.cat([s_real, s_imag], dim=1)
        for i in range(w.num_upsamples):
            x = F.leaky_relu(x, w.lrelu_slope)
            x = w.gen.ups[i](x)
            if i == w.num_upsamples - 1:
                x = w.gen.reflection_pad(x)
            si = w.gen.source_downs[i](s_stft)
            si = w.gen.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(w.num_kernels):
                r = w.gen.resblocks[i * w.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / w.num_kernels
        x = F.leaky_relu(x)
        x = w.gen.conv_post(x)
        n_bins = w.n_fft // 2 + 1
        magnitude = torch.exp(x[:, :n_bins, :])
        phase = torch.sin(x[:, n_bins:, :])
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return real, imag


def make_structured_mel(T):
    torch.manual_seed(42)
    raw = torch.randn(1, 80, T) * 0.3
    smooth = torch.cumsum(raw, dim=-1) * 0.05
    bias = torch.linspace(-2.0, -6.0, 80).view(1, 80, 1)
    return smooth + bias


def main():
    T = 250
    mel = make_structured_mel(T)

    gen = build_gen()
    wrapper = HiFTCoreML(gen).eval()
    exporter = IntermediateExporter(wrapper).eval()

    with torch.no_grad():
        real_t, imag_t = exporter(mel)
    print(f"pre-ISTFT real: shape={tuple(real_t.shape)} range=[{real_t.min().item():.3f}, {real_t.max().item():.3f}]")

    traced = torch.jit.trace(exporter, mel, strict=False)

    # Check traced model still matches eager-pytorch
    with torch.no_grad():
        real_traced, imag_traced = traced(mel)
    d_real = (real_t - real_traced).abs()
    print(f"eager vs traced (real): MAE={d_real.mean().item():.3e} max={d_real.max().item():.3e}")
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=mel.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="real", dtype=np.float32), ct.TensorType(name="imag", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    out_dir = Path(__file__).parent.parent / "build"
    mlp = out_dir / "intermediate.mlpackage"
    ml.save(str(mlp))

    ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({"mel": mel.numpy()})
    real_m = out["real"]
    imag_m = out["imag"]

    def stats(name, t, m):
        d = np.abs(t - m)
        print(f"{name}: MAE={d.mean():.3e} max={d.max():.3e}")
        # Per-frame column
        per_col = d.mean(axis=(0, 1))
        T_ = per_col.shape[0]
        for lbl, start in [("start", 0), ("mid", T_ // 2), ("end", T_ - T_ // 10)]:
            chunk = per_col[start : start + T_ // 10]
            print(f"  {lbl} cols [{start}:{start + T_ // 10}] MAE: {chunk.mean():.3e} max: {chunk.max():.3e}")

    stats("real", real_t.numpy(), real_m)
    stats("imag", imag_t.numpy(), imag_m)


if __name__ == "__main__":
    main()
