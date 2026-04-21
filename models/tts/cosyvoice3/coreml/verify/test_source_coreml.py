"""Convert only the mel -> source (f0 + upsample + SineGen) path and test parity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

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


class SourceOnly(nn.Module):
    """mel -> source `s` (the precomputed sine source fed to decode)."""
    def __init__(self, w: HiFTCoreML):
        super().__init__()
        self.w = w

    def forward(self, mel):
        w = self.w
        f0 = w.gen.f0_predictor(mel, finalize=True)
        s = w.gen.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = w.gen.m_source(s)
        s = s.transpose(1, 2)
        return s, f0


def main():
    T_mel = 250
    torch.manual_seed(42)
    raw = torch.randn(1, 80, T_mel) * 0.3
    mel = torch.cumsum(raw, dim=-1) * 0.05 + torch.linspace(-2.0, -6.0, 80).view(1, 80, 1)

    gen = build_gen()
    wrapper = HiFTCoreML(gen).eval()
    src = SourceOnly(wrapper).eval()

    with torch.no_grad():
        s_t, f0_t = src(mel)
    print(f"torch s: shape={tuple(s_t.shape)} range=[{s_t.min():.3f}, {s_t.max():.3f}]")
    print(f"torch f0: shape={tuple(f0_t.shape)} range=[{f0_t.min():.3f}, {f0_t.max():.3f}]")

    traced = torch.jit.trace(src, mel, strict=False)
    with torch.no_grad():
        s_tr, f0_tr = traced(mel)
    d_s = (s_t - s_tr).abs()
    print(f"eager vs traced s: MAE={d_s.mean():.3e} max={d_s.max():.3e}")

    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=mel.shape, dtype=np.float32)],
        outputs=[
            ct.TensorType(name="s", dtype=np.float32),
            ct.TensorType(name="f0", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    out_dir = Path(__file__).parent.parent / "build"
    mlp = out_dir / "source-only.mlpackage"
    ml.save(str(mlp))

    ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({"mel": mel.numpy()})
    s_m = out["s"]
    f0_m = out["f0"]

    # f0 parity
    df = np.abs(f0_t.numpy() - f0_m)
    print(f"f0: MAE={df.mean():.3e} max={df.max():.3e}")

    # s parity per region (s is shape (1, 1, 120000))
    ds = np.abs(s_t.numpy() - s_m)
    print(f"s overall: MAE={ds.mean():.3e} max={ds.max():.3e}")
    L = ds.shape[-1]
    N = L // 10
    for lbl, start in [("start", 0), ("mid", L // 2), ("end", L - N)]:
        chunk = ds[..., start:start + N]
        print(f"  s {lbl} [{start}:{start+N}] MAE={chunk.mean():.3e} max={chunk.max():.3e}")


if __name__ == "__main__":
    main()
