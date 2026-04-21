"""Convert only the decoder (taking mel + precomputed source) and test parity."""
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


class DecodeOnly(nn.Module):
    def __init__(self, w: HiFTCoreML):
        super().__init__()
        self.w = w

    def forward(self, mel, s):
        return self.w._decode(mel, s)


def main():
    T_mel = 250
    torch.manual_seed(42)
    raw = torch.randn(1, 80, T_mel) * 0.3
    mel = torch.cumsum(raw, dim=-1) * 0.05 + torch.linspace(-2.0, -6.0, 80).view(1, 80, 1)

    gen = build_gen()
    wrapper = HiFTCoreML(gen).eval()

    with torch.no_grad():
        # Precompute source via wrapper (so it matches expected distribution)
        f0 = wrapper.gen.f0_predictor(mel, finalize=True)
        sraw = wrapper.gen.f0_upsamp(f0[:, None]).transpose(1, 2)
        sraw, _, _ = wrapper.gen.m_source(sraw)
        s = sraw.transpose(1, 2)  # (1, 1, 120000)

    decode_only = DecodeOnly(wrapper).eval()
    with torch.no_grad():
        audio_t = decode_only(mel, s)

    traced = torch.jit.trace(decode_only, (mel, s), strict=False)
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=mel.shape, dtype=np.float32),
            ct.TensorType(name="s", shape=s.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="audio", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    out_dir = Path(__file__).parent.parent / "build"
    mlp = out_dir / "decode-only.mlpackage"
    ml.save(str(mlp))

    ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({"mel": mel.numpy(), "s": s.numpy()})
    audio_m = out[list(out.keys())[0]]

    d = np.abs(audio_t.numpy() - audio_m)
    print(f"decode-only MAE={d.mean():.3e} max={d.max():.3e}")
    L = d.shape[-1]
    for lbl, start in [("start", 0), ("mid", L // 2), ("end", L - L // 10)]:
        chunk = d[..., start:start + L // 10]
        print(f"  {lbl}: MAE={chunk.mean():.3e} max={chunk.max():.3e}")


if __name__ == "__main__":
    main()
