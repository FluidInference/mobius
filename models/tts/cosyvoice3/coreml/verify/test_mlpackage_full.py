"""Run the full mlpackage end-to-end and print audio-quality stats."""
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
from hyperpyyaml import load_hyperpyyaml


def build_gen():
    here = Path(__file__).parent.parent
    with open(here / "cosyvoice3_dl" / "cosyvoice3.yaml") as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    gen = cfg["hift"]
    sd = torch.load(str(here / "cosyvoice3_dl" / "hift.pt"), map_location="cpu", weights_only=False)
    gen.load_state_dict(sd, strict=False)
    gen.eval()
    return gen


def main():
    mlp_path = Path(__file__).parent.parent / "build" / "hift-fp32" / "HiFT-T250-fp32.mlpackage"
    if not mlp_path.exists():
        print(f"ERROR: {mlp_path} not found. Run convert-coreml.py first.")
        return

    ml = ct.models.MLModel(str(mlp_path), compute_units=ct.ComputeUnit.CPU_ONLY)

    gen = build_gen()
    wrapper = HiFTCoreML(gen).eval()

    torch.manual_seed(42)
    raw = torch.randn(1, 80, 250) * 0.3
    mel = torch.cumsum(raw, dim=-1) * 0.05 + torch.linspace(-2.0, -6.0, 80).view(1, 80, 1)

    with torch.no_grad():
        audio_t = wrapper(mel)
    a_t = audio_t.numpy().flatten()

    out = ml.predict({"mel": mel.numpy()})
    a_m = list(out.values())[0].flatten()

    L = min(a_t.size, a_m.size)
    a_t = a_t[:L]
    a_m = a_m[:L]
    print(f"audio length: {L} samples = {L / 24000:.2f} sec")
    print(f"torch:  range=[{a_t.min():.4f}, {a_t.max():.4f}]  std={a_t.std():.4f}  mean_abs={np.mean(np.abs(a_t)):.4f}")
    print(f"coreml: range=[{a_m.min():.4f}, {a_m.max():.4f}]  std={a_m.std():.4f}  mean_abs={np.mean(np.abs(a_m)):.4f}")
    d = np.abs(a_t - a_m)
    corr = np.corrcoef(a_t, a_m)[0, 1]
    print(f"overall: MAE={d.mean():.3e}  max={d.max():.3e}  corr={corr:.6f}")

    # Chunks
    N = L // 10
    for i in range(10):
        s, e = i * N, (i + 1) * N
        c = np.corrcoef(a_t[s:e], a_m[s:e])[0, 1]
        print(f"  [{s:>6d}:{e:>6d}] MAE={d[s:e].mean():.3e} max={d[s:e].max():.3e} "
              f"corr={c:.5f} t_std={a_t[s:e].std():.3e} m_std={a_m[s:e].std():.3e}")


if __name__ == "__main__":
    main()
