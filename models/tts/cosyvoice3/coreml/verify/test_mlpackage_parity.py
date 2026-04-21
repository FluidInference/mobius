"""Better parity test: structured mel input, unclamped comparison."""
import sys
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import coremltools as ct
import numpy as np
import torch

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


def make_structured_mel(T: int) -> torch.Tensor:
    """Create a smoother, lower-amplitude mel that won't saturate the vocoder."""
    torch.manual_seed(42)
    # Low-frequency smooth noise via cumsum
    raw = torch.randn(1, 80, T) * 0.3
    smooth = torch.cumsum(raw, dim=-1) * 0.05
    # Roughly mel-like: higher bins have lower energy, add baseline
    bias = torch.linspace(-2.0, -6.0, 80).view(1, 80, 1)
    return smooth + bias


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlpackage", default="build/hift-fp32/HiFT-T250-fp32.mlpackage")
    p.add_argument("--mel-frames", type=int, default=250)
    p.add_argument("--compute-units", default="CPU_ONLY", choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"])
    args = p.parse_args()

    T = args.mel_frames
    mel = make_structured_mel(T)

    print("Building wrapper...")
    gen = build_gen()
    wrapper = HiFTCoreML(gen).eval()

    print("PyTorch forward...")
    with torch.no_grad():
        audio_torch = wrapper(mel)
    print(f"  torch range: [{audio_torch.min().item():.4f}, {audio_torch.max().item():.4f}]")

    print(f"Loading mlpackage {args.mlpackage}...")
    cu = getattr(ct.ComputeUnit, args.compute_units)
    mlmodel = ct.models.MLModel(args.mlpackage, compute_units=cu)

    print("CoreML predict...")
    out = mlmodel.predict({"mel": mel.numpy()})
    audio_ml = out[list(out.keys())[0]]
    print(f"  coreml range: [{audio_ml.min():.4f}, {audio_ml.max():.4f}]")

    a_t = audio_torch.numpy().flatten()
    a_m = audio_ml.flatten()
    L = min(a_t.size, a_m.size)
    a_t, a_m = a_t[:L], a_m[:L]
    diff = np.abs(a_t - a_m)
    corr = np.corrcoef(a_t, a_m)[0, 1]
    print(f"\nMAE: {diff.mean():.3e}  max: {diff.max():.3e}  corr: {corr:.6f}")

    # Report worst 5 samples
    idx_worst = np.argsort(diff)[-5:]
    print("worst 5 samples:")
    for i in idx_worst:
        print(f"  idx={i}  torch={a_t[i]:.4f}  coreml={a_m[i]:.4f}  diff={diff[i]:.4e}")

    # Per-region MAE
    print("\nPer-region MAE (start, middle, end):")
    N = L // 10
    for start in [0, L // 2 - N // 2, L - N]:
        end = start + N
        d = diff[start:end]
        print(f"  samples [{start:>6d}, {end:>6d}]  MAE={d.mean():.3e}  max={d.max():.3e}")


if __name__ == "__main__":
    main()
