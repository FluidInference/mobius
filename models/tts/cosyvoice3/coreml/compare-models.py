"""Validate CoreML mlpackage HiFT output matches PyTorch reference.

Loads the mlpackage saved by `convert-coreml.py`, runs prediction with the
stored reference input, and reports MAE / max / correlation vs the PyTorch
reference output (both stored in `ref-T{N}.pt`).

Usage:
    uv run python compare-models.py \
        --mlpackage build/hift-fp32/HiFT-T250-fp32.mlpackage \
        --reference build/hift-fp32/ref-T250.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mlpackage", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument(
        "--compute-units",
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    p.add_argument("--runs", type=int, default=3, help="Timed runs after 2 warmups")
    p.add_argument("--save-audio", help="Optional path to save wrapper+coreml audio as .npz")
    args = p.parse_args()

    cu = getattr(ct.ComputeUnit, args.compute_units)
    print(f"Loading {args.mlpackage} with compute_units={args.compute_units}...")
    mlmodel = ct.models.MLModel(args.mlpackage, compute_units=cu)

    ref = torch.load(args.reference, map_location="cpu", weights_only=False)
    mel = ref["mel"].numpy()  # (1, 80, T)
    audio_torch = ref["audio"].numpy()

    print(f"mel shape: {mel.shape}   audio ref shape: {audio_torch.shape}")

    # Warmup
    for _ in range(2):
        out = mlmodel.predict({"mel": mel})
    audio_key = list(out.keys())[0]
    audio_ml = out[audio_key]
    print(f"audio coreml shape: {audio_ml.shape}")

    # Timed runs
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        out = mlmodel.predict({"mel": mel})
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"latency: min={min(times):.1f} ms  median={sorted(times)[len(times)//2]:.1f} ms  mean={sum(times)/len(times):.1f} ms")

    # Parity
    a_t = audio_torch.flatten()
    a_m = audio_ml.flatten()
    L = min(a_t.size, a_m.size)
    a_t, a_m = a_t[:L], a_m[:L]
    diff = np.abs(a_t - a_m)
    corr = np.corrcoef(a_t, a_m)[0, 1]
    print(f"audio MAE: {diff.mean():.3e}  max: {diff.max():.3e}  corr: {corr:.6f}")
    print(f"  torch range: [{a_t.min():.3f}, {a_t.max():.3f}]")
    print(f"  coreml range: [{a_m.min():.3f}, {a_m.max():.3f}]")

    if args.save_audio:
        np.savez(args.save_audio, torch=a_t, coreml=a_m)
        print(f"saved audio to {args.save_audio}")


if __name__ == "__main__":
    main()
