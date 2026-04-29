"""Export Package C: predictor.F0Ntrain — F0 + energy prediction over mel time.

Inputs:
  en: (1, 640, T_mel)
  s:  (1, 128)

Outputs:
  F0: (1, T_mel)
  N:  (1, T_mel)

Compute units: ANE-eligible (AdainResBlk1d stack + bidirectional LSTM with
fixed seq dim). T_mel bucketed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _styletts2_lib import (  # noqa: E402
    COREML_DIR,
    DEFAULT_CHECKPOINT,
    F0NEnergyTraceable,
    LibriTTSConfig,
    load_inference_modules,
)

DEFAULT_MEL_BUCKETS = (256, 512, 1024, 2048, 4096)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out", type=Path, default=COREML_DIR / "styletts2_f0n_energy.mlpackage"
    )
    parser.add_argument("--mel-buckets", type=int, nargs="+", default=list(DEFAULT_MEL_BUCKETS))
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("[03] loading inference modules …")
    modules, cfg = load_inference_modules(args.checkpoint)

    print("[03] building F0NEnergyTraceable …")
    wrapper = F0NEnergyTraceable(modules).eval()

    # F0Ntrain.shared LSTM expects hidden_dim + style_dim (= 640) channels:
    # upstream feeds it `d.transpose(-1,-2) @ alignment` where d is the
    # DurationEncoder output (channel dim = h+s).
    en_channels = cfg.hidden_dim + cfg.style_dim
    largest = max(args.mel_buckets)
    example_en = torch.zeros(1, en_channels, largest, dtype=torch.float32)
    example_s = torch.zeros(1, cfg.style_dim, dtype=torch.float32)

    print(f"[03] sanity forward at T_mel={largest} …")
    with torch.no_grad():
        F0, N = wrapper(example_en, example_s)
    print(f"[03]   F0: {tuple(F0.shape)}  N: {tuple(N.shape)}")

    print("[03] tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_en, example_s), strict=False)

    if args.trace_only:
        print("[03] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct

    print(f"[03] converting with mel buckets {args.mel_buckets} …")
    en_enum = ct.EnumeratedShapes(
        shapes=[(1, en_channels, b) for b in sorted(args.mel_buckets)],
        default=(1, en_channels, largest),
    )
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="en", shape=en_enum, dtype=np.float32),
            ct.TensorType(name="s", shape=(1, cfg.style_dim), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="F0"), ct.TensorType(name="N")],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    mlmodel.short_description = "StyleTTS2 F0Ntrain (LibriTTS). ANE-eligible."
    mlmodel.save(str(args.out))
    print(f"[03] saved → {args.out}")


if __name__ == "__main__":
    main()
