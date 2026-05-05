"""Stage 6/7 — Noise (SineGen alone, fp32).

ComputeUnit.ALL, fp32 + int8pal kmeans. Phase precision in fp16 saturates on
long sequences, so this graph stays fp32 — the only fp32 graph in the pipeline.

The SineGen `_f02sine` constant-fold patch (`install_sinegen_v2_constfold_fix`)
must be re-applied per T_a because `fracs` is precomputed per bucket. We
export at MAX_T_A for a single graph; the host pads F0_curve to MAX_T_A.

Input:
  F0_curve [1, MAX_T_A * 2]    fp32  (Prosody output T_a*2 — predictor.F0 upsamples)

Outputs:
  sine_waves [1, MAX_T_A * 2 * UPSAMPLE_SCALE, harm+1]  fp32
  uv         [1, MAX_T_A * 2, 1]                        fp32

`harm+1` = `harmonic_num + 1` per upstream HiFi-GAN (default = 8 in the
LibriTTS config).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _styletts2_ane_lib import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MAX_T_A,
    NoiseTraceable,
    install_sinegen_v2_constfold_fix,
    load_modules_for_ane,
    palettize_int8,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out-dir", type=Path,
        default=THIS_DIR.parent.parent / "coreml" / "build" / "ane",
    )
    parser.add_argument("--no-palettize", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--max-T-a", type=int, default=MAX_T_A)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[06-ane] loading modules …")
    modules, _cfg = load_modules_for_ane(args.checkpoint)

    print(f"[06-ane] applying SineGen const-fold for T_a={args.max_T_a} …")
    install_sinegen_v2_constfold_fix(t_mel=args.max_T_a)

    wrapper = NoiseTraceable(modules["decoder"]).eval()

    # Prosody emits F0 at T_a*2 (predictor.F0[0] has upsample=True).
    F0_e = torch.zeros(1, args.max_T_a * 2, dtype=torch.float32)

    print(f"[06-ane] sanity forward (static T_a={args.max_T_a}) …")
    with torch.no_grad():
        sine_waves, uv = wrapper(F0_e)
    print(f"[06-ane]   sine_waves: {tuple(sine_waves.shape)} {sine_waves.dtype}")
    print(f"[06-ane]   uv:         {tuple(uv.shape)} {uv.dtype}")

    print("[06-ane] tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (F0_e,), strict=False)

    if args.trace_only:
        print("[06-ane] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct

    out_path = args.out_dir / "styletts2_ane_noise.mlpackage"
    print(f"[06-ane] converting → {out_path.name} (ALL, fp32, STATIC) …", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="F0_curve", shape=(1, args.max_T_a * 2), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="sine_waves"),
            ct.TensorType(name="uv"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )
    mlmodel.short_description = (
        f"StyleTTS2-ANE Stage 6: SineGen (static T_a={args.max_T_a}) — fp32, ALL."
    )
    mlmodel.save(str(out_path))
    print(f"[06-ane]   saved → {out_path}")

    if not args.no_palettize:
        print("[06-ane] palettizing …", flush=True)
        palettize_int8(out_path)
        print(f"[06-ane]   palettized → {out_path}")


if __name__ == "__main__":
    main()
