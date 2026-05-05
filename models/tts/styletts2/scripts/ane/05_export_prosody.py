"""Stage 5/7 — Prosody (F0Ntrain).

ANE-resident, fp16 + int8pal kmeans. **Single fixed shape** at MAX_T_A — this
is the fix for the E5RT FlexibleShapeInfo bug
("tensor_buffer has known strides while the model has FlexibleShapeInfo")
that pinned the legacy `f0n_energy` graph to CPU. The host pads `en` to
MAX_T_A on the time axis before calling.

Inputs:
  en  [1, 640, MAX_T_A]   fp32   (= d.transpose @ alignment, padded on host)
  s   [1, 128]            fp32   (prosody half of ref_s)

Outputs:
  F0  [1, MAX_T_A]        fp32
  N   [1, MAX_T_A]        fp32

The host slices F0/N to the actual T_a after the call.
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
    ProsodyTraceable,
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

    print("[05-ane] loading modules …")
    modules, cfg = load_modules_for_ane(args.checkpoint)
    wrapper = ProsodyTraceable(modules).eval()

    h_plus_s = cfg.hidden_dim + cfg.style_dim
    en_e = torch.zeros(1, h_plus_s, args.max_T_a, dtype=torch.float32)
    s_e = torch.zeros(1, cfg.style_dim, dtype=torch.float32)

    print(f"[05-ane] sanity forward (static T_a={args.max_T_a}) …")
    with torch.no_grad():
        F0, N = wrapper(en_e, s_e)
    print(f"[05-ane]   F0: {tuple(F0.shape)} {F0.dtype}")
    print(f"[05-ane]   N:  {tuple(N.shape)} {N.dtype}")

    print("[05-ane] tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (en_e, s_e), strict=False)

    if args.trace_only:
        print("[05-ane] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct

    out_path = args.out_dir / "styletts2_ane_prosody.mlpackage"
    print(f"[05-ane] converting → {out_path.name} (cpu_and_ne, fp16, STATIC) …", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="en", shape=(1, h_plus_s, args.max_T_a), dtype=np.float32),
            ct.TensorType(name="s", shape=(1, cfg.style_dim), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="F0"),
            ct.TensorType(name="N"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    mlmodel.short_description = (
        f"StyleTTS2-ANE Stage 5: Prosody F0Ntrain (static T_a={args.max_T_a}) — fp16, ANE."
    )
    mlmodel.save(str(out_path))
    print(f"[05-ane]   saved → {out_path}")

    if not args.no_palettize:
        print("[05-ane] palettizing …", flush=True)
        palettize_int8(out_path)
        print(f"[05-ane]   palettized → {out_path}")


if __name__ == "__main__":
    main()
