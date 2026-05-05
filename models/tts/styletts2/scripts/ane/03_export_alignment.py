"""Stage 3/7 — Alignment (cumsum + broadcast).

ANE-resident, fp16 + int8pal kmeans. RangeDim(2..MAX_T_TOK) on T_tok.

Inputs:
  pred_dur [1, T_tok]   fp32   (already sigmoid+sum+round in Swift)
  d        [1, T_tok, 640] fp32
  t_en     [1, 512, T_tok] fp32

Outputs:
  en       [1, 640, MAX_T_A] fp32
  asr      [1, 512, MAX_T_A] fp32

Mirrors Kokoro-ANE's standalone Alignment graph. The host slices outputs to
the actual T_a (= sum(pred_dur)) before passing to Prosody / Vocoder.
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
    AlignmentTraceable,
    DEFAULT_CHECKPOINT,
    MAX_T_A,
    MAX_T_TOK,
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
    parser.add_argument("--example-T-tok", type=int, default=64)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # We don't really need modules here, but loading once keeps the script
    # skeleton uniform.
    print("[03-ane] loading modules (for cfg) …")
    _modules, cfg = load_modules_for_ane(args.checkpoint)

    wrapper = AlignmentTraceable(max_T_a=args.max_T_a).eval()

    h_plus_s = cfg.hidden_dim + cfg.style_dim
    T = args.example_T_tok

    pred_dur_e = torch.ones(1, T, dtype=torch.float32)  # all-1 valid example
    d_e = torch.zeros(1, T, h_plus_s, dtype=torch.float32)
    t_en_e = torch.zeros(1, cfg.hidden_dim, T, dtype=torch.float32)

    print(f"[03-ane] sanity forward T_tok={T} max_T_a={args.max_T_a} …")
    with torch.no_grad():
        en, asr = wrapper(pred_dur_e, d_e, t_en_e)
    print(f"[03-ane]   en:  {tuple(en.shape)} {en.dtype}")
    print(f"[03-ane]   asr: {tuple(asr.shape)} {asr.dtype}")

    print("[03-ane] tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (pred_dur_e, d_e, t_en_e), strict=False)

    if args.trace_only:
        print("[03-ane] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct

    out_path = args.out_dir / "styletts2_ane_alignment.mlpackage"
    print(f"[03-ane] converting → {out_path.name} (cpu_and_ne, fp16, RangeDim) …", flush=True)
    rd = ct.RangeDim(lower_bound=2, upper_bound=MAX_T_TOK, default=T)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="pred_dur", shape=(1, rd), dtype=np.float32),
            ct.TensorType(name="d", shape=(1, rd, h_plus_s), dtype=np.float32),
            ct.TensorType(name="t_en", shape=(1, cfg.hidden_dim, rd), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="en"),
            ct.TensorType(name="asr"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    mlmodel.short_description = (
        f"StyleTTS2-ANE Stage 3: Alignment (cumsum+broadcast, T_a≤{args.max_T_a}) — fp16, ANE."
    )
    mlmodel.save(str(out_path))
    print(f"[03-ane]   saved → {out_path}")

    if not args.no_palettize:
        print("[03-ane] palettizing …", flush=True)
        palettize_int8(out_path)
        print(f"[03-ane]   palettized → {out_path}")


if __name__ == "__main__":
    main()
