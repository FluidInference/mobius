"""Stage 1/7 — PLBert (Albert) → CoreML mlpackage.

ANE-resident, fp16 + int8pal kmeans. RangeDim(2..MAX_T_TOK) on tokens.

Input:  tokens [1, T_tok] int32
Output: bert_dur [1, T_tok, 768]

Bucketing: single mlpackage with `RangeDim(2..MAX_T_TOK)` on the T_tok axis —
matches Kokoro-ANE's PLBERT graph.
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
    MAX_T_TOK,
    PLBertTraceable,
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
    parser.add_argument("--example-T", type=int, default=64)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[01-ane] loading modules …")
    modules, _cfg = load_modules_for_ane(args.checkpoint)
    wrapper = PLBertTraceable(modules).eval()

    example_tokens = torch.zeros(1, args.example_T, dtype=torch.long)
    print(f"[01-ane] sanity forward T_tok={args.example_T} …")
    with torch.no_grad():
        bert_dur = wrapper(example_tokens)
    print(f"[01-ane]   bert_dur: {tuple(bert_dur.shape)} {bert_dur.dtype}")

    print("[01-ane] tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_tokens,), strict=False)

    if args.trace_only:
        print("[01-ane] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct  # local heavy import

    out_path = args.out_dir / "styletts2_ane_plbert.mlpackage"
    print(f"[01-ane] converting → {out_path.name} (cpu_and_ne, fp16, RangeDim) …", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="tokens",
                shape=(1, ct.RangeDim(lower_bound=2, upper_bound=MAX_T_TOK, default=args.example_T)),
                dtype=np.int32,
            ),
        ],
        outputs=[ct.TensorType(name="bert_dur")],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    mlmodel.short_description = "StyleTTS2-ANE Stage 1: PLBERT (Albert) — fp16, ANE."
    mlmodel.save(str(out_path))
    print(f"[01-ane]   saved → {out_path}")

    if not args.no_palettize:
        print("[01-ane] palettizing (kmeans, nbits=8) …", flush=True)
        palettize_int8(out_path)
        print(f"[01-ane]   palettized → {out_path}")


if __name__ == "__main__":
    main()
