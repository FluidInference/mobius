"""Stage 2/7 — PostBert (TextEncoder + DurationEncoder + duration head).

ANE-resident, fp16 + int8pal kmeans. RangeDim(2..MAX_T_TOK).

Inputs:
  bert_dur [1, T_tok, 768]
  tokens   [1, T_tok] int32
  style    [1, 256]   fp32   (full ref_s = [acoustic | prosody])

Outputs:
  t_en             [1, 512, T_tok]
  d                [1, T_tok, 640]
  pred_dur_log     [1, T_tok, 50]
  fixed_embedding  [1, T_tok, 768]

The BiLSTM unroll inside `_DurationEncoderUnrolled` follows Kokoro-ANE.
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
    PostBertTraceable,
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
    parser.add_argument("--fp32", action="store_true",
                        help="Diagnostic: keep compute_precision=FLOAT32 to isolate fp16 BiLSTM loss.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[02-ane] loading modules …")
    modules, cfg = load_modules_for_ane(args.checkpoint)
    wrapper = PostBertTraceable(modules, cfg).eval()

    bert_hidden = modules["bert"].config.hidden_size
    style_dim_2x = cfg.style_dim * 2
    T = args.example_T

    bert_dur_e = torch.zeros(1, T, bert_hidden, dtype=torch.float32)
    tokens_e = torch.zeros(1, T, dtype=torch.long)
    style_e = torch.zeros(1, style_dim_2x, dtype=torch.float32)

    print(f"[02-ane] sanity forward T_tok={T} …")
    with torch.no_grad():
        outs = wrapper(bert_dur_e, tokens_e, style_e)
    for name, t in zip(("t_en", "d", "pred_dur_log", "fixed_embedding"), outs):
        print(f"[02-ane]   {name}: {tuple(t.shape)} {t.dtype}")

    print("[02-ane] tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (bert_dur_e, tokens_e, style_e), strict=False)

    if args.trace_only:
        print("[02-ane] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct

    out_path = args.out_dir / "styletts2_ane_postbert.mlpackage"
    print(f"[02-ane] converting → {out_path.name} (cpu_and_ne, fp16, RangeDim) …", flush=True)
    rd = ct.RangeDim(lower_bound=2, upper_bound=MAX_T_TOK, default=T)
    # Match laishere Kokoro convert-coreml.py:674-682 exactly: fp16 input dtypes
    # and ComputeUnit.ALL. Empirically the CPU_AND_NE + fp32-input combo
    # collapses the duration encoder's style-cat tile to 0 reps under RangeDim.
    in_dtype = np.float32 if args.fp32 else np.float16
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="bert_dur", shape=(1, rd, bert_hidden), dtype=in_dtype),
            ct.TensorType(name="tokens", shape=(1, rd), dtype=np.int32),
            ct.TensorType(name="style", shape=(1, style_dim_2x), dtype=in_dtype),
        ],
        outputs=[
            ct.TensorType(name="t_en"),
            ct.TensorType(name="d"),
            ct.TensorType(name="pred_dur_log"),
            ct.TensorType(name="fixed_embedding"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16,
        skip_model_load=True,
    )
    mlmodel.short_description = (
        "StyleTTS2-ANE Stage 2: TextEncoder + DurationEncoder + duration head — fp16, ANE."
    )
    mlmodel.save(str(out_path))
    print(f"[02-ane]   saved → {out_path}")

    if not args.no_palettize:
        print("[02-ane] palettizing …", flush=True)
        palettize_int8(out_path)
        print(f"[02-ane]   palettized → {out_path}")


if __name__ == "__main__":
    main()
