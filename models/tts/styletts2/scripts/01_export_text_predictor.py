"""Export Package A: text encoder + PL-BERT + bert_encoder + duration predictor.

Inputs:
  tokens: (1, T_tok) int32, T_tok ∈ buckets
  style:  (1, 128)   float32      (the `s` half of the predicted style)

Outputs:
  t_en:            (1, 512, T_tok)
  d_en:            (1, 512, T_tok)
  d:               (1, T_tok, 640)            (h + s)
  pred_dur_log:    (1, T_tok, max_dur=50)     pre-sigmoid duration logits
  fixed_embedding: (1, T_tok, 768)            for CFG uncond branch in Swift
  bert_dur:        (1, T_tok, 768)            cond branch input

Compute units: CPU + GPU (BiLSTM is not ANE-friendly).

Bucketing: a single mlpackage with EnumeratedShapes on `tokens` compiles but
mis-propagates symbolic shape: at runtime `d` collapses to (1, T, 512) instead
of (1, T, 640), and `fixed_embedding` collapses to (1, 0, 0). Mirroring
04_export_decoder.py / 02_export_diffusion_step.py, emit one fixed-shape
mlpackage per token bucket: `styletts2_text_predictor_<T_tok>.mlpackage`.
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
    LibriTTSConfig,
    TextPredictorTraceable,
    load_inference_modules,
)

DEFAULT_TOK_BUCKETS = (32, 64, 128, 256, 512)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out-dir", type=Path, default=COREML_DIR,
        help="Directory; per-bucket mlpackages are saved as styletts2_text_predictor_<T_tok>.mlpackage.",
    )
    parser.add_argument("--buckets", type=int, nargs="+", default=list(DEFAULT_TOK_BUCKETS))
    parser.add_argument("--trace-only", action="store_true", help="Run trace; skip CoreML conversion.")
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpu_and_gpu", "cpu_only"],
        default="cpu_and_gpu",
        help="BiLSTM rejects ANE placement; cpu_and_gpu is the default.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[01] loading inference modules …")
    modules, cfg = load_inference_modules(args.checkpoint)

    print("[01] building TextPredictorTraceable …")
    wrapper = TextPredictorTraceable(modules).eval()

    if args.trace_only:
        largest = max(args.buckets)
        example_tokens = torch.zeros(1, largest, dtype=torch.long)
        example_style = torch.zeros(1, cfg.style_dim, dtype=torch.float32)
        print(f"[01] sanity forward at T_tok={largest} …")
        with torch.no_grad():
            outs = wrapper(example_tokens, example_style)
        for name, t in zip(
            ("t_en", "d_en", "d", "pred_dur_log", "fixed_embedding", "bert_dur"), outs
        ):
            print(f"[01]   {name}: {tuple(t.shape)} {t.dtype}")
        print("[01] tracing (largest bucket only) …")
        with torch.no_grad():
            torch.jit.trace(wrapper, (example_tokens, example_style), strict=False)
        print("[01] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct  # local import — heavy

    cu = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }[args.compute_units]

    # See module docstring: EnumeratedShapes mis-propagates symbolic dims (d
    # output collapses to 512-d, fixed_embedding collapses to (1,0,0)). Emit
    # one fixed-shape mlpackage per token bucket; route at runtime.
    for t_tok in sorted(args.buckets):
        out_path = args.out_dir / f"styletts2_text_predictor_{t_tok}.mlpackage"
        print(f"[01] === T_tok={t_tok}  →  {out_path.name} ===")

        example_tokens = torch.zeros(1, t_tok, dtype=torch.long)
        example_style = torch.zeros(1, cfg.style_dim, dtype=torch.float32)

        with torch.no_grad():
            outs = wrapper(example_tokens, example_style)
        for name, t in zip(
            ("t_en", "d_en", "d", "pred_dur_log", "fixed_embedding", "bert_dur"), outs
        ):
            print(f"[01]   sanity {name}: {tuple(t.shape)} {t.dtype}")

        print("[01]   tracing …")
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (example_tokens, example_style), strict=False)

        print(f"[01]   converting (compute_units={args.compute_units}) …", flush=True)
        # skip_model_load avoids the post-convert MLModel instantiation which
        # otherwise hangs Apple's anecompilerservice for several minutes per
        # bucket. The mlpackage is saved correctly.
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="tokens", shape=(1, t_tok), dtype=np.int32),
                ct.TensorType(name="style", shape=(1, cfg.style_dim), dtype=np.float32),
            ],
            outputs=[
                ct.TensorType(name="t_en"),
                ct.TensorType(name="d_en"),
                ct.TensorType(name="d"),
                ct.TensorType(name="pred_dur_log"),
                ct.TensorType(name="fixed_embedding"),
                ct.TensorType(name="bert_dur"),
            ],
            minimum_deployment_target=ct.target.iOS17,
            compute_units=cu,
            convert_to="mlprogram",
            skip_model_load=True,
        )
        mlmodel.short_description = (
            f"StyleTTS2 text+predictor (LibriTTS HiFi-GAN) @ T_tok={t_tok}."
        )
        mlmodel.save(str(out_path))
        print(f"[01]   saved → {out_path}")


if __name__ == "__main__":
    main()
