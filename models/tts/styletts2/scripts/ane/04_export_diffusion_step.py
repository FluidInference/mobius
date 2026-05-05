"""Stage 4/7 — DiffusionStep (single ADPM2/Karras denoise step).

ANE-resident, fp16 + int8pal kmeans. **Fully static shapes** (no enum, no
RangeDim) — this is the key fix vs the legacy diffusion graph, which used
EnumeratedShapes on `embedding`/`attention_mask` and triggered the E5RT
FlexibleShapeInfo runtime exception.

The 5-step ADPM2 sampler (11 invocations per utterance) lives in Swift —
the existing `StyleTTS2Sampler` is model-agnostic and is reused unchanged.

Inputs (all fixed):
  x_noisy   [1, 1, 256]    fp32
  sigma     [1]            fp32
  embedding [1, 512, 768]  fp32   (max_position_embeddings — uncond uses fixed_embedding)
  features  [1, 256]       fp32   (= ref_s)

Output:
  denoised  [1, 1, 256]    fp32
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
    DiffusionStepTraceable,
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
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[04-ane] loading modules …")
    modules, cfg = load_modules_for_ane(args.checkpoint)
    wrapper = DiffusionStepTraceable(modules, sigma_data=cfg.diffusion_sigma_data).eval()

    bert_hidden = modules["bert"].config.hidden_size
    T_emb = modules["bert"].config.max_position_embeddings
    style_dim_2x = cfg.style_dim * 2

    x_e = torch.zeros(1, 1, style_dim_2x, dtype=torch.float32)
    sigma_e = torch.tensor([1.0], dtype=torch.float32)
    emb_e = torch.zeros(1, T_emb, bert_hidden, dtype=torch.float32)
    feat_e = torch.zeros(1, style_dim_2x, dtype=torch.float32)

    print(f"[04-ane] sanity forward (static T_emb={T_emb}) …")
    with torch.no_grad():
        out = wrapper(x_e, sigma_e, emb_e, feat_e)
    print(f"[04-ane]   denoised: {tuple(out.shape)} {out.dtype}")

    print("[04-ane] tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (x_e, sigma_e, emb_e, feat_e), strict=False)

    if args.trace_only:
        print("[04-ane] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct

    out_path = args.out_dir / "styletts2_ane_diffusion_step.mlpackage"
    print(f"[04-ane] converting → {out_path.name} (cpu_and_ne, fp16, STATIC shapes) …", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_noisy", shape=(1, 1, style_dim_2x), dtype=np.float32),
            ct.TensorType(name="sigma", shape=(1,), dtype=np.float32),
            ct.TensorType(name="embedding", shape=(1, T_emb, bert_hidden), dtype=np.float32),
            ct.TensorType(name="features", shape=(1, style_dim_2x), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="denoised")],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    mlmodel.short_description = (
        "StyleTTS2-ANE Stage 4: ADPM2 single denoise step (static shapes) — fp16, ANE."
    )
    mlmodel.save(str(out_path))
    print(f"[04-ane]   saved → {out_path}")

    if not args.no_palettize:
        print("[04-ane] palettizing …", flush=True)
        palettize_int8(out_path)
        print(f"[04-ane]   palettized → {out_path}")


if __name__ == "__main__":
    main()
