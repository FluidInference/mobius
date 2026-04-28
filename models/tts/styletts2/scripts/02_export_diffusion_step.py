"""Export Package B: a single ADPM2 / Karras denoising step of the style UNet.

Sampler loop runs in Swift. CFG combination (cond + uncond) also in Swift —
Swift calls this model twice with different `embedding` inputs.

Inputs:
  x_noisy:   (1, 1, 256)
  sigma:     (1,)                  scalar noise level
  embedding: (1, T_tok, 768)       bert_dur (cond) or fixed_embedding (uncond)
  features:  (1, 256)              ref_s

Output:
  denoised:  (1, 1, 256)

Compute units: CPU + GPU. Small UNet, ~5 calls per utterance.

Bucketing: a single mlpackage with EnumeratedShapes on `embedding` compiles
fine but fails to load on iOS17 mlprogram with
  "tensor_buffer has known strides while the model has FlexibleShapeInfo.
   Strides must be unknown on all dimensions." (CoreML err -14)
Mirroring 04_export_decoder.py, we emit one fixed-shape mlpackage per token
bucket: `styletts2_diffusion_step_<T_tok>.mlpackage`. Callers route by length.
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
    DiffusionDenoiseTraceable,
    LibriTTSConfig,
    load_inference_modules,
)

DEFAULT_TOK_BUCKETS = (32, 64, 128, 256, 512)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out-dir", type=Path, default=COREML_DIR,
        help="Directory; per-bucket mlpackages are saved as styletts2_diffusion_step_<T_tok>.mlpackage.",
    )
    parser.add_argument("--buckets", type=int, nargs="+", default=list(DEFAULT_TOK_BUCKETS))
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpu_and_gpu", "cpu_only"],
        default="all",
        help="Default 'all' targets ANE; fall back to cpu_and_gpu if placement fails.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[02] loading inference modules …")
    modules, cfg = load_inference_modules(args.checkpoint)

    print("[02] building DiffusionDenoiseTraceable …")
    wrapper = DiffusionDenoiseTraceable(modules, cfg.diffusion_sigma_data).eval()

    style_ch = cfg.style_dim * 2  # 256
    bert_dim = modules["bert"].config.hidden_size  # 768

    if args.trace_only:
        largest = max(args.buckets)
        example_x = torch.zeros(1, 1, style_ch, dtype=torch.float32)
        example_sigma = torch.tensor([1.0], dtype=torch.float32)
        example_embedding = torch.zeros(1, largest, bert_dim, dtype=torch.float32)
        example_features = torch.zeros(1, style_ch, dtype=torch.float32)
        print(f"[02] sanity forward at T_tok={largest} …")
        with torch.no_grad():
            out = wrapper(example_x, example_sigma, example_embedding, example_features)
        print(f"[02]   denoised: {tuple(out.shape)} {out.dtype}")
        print("[02] tracing (largest bucket only) …")
        with torch.no_grad():
            torch.jit.trace(
                wrapper,
                (example_x, example_sigma, example_embedding, example_features),
                strict=False,
            )
        print("[02] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct

    cu = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }[args.compute_units]

    # See module docstring: EnumeratedShapes on `embedding` triggers the
    # iOS17 mlprogram FlexibleShapeInfo strides bug at runtime. Emit one
    # fixed-shape mlpackage per token bucket; route at runtime.
    for t_tok in sorted(args.buckets):
        out_path = args.out_dir / f"styletts2_diffusion_step_{t_tok}.mlpackage"
        print(f"[02] === T_tok={t_tok}  →  {out_path.name} ===")

        example_x = torch.zeros(1, 1, style_ch, dtype=torch.float32)
        example_sigma = torch.tensor([1.0], dtype=torch.float32)
        example_embedding = torch.zeros(1, t_tok, bert_dim, dtype=torch.float32)
        example_features = torch.zeros(1, style_ch, dtype=torch.float32)

        with torch.no_grad():
            out = wrapper(example_x, example_sigma, example_embedding, example_features)
        print(f"[02]   sanity denoised: {tuple(out.shape)} {out.dtype}")

        print("[02]   tracing …")
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                (example_x, example_sigma, example_embedding, example_features),
                strict=False,
            )

        print(f"[02]   converting (compute_units={args.compute_units}) …", flush=True)
        # skip_model_load avoids the post-convert MLModel instantiation which
        # otherwise hangs Apple's anecompilerservice for several minutes per
        # bucket. The mlpackage is saved correctly; first runtime load pays
        # the compile cost once, then it's cached.
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="x_noisy", shape=(1, 1, style_ch), dtype=np.float32),
                ct.TensorType(name="sigma", shape=(1,), dtype=np.float32),
                ct.TensorType(name="embedding", shape=(1, t_tok, bert_dim), dtype=np.float32),
                ct.TensorType(name="features", shape=(1, style_ch), dtype=np.float32),
            ],
            outputs=[ct.TensorType(name="denoised")],
            minimum_deployment_target=ct.target.iOS17,
            compute_units=cu,
            convert_to="mlprogram",
            skip_model_load=True,
        )
        mlmodel.short_description = (
            f"StyleTTS2 KDiffusion single denoise step (StyleTransformer1d) @ T_tok={t_tok}."
        )
        mlmodel.save(str(out_path))
        print(f"[02]   saved → {out_path}")


if __name__ == "__main__":
    main()
