"""Fuse the 8-step VectorEstimator denoising loop into ONE CoreML graph.

Motivation
----------
The shipped int4 L-bucketed VectorEstimator (94% ANE, 3.8 ms/step) is
dispatched 8x per chunk by the Swift host (Supertonic3Synthesizer): the
denoising loop is ~31 ms of an ~81 ms synth. Following the Trial-16 /
flow_decoder_fused precedent, this converter unrolls all 8 steps into a
single traced graph:

  - `current_step` 0..7 / `total_step` 8 become trace-time constants
    (the per-step time embeddings are precomputed eagerly and baked in
    as buffers -> MIL consts).
  - Step-invariant work is hoisted out of the loop and computed ONCE:
    the CFG batch duplication of text_emb / style_key / style_value
    (UncondMasker) and the latent/text mask concats. The single-step
    graph redoes all of this every dispatch.
  - The denoised->noisy feedback is internal to the graph.

Weights are shared across the 8 unrolled steps (same nn.Parameter objects
under torch.jit.trace -> one MIL const each), so the fused model is the
same size as the single-step build.

Quantization follows the shipped pipeline's own order (convert_ve_quant.py):
fp16 trace first, then post-training 4-bit k-means palettization via
`coremltools.optimize.coreml` on the converted mlpackage.

Usage:
    python3.11 -m coreml.convert_ve_fused \
        --onnx build/_onnx/vector_estimator.onnx \
        --out-dir build/_mlpackage_ve_fused \
        --L 128 --T 128 --steps 8 \
        --variants fp16 palette4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
import torch.nn as nn

from .vector_estimator import (
    W_COND,
    W_UNCOND,
    VectorEstimator,
    build_vector_estimator_from_onnx,
)


class FusedVectorEstimator(nn.Module):
    """8-step flow-matching loop unrolled into a single forward pass.

    Mirrors Supertonic3Synthesizer's host loop exactly: for k in 0..steps-1,
    denoised = VE(noisy, ..., current_step=k, total_step=steps); noisy = denoised.
    """

    def __init__(self, base: VectorEstimator, total_steps: int = 8) -> None:
        super().__init__()
        self.base = base
        self.total_steps = total_steps
        self.step_scale = 1.0 / float(total_steps)
        # current_step/total_step are compile-time constants in the fused
        # graph; the time embeddings they feed are therefore constants too.
        # Precompute them eagerly (batch 2 = [cond | uncond]).
        with torch.no_grad():
            embs = [
                base.time_encoder(torch.full((2,), float(k) / float(total_steps)))
                for k in range(total_steps)
            ]
        self.register_buffer("time_embs", torch.stack(embs, dim=0))  # (S, 2, 64, 1)

    def forward(
        self,
        noisy_latent: torch.Tensor,  # [1, 144, L]
        text_emb: torch.Tensor,      # [1, 256, T]
        style_ttl: torch.Tensor,     # [1, 50, 256]
        latent_mask: torch.Tensor,   # [1, 1, L]
        text_mask: torch.Tensor,     # [1, 1, T]
    ) -> torch.Tensor:
        b = self.base
        B = noisy_latent.shape[0]
        T_text = text_emb.shape[-1]

        # Step-invariant: CFG batch duplication, computed ONCE (the
        # single-step graph rebuilds these every dispatch).
        text_b, style_key_b, style_value_b = b.uncond_masker(text_emb, style_ttl, T_text)
        latent_mask_b = torch.cat([latent_mask, latent_mask], dim=0)
        text_mask_b = torch.cat([text_mask, text_mask], dim=0)

        noisy = noisy_latent
        for k in range(self.total_steps):
            time_emb = self.time_embs[k]                       # (2, 64, 1) const
            noisy_b = torch.cat([noisy, noisy], dim=0)         # (2, 144, L)
            x = b.proj_in(noisy_b) * latent_mask_b             # (2, 512, L)
            for block in b.main_blocks:
                x = block(x, time_emb, text_b, style_key_b, style_value_b,
                          latent_mask_b, text_mask_b)
            for blk in b.last_convnext:
                x = blk(x * latent_mask_b, mask=latent_mask_b)
            v = b.proj_out(x) * latent_mask_b                  # (2, 144, L)
            cond = v[:B]
            uncond = v[B:]
            noisy = (noisy + self.step_scale * (W_COND * cond - W_UNCOND * uncond)) * latent_mask
        return noisy


def _sample(L: int, T: int):
    return (
        torch.randn(1, 144, L, dtype=torch.float32),
        torch.randn(1, 256, T, dtype=torch.float32),
        torch.randn(1, 50, 256, dtype=torch.float32),
        torch.ones(1, 1, L, dtype=torch.float32),
        torch.ones(1, 1, T, dtype=torch.float32),
    )


def _eager_self_check(base: VectorEstimator, fused: FusedVectorEstimator,
                      L: int, T: int, steps: int) -> None:
    """Fused module vs the literal host loop over base.forward, fp32 eager."""
    sample = _sample(L, T)
    noisy, text, style, lmask, tmask = sample
    total = torch.tensor([float(steps)])
    with torch.no_grad():
        ref = noisy
        for k in range(steps):
            ref = base(ref, text, style, lmask, tmask, torch.tensor([float(k)]), total)
        out = fused(noisy, text, style, lmask, tmask)
    max_abs = (out - ref).abs().max().item()
    print(f"eager self-check (fused vs {steps}x base loop, fp32): max_abs={max_abs:.3e}")
    if max_abs > 1e-5:
        raise SystemExit("fused module diverges from the host loop in eager fp32 — abort")


def _compress(base_ml: ct.models.MLModel, variant: str) -> ct.models.MLModel:
    if variant == "fp16":
        return base_ml
    if variant == "int8":
        cfg = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8", granularity="per_channel"
            )
        )
        return cto.linear_quantize_weights(base_ml, cfg)
    if variant.startswith("palette"):
        nbits = int(variant.replace("palette", ""))
        cfg = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(nbits=nbits, mode="kmeans")
        )
        return cto.palettize_weights(base_ml, cfg)
    raise SystemExit(f"unknown variant: {variant}")


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--variants", nargs="+", default=["fp16", "palette4"])
    args = p.parse_args()

    base = build_vector_estimator_from_onnx(args.onnx).eval()
    fused = FusedVectorEstimator(base, total_steps=args.steps).eval()
    _eager_self_check(base, fused, args.L, args.T, args.steps)

    print(f"Tracing fused VE: steps={args.steps}, L={args.L}, T={args.T}")
    with torch.no_grad():
        traced = torch.jit.trace(fused, _sample(args.L, args.T))

    L, T = args.L, args.T
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="noisy_latent", shape=(1, 144, L), dtype=np.float32),
            ct.TensorType(name="text_emb", shape=(1, 256, T), dtype=np.float32),
            ct.TensorType(name="style_ttl", shape=(1, 50, 256), dtype=np.float32),
            ct.TensorType(name="latent_mask", shape=(1, 1, L), dtype=np.float32),
            ct.TensorType(name="text_mask", shape=(1, 1, T), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="denoised_latent", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        print(f"\n=== variant: {variant} ===")
        m = _compress(mlmodel, variant)
        out = args.out_dir / f"VectorEstimator_L{L}_fused{args.steps}_{variant}.mlpackage"
        m.save(str(out))
        print(f"saved: {out}  ({_dir_size_mb(out):.1f} MB)")


if __name__ == "__main__":  # pragma: no cover
    main()
