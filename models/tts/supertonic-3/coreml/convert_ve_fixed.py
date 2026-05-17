"""Convert VectorEstimator with FIXED L and T for ANE profiling.

The default `convert_coreml.py` uses `RangeDim(17..512)` for the latent and
text axes, which ANE rejects with "Data-dependent shapes were disabled".
This helper traces with a single fixed L and saves an FP16 mlpackage so
the ANE compiler has concrete shapes to plan against.

Status (M2, macOS 26.5): 89.6% of ops are ANE-eligible, but ANE plan build
still fails on one bool `tile` op in the style cross-attention mask. See
`trials.md` for the analysis. Until that is fixed, the model falls back to
CPU+GPU at runtime.

Usage:
    uv run python -m coreml.convert_ve_fixed \
        --onnx build/_onnx/vector_estimator.onnx \
        --out  build/_mlpackage_fp16_fixed/VectorEstimator_L128.mlpackage \
        --L 128 --T 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from .vector_estimator import build_vector_estimator_from_onnx


class _Int32Wrapper(torch.nn.Module):
    """No-op wrapper kept for symmetry with the other converters that need
    `text_ids.long()` casting inside the traced graph. vector_estimator has
    no int inputs, so this only fixes the input signature."""

    def __init__(self, mod: torch.nn.Module) -> None:
        super().__init__()
        self.mod = mod

    def forward(self, noisy, text_emb, style_ttl, latent_mask, text_mask, cur, total):
        return self.mod(noisy, text_emb, style_ttl, latent_mask, text_mask, cur, total)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--L", type=int, default=128, help="Fixed latent length (TTL slots)")
    p.add_argument("--T", type=int, default=128, help="Fixed text length")
    args = p.parse_args()

    torch_mod = build_vector_estimator_from_onnx(args.onnx).eval()
    L, T = args.L, args.T
    print(f"Tracing with fixed L={L}, T={T}")

    sample = (
        torch.randn(1, 144, L, dtype=torch.float32),
        torch.randn(1, 256, T, dtype=torch.float32),
        torch.randn(1, 50, 256, dtype=torch.float32),
        torch.ones(1, 1, L, dtype=torch.float32),
        torch.ones(1, 1, T, dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([8.0], dtype=torch.float32),
    )

    wrapped = _Int32Wrapper(torch_mod).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, sample)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="noisy_latent", shape=(1, 144, L), dtype=np.float32),
            ct.TensorType(name="text_emb", shape=(1, 256, T), dtype=np.float32),
            ct.TensorType(name="style_ttl", shape=(1, 50, 256), dtype=np.float32),
            ct.TensorType(name="latent_mask", shape=(1, 1, L), dtype=np.float32),
            ct.TensorType(name="text_mask", shape=(1, 1, T), dtype=np.float32),
            ct.TensorType(name="current_step", shape=(1,), dtype=np.float32),
            ct.TensorType(name="total_step", shape=(1,), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="denoised_latent", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    print("saved:", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
