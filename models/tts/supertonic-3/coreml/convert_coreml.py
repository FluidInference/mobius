"""Trace each PyTorch port and convert to a CoreML mlpackage.

Each module is traced with concrete shapes that match the validation harness, then
converted with `coremltools.convert` using `RangeDim` on the variable axes so the
resulting mlpackage handles the real-world dynamic input sizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import coremltools as ct
from coremltools import ImageType  # noqa: F401  (silence import warning)

from .vocoder import build_vocoder_from_onnx
from .text_encoder import build_text_encoder_from_onnx
from .duration_predictor import build_duration_predictor_from_onnx
from .vector_estimator import build_vector_estimator_from_onnx


# --- shared conversion settings ------------------------------------------------
MIN_DEPLOY = ct.target.iOS18       # ≥ macOS 15, ≥ iOS 18 (multi-enum shapes)
# FP32 = numerical-parity reference; FP16 = required for ANE residency (3/4
# modules; vector_estimator still blocked by 1 bool-tile op — see trials.md).
COMPUTE_PRECISION = ct.precision.FLOAT32
CONVERT_TO = "mlprogram"
COMPUTE_UNITS = ct.ComputeUnit.CPU_AND_NE


def _save(mlmodel, out_path: Path, *, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"  [OK] saved {label} → {out_path}")


# --- vocoder -------------------------------------------------------------------
def convert_vocoder(onnx_path: Path, tts_json: Path, out_dir: Path) -> None:
    print("=== Vocoder → CoreML ===")
    model = build_vocoder_from_onnx(onnx_path, tts_json_path=tts_json).eval()
    L_ttl = 4
    example = torch.randn(1, 144, L_ttl)
    traced = torch.jit.trace(model, example)
    # Lower bound 4 keeps unpacked L=24 ≥ max convnext replicate pad (k=7,dil=4 → pad=12).
    L_range = ct.RangeDim(lower_bound=4, upper_bound=512, default=L_ttl)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=(1, 144, L_range), dtype=np.float32)],
        outputs=[ct.TensorType(name="wav", dtype=np.float32)],
        convert_to=CONVERT_TO,
        compute_precision=COMPUTE_PRECISION,
        compute_units=COMPUTE_UNITS,
        minimum_deployment_target=MIN_DEPLOY,
    )
    _save(mlmodel, out_dir / "Vocoder.mlpackage", label="vocoder")


# --- text_encoder --------------------------------------------------------------
def convert_text_encoder(onnx_path: Path, out_dir: Path, T: int = 128) -> None:
    """Relative-position attention pads with T-dependent 4-D widths which trace
    records dynamically; use a fixed T (callers must pad text to this length)."""
    print(f"=== TextEncoder → CoreML (fixed T={T}) ===")
    model = build_text_encoder_from_onnx(onnx_path).eval()
    example = (
        torch.zeros(1, T, dtype=torch.int32),
        torch.randn(1, 50, 256),
        torch.ones(1, 1, T),
    )

    class _Int32Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, text_ids, style_ttl, text_mask):
            return self.m(text_ids.long(), style_ttl, text_mask)

    traced = torch.jit.trace(_Int32Wrapper(model), example)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="text_ids", shape=(1, T), dtype=np.int32),
            ct.TensorType(name="style_ttl", shape=(1, 50, 256), dtype=np.float32),
            ct.TensorType(name="text_mask", shape=(1, 1, T), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="text_emb", dtype=np.float32)],
        convert_to=CONVERT_TO,
        compute_precision=COMPUTE_PRECISION,
        compute_units=COMPUTE_UNITS,
        minimum_deployment_target=MIN_DEPLOY,
    )
    _save(mlmodel, out_dir / "TextEncoder.mlpackage", label="text_encoder")


# --- duration_predictor --------------------------------------------------------
def convert_duration_predictor(onnx_path: Path, out_dir: Path, T: int = 128) -> None:
    """Same dynamic-pad limitation as text_encoder → fixed T (caller pads)."""
    print(f"=== DurationPredictor → CoreML (fixed T={T}) ===")
    model = build_duration_predictor_from_onnx(onnx_path).eval()
    example = (
        torch.zeros(1, T, dtype=torch.int32),
        torch.randn(1, 8, 16),
        torch.ones(1, 1, T),
    )

    class _Int32Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, text_ids, style_dp, text_mask):
            return self.m(text_ids.long(), style_dp, text_mask)

    traced = torch.jit.trace(_Int32Wrapper(model), example)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="text_ids", shape=(1, T), dtype=np.int32),
            ct.TensorType(name="style_dp", shape=(1, 8, 16), dtype=np.float32),
            ct.TensorType(name="text_mask", shape=(1, 1, T), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="duration", dtype=np.float32)],
        convert_to=CONVERT_TO,
        compute_precision=COMPUTE_PRECISION,
        compute_units=COMPUTE_UNITS,
        minimum_deployment_target=MIN_DEPLOY,
    )
    _save(mlmodel, out_dir / "DurationPredictor.mlpackage", label="duration_predictor")


# --- vector_estimator ----------------------------------------------------------
def convert_vector_estimator(onnx_path: Path, out_dir: Path) -> None:
    print("=== VectorEstimator → CoreML ===")
    model = build_vector_estimator_from_onnx(onnx_path).eval()
    L, T_text = 24, 24
    example = (
        torch.randn(1, 144, L),
        torch.randn(1, 256, T_text),
        torch.randn(1, 50, 256),
        torch.ones(1, 1, L),
        torch.ones(1, 1, T_text),
        torch.tensor([7.0]),
        torch.tensor([8.0]),
    )
    traced = torch.jit.trace(model, example)
    # Lower bounds cover max convnext replicate pad (k=5, dil=8 → pad=16).
    L_range = ct.RangeDim(lower_bound=17, upper_bound=512, default=L)
    T_range = ct.RangeDim(lower_bound=17, upper_bound=512, default=T_text)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="noisy_latent", shape=(1, 144, L_range), dtype=np.float32),
            ct.TensorType(name="text_emb",     shape=(1, 256, T_range), dtype=np.float32),
            ct.TensorType(name="style_ttl",    shape=(1, 50, 256),      dtype=np.float32),
            ct.TensorType(name="latent_mask",  shape=(1, 1, L_range),   dtype=np.float32),
            ct.TensorType(name="text_mask",    shape=(1, 1, T_range),   dtype=np.float32),
            ct.TensorType(name="current_step", shape=(1,),              dtype=np.float32),
            ct.TensorType(name="total_step",   shape=(1,),              dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="denoised_latent", dtype=np.float32)],
        convert_to=CONVERT_TO,
        compute_precision=COMPUTE_PRECISION,
        compute_units=COMPUTE_UNITS,
        minimum_deployment_target=MIN_DEPLOY,
    )
    _save(mlmodel, out_dir / "VectorEstimator.mlpackage", label="vector_estimator")


# --- driver --------------------------------------------------------------------
def main(onnx_dir: Path, out_dir: Path, only: List[str] | None = None) -> None:
    tts_json = onnx_dir / "tts.json"
    all_stages = {
        "vocoder": lambda: convert_vocoder(onnx_dir / "vocoder.onnx", tts_json, out_dir),
        "text_encoder": lambda: convert_text_encoder(onnx_dir / "text_encoder.onnx", out_dir),
        "duration_predictor": lambda: convert_duration_predictor(onnx_dir / "duration_predictor.onnx", out_dir),
        "vector_estimator": lambda: convert_vector_estimator(onnx_dir / "vector_estimator.onnx", out_dir),
    }
    stages = only or list(all_stages.keys())
    for name in stages:
        if name not in all_stages:
            raise SystemExit(f"unknown stage: {name}")
        all_stages[name]()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("onnx_dir", type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--fp16", action="store_true",
                   help="convert with FLOAT16 precision (required for ANE residency)")
    p.add_argument("--stage", action="append", default=None,
                   help="restrict to one or more stages (repeatable)")
    args = p.parse_args()

    if args.fp16:
        COMPUTE_PRECISION = ct.precision.FLOAT16  # noqa: F811
    out_dir = args.out_dir or (args.onnx_dir.parent / ("_mlpackage_fp16" if args.fp16 else "_mlpackage"))
    main(args.onnx_dir, out_dir, args.stage)
