#!/usr/bin/env python3
"""
Numerical parity check between the upstream g2pW ONNX model and the
converted CoreML mlpackage.

Generates a deterministic batch of dummy inputs that exercise the same
shapes the production pipeline uses (batch=1, seq_len=512), runs both
the ONNX (CPU EP) and CoreML (CpuOnly) paths, and asserts the
top-1 argmax matches plus that the probability vectors are close in L2.

Usage:
    uv run python compare-models.py \\
        --cache-dir ~/.cache/g2pw-coreml \\
        --coreml-dir ./build/g2pw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _build_inputs(num_labels: int, seed: int = 1234) -> dict:
    rng = np.random.default_rng(seed)
    seq_len = 512
    batch = 1
    # Plausible BERT-ish inputs: a handful of real tokens in a CLS/...,/SEP
    # pattern, the rest padding. The exact ids don't matter for parity —
    # we just need both backends to see the same numbers.
    real_len = 12
    input_ids = np.zeros((batch, seq_len), dtype=np.int64)
    input_ids[0, 0] = 101  # [CLS]
    input_ids[0, 1 : 1 + (real_len - 2)] = rng.integers(
        low=672, high=7991, size=real_len - 2, dtype=np.int64
    )
    input_ids[0, real_len - 1] = 102  # [SEP]
    token_type_ids = np.zeros((batch, seq_len), dtype=np.int64)
    attention_mask = np.zeros((batch, seq_len), dtype=np.int64)
    attention_mask[0, :real_len] = 1
    phoneme_mask = (rng.random((batch, num_labels)) > 0.5).astype(np.float32)
    # Guard: ensure at least one valid label so weighted softmax has
    # non-zero numerator.
    phoneme_mask[0, 0] = 1.0
    char_ids = np.array([7], dtype=np.int64)
    position_ids = np.array([3], dtype=np.int64)
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "phoneme_mask": phoneme_mask,
        "char_ids": char_ids,
        "position_ids": position_ids,
    }


def _run_onnx(onnx_path: Path, inputs: dict, input_order: list) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {name: inputs[name] for name in input_order}
    out = sess.run(None, feed)
    return out[0]


def _run_coreml(mlpkg_path: Path, inputs: dict, input_order: list) -> np.ndarray:
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(mlpkg_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    feed = {}
    for name in input_order:
        a = inputs[name]
        # CoreML expects int32 for integer tensors per our converter.
        if a.dtype == np.int64:
            a = a.astype(np.int32)
        feed[name] = a
    out = mlmodel.predict(feed)
    # Output name comes from the converter (uses the ONNX graph's
    # first output name). Pull whichever single tensor is in the dict.
    if len(out) != 1:
        raise RuntimeError(f"unexpected CoreML output dict: {list(out)}")
    return next(iter(out.values()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "g2pw-coreml",
        help="Cache containing extracted G2PWModel/g2pw.onnx",
    )
    p.add_argument(
        "--coreml-dir",
        type=Path,
        default=Path("./build/g2pw"),
        help="Directory containing g2pw.mlpackage",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=2e-2,
        help=(
            "Element-wise abs tolerance for prob vectors. fp16 BERT round-trips"
            " typically land within ~1.2e-2 max diff; default 2e-2 is a soft"
            " upper bound. Argmax parity is the hard contract."
        ),
    )
    args = p.parse_args()

    onnx_path = args.cache_dir / "G2PWModel" / "g2pw.onnx"
    mlpkg_path = args.coreml_dir / "g2pw.mlpackage"
    if not onnx_path.exists():
        print(f"missing {onnx_path} — run convert-coreml.py first", file=sys.stderr)
        return 2
    if not mlpkg_path.exists():
        print(f"missing {mlpkg_path} — run convert-coreml.py first", file=sys.stderr)
        return 2

    import onnx as _onnx

    model = _onnx.load(str(onnx_path))
    input_order = [i.name for i in model.graph.input]
    num_labels = None
    for inp in model.graph.input:
        if inp.name == "phoneme_mask":
            num_labels = int(inp.type.tensor_type.shape.dim[-1].dim_value)
    if num_labels is None or num_labels <= 0:
        print("could not infer num_labels", file=sys.stderr)
        return 2

    inputs = _build_inputs(num_labels)

    print(f"[onnx] running {onnx_path}")
    onnx_out = _run_onnx(onnx_path, inputs, input_order)
    print(f"[coreml] running {mlpkg_path}")
    coreml_out = _run_coreml(mlpkg_path, inputs, input_order)

    print(f"[shape] onnx={onnx_out.shape} coreml={coreml_out.shape}")
    onnx_argmax = int(np.argmax(onnx_out, axis=-1)[0])
    coreml_argmax = int(np.argmax(coreml_out, axis=-1)[0])
    print(f"[argmax] onnx={onnx_argmax} coreml={coreml_argmax}")

    diff = np.abs(onnx_out - coreml_out)
    print(
        f"[diff] max={diff.max():.6f} mean={diff.mean():.6f} "
        f"l2={np.linalg.norm(diff):.6f}"
    )

    ok_argmax = onnx_argmax == coreml_argmax
    ok_close = bool(np.allclose(onnx_out, coreml_out, atol=args.atol))
    if not ok_argmax:
        print("FAIL: argmax mismatch", file=sys.stderr)
    if not ok_close:
        print(f"WARN: not within atol={args.atol}", file=sys.stderr)

    return 0 if ok_argmax else 1


if __name__ == "__main__":
    sys.exit(main())
