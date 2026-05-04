#!/usr/bin/env python3
"""
Smoke test for the converted g2pW CoreML mlpackage.

Loads `./build/g2pw/g2pw.mlpackage`, feeds a single batched dummy input
that matches the production shape contract (batch=1, seq_len=512), and
asserts the model returns a plausible probability vector summing to ~1.

Doesn't require the upstream ONNX checkpoint — only the CoreML output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coreml-dir",
        type=Path,
        default=Path("./build/g2pw"),
    )
    args = p.parse_args()

    mlpkg = args.coreml_dir / "g2pw.mlpackage"
    if not mlpkg.exists():
        print(f"missing {mlpkg}", file=sys.stderr)
        return 2

    import coremltools as ct

    mlmodel = ct.models.MLModel(str(mlpkg), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = mlmodel.get_spec()
    input_shapes = {
        i.name: tuple(i.type.multiArrayType.shape) for i in spec.description.input
    }
    print(f"[spec] inputs: {input_shapes}")
    print(
        "[spec] outputs:",
        [o.name for o in spec.description.output],
    )

    seq_len = 512
    batch = 1
    num_labels = input_shapes["phoneme_mask"][-1]

    feed = {
        "input_ids": np.zeros((batch, seq_len), dtype=np.int32),
        "token_type_ids": np.zeros((batch, seq_len), dtype=np.int32),
        "attention_mask": np.zeros((batch, seq_len), dtype=np.int32),
        "phoneme_mask": np.ones((batch, num_labels), dtype=np.float32),
        "char_ids": np.zeros((batch,), dtype=np.int32),
        "position_ids": np.zeros((batch,), dtype=np.int32),
    }
    feed["attention_mask"][0, :8] = 1

    out = mlmodel.predict(feed)
    if len(out) != 1:
        print(f"unexpected output dict: {list(out)}", file=sys.stderr)
        return 1
    probs = next(iter(out.values()))
    print(f"[out] shape={probs.shape} dtype={probs.dtype}")
    s = float(probs.sum(axis=-1)[0])
    top1 = int(np.argmax(probs, axis=-1)[0])
    print(f"[out] sum={s:.4f} top1={top1}")

    if not (0.95 <= s <= 1.05):
        print(f"FAIL: probs don't sum to ~1 (got {s:.4f})", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
