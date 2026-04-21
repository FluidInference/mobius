"""Inspect FP16 decoder weight names to find the tied-embedding const name."""
from __future__ import annotations

from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import get_weights_metadata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DECODER = ROOT / "hf-upload/f16-download/f16/cohere_decoder_stateful.mlpackage"


def main():
    print(f"Loading {DECODER}...")
    model = ct.models.MLModel(str(DECODER), compute_units=ct.ComputeUnit.CPU_ONLY)
    meta = get_weights_metadata(model, weight_threshold=1024)
    rows = []
    for name, info in meta.items():
        val = getattr(info, "val", None)
        if val is None:
            continue
        shape = tuple(val.shape)
        numel = 1
        for d in shape:
            numel *= int(d)
        child = getattr(info, "child_ops", [])
        rows.append((numel, shape, name, child))
    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"\n{len(rows)} weights >=1024 numel. Top 10:")
    for n, shape, name, child in rows[:10]:
        child_info = ""
        if child:
            c0 = child[0]
            child_info = f"  -> {getattr(c0, 'op_type', '?')} / {getattr(c0, 'name', '?')}"
        print(f"  {n:>14,}  shape={str(shape):25s}  const_name={name}{child_info}")

    # Focus on anything that connects to a matmul/linear producing logit-shaped output
    print("\nOps of type linear/matmul whose weight is in the metadata:")
    for n, shape, name, child in rows:
        for c in child or []:
            ct_ = getattr(c, "op_type", "")
            if ct_ in ("linear", "matmul"):
                print(f"  numel={n:>12,}  weight_const={name}  op_type={ct_}  op_name={getattr(c, 'name', '?')}")


if __name__ == "__main__":
    main()
