"""Inspect the q8 stateful decoder via coremltools MIL program API."""
from __future__ import annotations

from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import get_weights_metadata  # type: ignore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DECODER = ROOT / "hf-upload/q8-download/q8/cohere_decoder_stateful.mlpackage"


def main():
    print(f"Loading {DECODER}...")
    model = ct.models.MLModel(str(DECODER), compute_units=ct.ComputeUnit.CPU_ONLY)
    try:
        meta = get_weights_metadata(model, weight_threshold=1024)
    except Exception as e:
        print(f"  get_weights_metadata failed: {e}")
        meta = {}

    rows = []
    for name, info in meta.items():
        # info.val is the ndarray; info.op is the op name
        val = getattr(info, "val", None)
        if val is None:
            continue
        shape = tuple(val.shape)
        numel = 1
        for d in shape:
            numel *= int(d)
        op_name = getattr(info, "op", None)
        op_type = getattr(info, "op_type", None)
        child_names = getattr(info, "child_ops", [])
        rows.append((numel, shape, op_name, op_type, child_names))

    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"\n{len(rows)} weights >=1024 numel. Top 20:")
    for n, shape, op_name, op_type, child in rows[:20]:
        child_info = ""
        if child:
            c0 = child[0]
            child_info = f" -> {getattr(c0, 'op_type', '?')} / {getattr(c0, 'name', '?')}"
        print(f"  {n:>14,}  shape={str(shape):25s}  op={op_name}{child_info}")


if __name__ == "__main__":
    main()
