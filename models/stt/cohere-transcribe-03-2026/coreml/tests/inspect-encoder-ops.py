"""Inspect the companion encoder weights via coremltools.optimize.

We need to know:
  1. Where the 7 GB lives (biggest weight tensors)
  2. Any shared/tied consts between ops (would force matched per-op configs
     during INT8 quantization, analogous to the decoder's lm_head / embed pair)
"""
from __future__ import annotations

from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import get_weights_metadata  # type: ignore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENCODER = ROOT / "hf-upload/cohere-transcribe-cache-external-coreml/cohere_encoder.mlpackage"


def main():
    print(f"Loading {ENCODER}...")
    model = ct.models.MLModel(str(ENCODER), compute_units=ct.ComputeUnit.CPU_ONLY)
    meta = get_weights_metadata(model, weight_threshold=2048)

    rows = []
    shared = []
    for name, info in meta.items():
        val = getattr(info, "val", None)
        if val is None:
            continue
        shape = tuple(val.shape)
        numel = 1
        for d in shape:
            numel *= int(d)
        op_name = getattr(info, "op", None)
        op_type = getattr(info, "op_type", None)
        child_ops = getattr(info, "child_ops", [])
        rows.append((numel, shape, op_name, op_type, child_ops))
        if len(child_ops) > 1:
            shared.append((numel, shape, op_name, child_ops))

    rows.sort(key=lambda r: r[0], reverse=True)
    total = sum(r[0] for r in rows)
    print(f"\n{len(rows)} weights >= 2048 numel. Total elements: {total:,} "
          f"(~{total*4/1e9:.2f} GB FP32 / ~{total*2/1e9:.2f} GB FP16 / ~{total/1e9:.2f} GB INT8)")
    print("\nTop 25 by numel:")
    for n, shape, op_name, op_type, child in rows[:25]:
        cstr = ""
        if child:
            c0 = child[0]
            cstr = f"  -> {getattr(c0, 'op_type', '?')}/{getattr(c0, 'name', '?')}"
            if len(child) > 1:
                cstr += f"  (+{len(child)-1} more consumers)"
        print(f"  {n:>14,}  shape={str(shape):28s}  op={op_name}{cstr}")

    print(f"\n\nShared consts (>1 consumer): {len(shared)}")
    if not shared:
        print("  (none — default per-channel config should not hit config-conflict errors)")
    else:
        shared.sort(key=lambda r: -r[0])
        for n, shape, op_name, child_ops in shared[:20]:
            types = [getattr(c, "op_type", "?") for c in child_ops]
            names = [getattr(c, "name", "?") for c in child_ops]
            print(f"  {op_name}  numel={n:,} shape={shape}")
            for t, nm in zip(types, names):
                print(f"    -> {t}/{nm}")


if __name__ == "__main__":
    main()
