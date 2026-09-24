#!/usr/bin/env python3
"""Parity + latency + compute-unit placement for a compiled Encoder.mlmodelc against the PyTorch reference."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np

CU = {
    "all": ct.ComputeUnit.ALL,
    "ane": ct.ComputeUnit.CPU_AND_NE,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu": ct.ComputeUnit.CPU_ONLY,
}


def placement(path: Path, cu: ct.ComputeUnit) -> str:
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(str(path), cu)
        prog = plan.model_structure.program
        counts: dict = {}
        cpu_ops: dict = {}
        for f in prog.functions.values():
            for op in f.block.operations:
                if op.operator_name == "const":
                    continue
                info = plan.get_compute_device_usage_for_mlprogram_operation(op)
                dev = type(info.preferred_compute_device).__name__ if info else "None"
                counts[dev] = counts.get(dev, 0) + 1
                if dev == "MLCPUComputeDevice" and cu != ct.ComputeUnit.CPU_ONLY:
                    cpu_ops[op.operator_name] = cpu_ops.get(op.operator_name, 0) + 1
        return f"{counts}  cpu-fallback ops: {cpu_ops}" if cpu_ops else str(counts)
    except Exception as e:  # noqa: BLE001
        return f"(compute plan unavailable: {e})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--ref", type=Path, default=Path("~/Documents/parakeet-redux-work/encoder_ref_redux.npz").expanduser())
    ap.add_argument("--units", default="ane,gpu")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--no-parity", action="store_true", help="latency only (e.g. stock encoder with other weights)")
    args = ap.parse_args()

    ref = np.load(args.ref)
    inputs = {"mel": ref["mel"].astype(np.float32), "mel_length": ref["mel_length"].astype(np.int32)}
    ref_enc = ref["encoder"]
    for name in args.units.split(","):
        cu = CU[name]
        t0 = time.time()
        m = ct.models.CompiledMLModel(str(args.model), compute_units=cu) if args.model.suffix == ".mlmodelc" \
            else ct.models.MLModel(str(args.model), compute_units=cu)
        load_s = time.time() - t0
        out = m.predict(inputs)
        enc = out["encoder"]
        ts = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            m.predict(inputs)
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1e3
        line = f"[{name}] load {load_s:.1f}s  latency median {np.median(ts):.1f} ms  min {ts.min():.1f} ms"
        if not args.no_parity:
            d = np.abs(enc - ref_enc)
            cos = float(np.dot(enc.flatten(), ref_enc.flatten()) / (np.linalg.norm(enc) * np.linalg.norm(ref_enc)))
            line += f"  | vs torch: max|d| {d.max():.4f} mean|d| {d.mean():.5f} rel {d.max() / np.abs(ref_enc).max():.5f} cos {cos:.6f}"
            line += f"  len {int(out['encoder_length'].flatten()[0])}/{int(ref['encoder_length'].flatten()[0])}"
        print(line)
        if args.model.suffix == ".mlmodelc":
            print("   placement:", placement(args.model, cu))


if __name__ == "__main__":
    main()
