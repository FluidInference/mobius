"""Probe ANE admission + placement for a converted graph.

Loads the package under CPU_AND_NE and runs a predict. If the ANE rejects the graph the
load/predict raises ANECCompile -14 (as the stateful decode graph does). Success => the ANE
accepted it. Also times CPU_AND_NE vs CPU_AND_GPU vs CPU_ONLY, and tries MLComputePlan for a
per-device op breakdown when available.
"""

import argparse
import time
from pathlib import Path

import numpy as np


def dummy_inputs(spec):
    feeds = {}
    for inp in spec.description.input:
        shape = tuple(int(d) for d in inp.type.multiArrayType.shape)
        feeds[inp.name] = np.random.randn(*shape).astype(np.float32) * 0.1
    return feeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    import coremltools as ct

    pkg = str(Path(args.package))
    results = {}
    for cu_name in ["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"]:
        cu = getattr(ct.ComputeUnit, cu_name)
        try:
            t0 = time.time()
            m = ct.models.MLModel(pkg, compute_units=cu)
            feeds = dummy_inputs(m.get_spec())
            _ = m.predict(feeds)  # first predict triggers ANE compile
            load_ms = (time.time() - t0) * 1000
            ts = []
            for _ in range(args.iters):
                t = time.time()
                m.predict(feeds)
                ts.append((time.time() - t) * 1000)
            ts = np.array(ts)
            results[cu_name] = (True, f"p50 {np.percentile(ts,50):.1f}ms  p99 {np.percentile(ts,99):.1f}ms  (load+compile {load_ms:.0f}ms)")
        except Exception as e:
            msg = str(e).replace("\n", " ")
            marker = "ANECCompile -14 (ANE REJECTED)" if "-14" in msg or "ANECCompile" in msg else msg[:120]
            results[cu_name] = (False, marker)

    print(f"\n=== Placement probe: {Path(args.package).name} ===")
    for cu_name, (ok, info) in results.items():
        print(f"{cu_name:14s} : {'OK   ' if ok else 'FAIL '} {info}")

    # Per-op device breakdown (best effort; API varies by coremltools version)
    try:
        from coremltools.models.compute_plan import MLComputePlan
        # MLComputePlan needs a compiled .mlmodelc; keep the MLModel alive so the temp
        # compiled dir is not GC-deleted before the plan loads.
        _keep = ct.models.MLModel(pkg)
        mlmodelc = _keep.get_compiled_model_path()
        plan = MLComputePlan.load_from_path(mlmodelc, compute_units=ct.ComputeUnit.CPU_AND_NE)
        prog = plan.model_structure.program
        counts = {}
        for func in prog.functions.values():
            for op in func.block.operations:
                du = plan.get_compute_device_usage_for_mlprogram_operation(op)
                dev = type(du.preferred_compute_device).__name__ if du else "None"
                counts[dev] = counts.get(dev, 0) + 1
        print("\nPer-op preferred device (CPU_AND_NE):")
        for dev, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {dev:20s} {n}")
    except Exception as e:
        print(f"\n(MLComputePlan breakdown unavailable: {str(e)[:100]})")


if __name__ == "__main__":
    main()
