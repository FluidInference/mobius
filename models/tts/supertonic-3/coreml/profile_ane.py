"""Profile CoreML mlpackage(s) for ANE residency via ct9 MLComputePlan.

Self-contained replacement for the mobius coreml-cli whose uv env cannot load
libmodelpackage. Uses the system coremltools 9.0 (python3.11), which loads
mlpackages fine.

For each mlpackage:
  - compile to .mlmodelc
  - load MLComputePlan with cpu_and_neural_engine
  - tally preferred compute device per mlprogram operation (ANE / GPU / CPU)
  - time a few predicts on cpu_and_neural_engine

The per-op preferred-device tally reflects what CoreML actually planned: if the
ANE compile fails (the opaque ANECompile error 11), ops fall back to CPU and
show up as CPU here. So a high ANE% == the model genuinely landed on ANE.

Usage:
    python3.11 -m coreml.profile_ane build/_mlpackage_ve_quant/*.mlpackage
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.models import compute_plan as cp


def _device_kind(dev) -> str:
    n = type(dev).__name__
    if "Neural" in n:
        return "ANE"
    if "GPU" in n:
        return "GPU"
    return "CPU"


def _walk_ops(block):
    for op in block.operations:
        yield op
        for attr in ("blocks", "block"):
            sub = getattr(op, attr, None)
            if sub is None:
                continue
            subs = sub if isinstance(sub, (list, tuple)) else [sub]
            for b in subs:
                if b is not None and hasattr(b, "operations"):
                    yield from _walk_ops(b)


def profile(mlpackage: Path) -> dict:
    name = mlpackage.stem
    size_mb = sum(f.stat().st_size for f in mlpackage.rglob("*") if f.is_file()) / 1e6

    # Compile to mlmodelc.
    model = ct.models.MLModel(str(mlpackage), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = model.get_compiled_model_path()

    plan = cp.MLComputePlan.load_from_path(
        compiled, compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    prog = plan.model_structure.program
    counts = {"ANE": 0, "GPU": 0, "CPU": 0}
    total = 0
    for fn_name, fn in prog.functions.items():
        for op in _walk_ops(fn.block):
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            kind = _device_kind(usage.preferred_compute_device)
            counts[kind] += 1
            total += 1

    # Build realistic feeds (masks=1, total_step=8, current_step=0).
    spec = model.get_spec()
    feeds = {}
    for inp in spec.description.input:
        shp = [int(d) for d in inp.type.multiArrayType.shape]
        nm = inp.name
        if "mask" in nm:
            feeds[nm] = np.ones(shp, dtype=np.float32)
        elif nm == "total_step":
            feeds[nm] = np.full(shp, 8.0, dtype=np.float32)
        elif nm == "current_step":
            feeds[nm] = np.zeros(shp, dtype=np.float32)
        else:
            feeds[nm] = np.random.randn(*shp).astype(np.float32)

    def _time(mdl) -> float:
        for _ in range(3):
            mdl.predict(feeds)
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            mdl.predict(feeds)
        return (time.perf_counter() - t0) / N * 1000

    pred_ne = _time(model)
    cpu_model = ct.models.MLModel(str(mlpackage), compute_units=ct.ComputeUnit.CPU_ONLY)
    pred_cpu = _time(cpu_model)

    pct = {k: (100.0 * v / total if total else 0.0) for k, v in counts.items()}
    return {
        "name": name,
        "size_mb": size_mb,
        "total_ops": total,
        "counts": counts,
        "pct": pct,
        "predict_ne_ms": pred_ne,
        "predict_cpu_ms": pred_cpu,
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: profile_ane.py <mlpackage> [<mlpackage> ...]")
    print(f"{'variant':<40} {'MB':>7} {'ops':>5} {'ANE%':>6} {'CPU%':>6} {'ne_ms':>7} {'cpu_ms':>7}")
    print("-" * 86)
    for p in paths:
        try:
            r = profile(p)
        except Exception as e:  # noqa: BLE001
            print(f"{p.stem:<40} ERROR: {e}")
            continue
        print(
            f"{r['name']:<40} {r['size_mb']:>7.1f} {r['total_ops']:>5} "
            f"{r['pct']['ANE']:>6.1f} {r['pct']['CPU']:>6.1f} "
            f"{r['predict_ne_ms']:>7.2f} {r['predict_cpu_ms']:>7.2f}"
        )


if __name__ == "__main__":
    main()
