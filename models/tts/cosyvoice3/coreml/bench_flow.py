#!/usr/bin/env python3
"""Benchmark Flow under every CoreML compute-unit setting.

Runs each Flow variant with random-but-valid inputs and measures per-iteration
latency, checking for NaN. Prints each row as soon as it completes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct


def log(msg: str = "") -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def make_inputs(rng: np.random.Generator) -> dict:
    return {
        "token_total": rng.integers(0, 6561, size=(1, 250), dtype=np.int32),
        "num_prompt_tokens": np.array([50], dtype=np.int32),
        "prompt_feat": rng.standard_normal((1, 500, 80)).astype(np.float32) * 0.5,
        "embedding": rng.standard_normal((1, 192)).astype(np.float32),
    }


def bench(
    model_path: Path,
    compute_units: ct.ComputeUnit,
    n_warmup: int,
    n_iter: int,
) -> dict:
    rng = np.random.default_rng(42)
    t_load0 = time.perf_counter()
    model = ct.models.MLModel(str(model_path), compute_units=compute_units)
    t_load = time.perf_counter() - t_load0

    # Warmup.
    for _ in range(n_warmup):
        _ = model.predict(make_inputs(rng))

    # Timed.
    latencies = []
    nan_count = 0
    for _ in range(n_iter):
        inp = make_inputs(rng)
        t0 = time.perf_counter()
        out = model.predict(inp)
        latencies.append((time.perf_counter() - t0) * 1000)
        if np.isnan(out["mel"]).any():
            nan_count += 1

    return {
        "load_ms": t_load * 1000,
        "p50_ms": float(np.percentile(latencies, 50)),
        "mean_ms": float(np.mean(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "nan_count": nan_count,
        "n_iter": n_iter,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iter", type=int, default=3)
    args = ap.parse_args()

    # (variant_name, model_path, [configs_to_test])
    # fp16 variants only need ANE-eligible configs — we know cpuOnly works for fp32.
    configs_all = [
        ("cpuOnly", ct.ComputeUnit.CPU_ONLY),
        ("cpuAndGPU", ct.ComputeUnit.CPU_AND_GPU),
        ("cpuAndNE", ct.ComputeUnit.CPU_AND_NE),
        ("all", ct.ComputeUnit.ALL),
    ]
    configs_ane = [
        ("cpuAndNE", ct.ComputeUnit.CPU_AND_NE),
        ("all", ct.ComputeUnit.ALL),
    ]

    runs = [
        ("fp32", Path("build/flow-fp32-n250/Flow-N250-fp32.mlpackage"), configs_all),
        ("fp16", Path("build/flow-fp16-n250/Flow-N250-fp16.mlpackage"), configs_ane),
        ("fp16v2", Path("build/flow-fp16v2-n250/Flow-N250-fp16.mlpackage"), configs_ane),
    ]

    header = (
        f"{'variant':<8} {'config':<10} {'load_ms':>9} {'p50_ms':>9} "
        f"{'mean_ms':>9} {'min_ms':>9} {'max_ms':>9} {'nan':>7}"
    )
    log(header)
    log("-" * len(header))

    for vname, vpath, configs in runs:
        if not vpath.exists():
            log(f"{vname:<8} SKIP — {vpath} not found")
            continue
        for cname, cu in configs:
            log(f"{vname:<8} {cname:<10} ... (running)")
            try:
                r = bench(vpath, cu, args.warmup, args.iter)
                log(
                    f"{vname:<8} {cname:<10} {r['load_ms']:>9.0f} {r['p50_ms']:>9.0f} "
                    f"{r['mean_ms']:>9.0f} {r['min_ms']:>9.0f} {r['max_ms']:>9.0f} "
                    f"{r['nan_count']:>3}/{r['n_iter']}"
                )
            except Exception as exc:
                log(f"{vname:<8} {cname:<10} ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
