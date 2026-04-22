#!/usr/bin/env python3
"""Run a single Flow (variant, compute-unit) benchmark.

Separated from bench_flow.py so a hang in one row can be timeout-killed
without losing data from prior rows.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cu", required=True,
                    choices=["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iter", type=int, default=3)
    args = ap.parse_args()

    cu_map = {
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuAndNE": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }
    cu = cu_map[args.cu]

    rng = np.random.default_rng(42)
    log(f"LOAD {args.cu} ...")
    t0 = time.perf_counter()
    model = ct.models.MLModel(args.model, compute_units=cu)
    load_ms = (time.perf_counter() - t0) * 1000
    log(f"LOAD_MS {load_ms:.0f}")

    log(f"WARMUP {args.warmup} ...")
    for _ in range(args.warmup):
        _ = model.predict(make_inputs(rng))

    lats = []
    nans = 0
    for i in range(args.iter):
        inp = make_inputs(rng)
        t1 = time.perf_counter()
        out = model.predict(inp)
        dt = (time.perf_counter() - t1) * 1000
        lats.append(dt)
        if np.isnan(out["mel"]).any():
            nans += 1
        log(f"ITER {i} ms={dt:.0f} nan={int(np.isnan(out['mel']).any())}")

    p50 = float(np.percentile(lats, 50))
    log(f"RESULT load_ms={load_ms:.0f} p50={p50:.0f} "
        f"mean={np.mean(lats):.0f} min={np.min(lats):.0f} "
        f"max={np.max(lats):.0f} nan={nans}/{args.iter}")


if __name__ == "__main__":
    main()
