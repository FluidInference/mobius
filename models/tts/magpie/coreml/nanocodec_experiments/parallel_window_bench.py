"""Profile parallel sliding-window decode of nanocodec_decoder_v3.

The Swift `MagpieNanocodec.decode()` runs ~20 sequential 24-frame window
calls per ~8s utterance, all on a single CPU core. The window iterations
are data-independent (each reads its own input slice, writes its own
output slice). Question: does parallelizing the calls across multiple CPU
cores give a real speedup, or are we already compute/bandwidth-bound?

Test:
  * Load nanocodec_decoder_v3.mlmodelc (CPU-only, fp32).
  * Fabricate input: 8 codebooks × 24 frames of int32 codes.
  * Run N independent prediction() calls:
      - sequential (one thread)
      - 2-way / 4-way / 8-way concurrent (ThreadPoolExecutor)
  * Report wall-clock + per-call ms.

Usage:
    uv run python parallel_window_bench.py \
        --model ../compiled/build/nanocodec_decoder_v3.mlmodelc \
        --calls 20 --warmup 4

Notes:
  * Uses the *compiled* mlmodelc (matches Swift runtime path).
  * coremltools.models.MLModel under the hood = same Apple CoreML stack
    Swift uses, so timings are representative.
  * `compute_units = ComputeUnit.CPU_ONLY` matches MagpieModelStore's
    forced policy for nanocodec.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import coremltools as ct


def make_input(num_codebooks: int = 8, t_in: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    # nanocodec codes are int32 in [0, 1024).
    arr = rng.integers(0, 1024, size=(1, num_codebooks, t_in), dtype=np.int32)
    return arr


def run_one(model: ct.models.MLModel, tokens: np.ndarray) -> float:
    t0 = time.perf_counter()
    _ = model.predict({"tokens": tokens})
    return time.perf_counter() - t0


def run_sequential(model, tokens_list: List[np.ndarray]) -> tuple[float, list[float]]:
    per_call: list[float] = []
    t0 = time.perf_counter()
    for toks in tokens_list:
        per_call.append(run_one(model, toks))
    elapsed = time.perf_counter() - t0
    return elapsed, per_call


def run_parallel(
    model, tokens_list: List[np.ndarray], workers: int
) -> tuple[float, list[float]]:
    """Run all calls through a ThreadPoolExecutor with `workers` threads.

    Reuses one shared model instance — Apple CoreML MLModel.predict()
    is documented thread-safe for concurrent calls.
    """
    per_call: list[float] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, model, toks) for toks in tokens_list]
        for f in futs:
            per_call.append(f.result())
    elapsed = time.perf_counter() - t0
    return elapsed, per_call


def fmt(times: list[float]) -> str:
    if not times:
        return "—"
    ms = [t * 1000 for t in times]
    return (
        f"min={min(ms):.0f}ms "
        f"median={statistics.median(ms):.0f}ms "
        f"max={max(ms):.0f}ms"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="../compiled/build/nanocodec_decoder_v3.mlmodelc",
        help="Path to compiled .mlmodelc",
    )
    p.add_argument("--calls", type=int, default=20, help="Calls per trial (~20 = one 8s utterance)")
    p.add_argument("--warmup", type=int, default=4, help="Warmup calls before timing")
    p.add_argument("--cool-secs", type=int, default=15, help="Cool-down between trials")
    p.add_argument("--workers", default="1,2,4,8", help="Comma-separated worker counts to test")
    p.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="Repeat each (workers) config this many times for variance check",
    )
    args = p.parse_args()

    model_path = os.path.abspath(args.model)
    print(f"Model: {model_path}")
    print(f"Calls per trial: {args.calls}")
    print(f"Warmup: {args.warmup}")
    print(f"Cooldown between trials: {args.cool_secs}s")
    print(f"Cycles per config: {args.cycles}")
    print()

    print("Loading mlmodelc (CPU_ONLY)...", flush=True)
    model = ct.models.MLModel(
        model_path, compute_units=ct.ComputeUnit.CPU_ONLY
    )
    print("Loaded.", flush=True)

    # Fabricate distinct inputs so we don't accidentally cache.
    tokens_list = [make_input(seed=i) for i in range(args.calls)]
    warmup_list = [make_input(seed=999 - i) for i in range(args.warmup)]

    print(f"\nWarming up ({args.warmup} calls)...", flush=True)
    _ = run_sequential(model, warmup_list)
    print("Warmup done.\n", flush=True)

    workers_list = [int(x) for x in args.workers.split(",")]

    results: list[dict] = []

    for w in workers_list:
        for cycle in range(args.cycles):
            tag = f"workers={w} cycle={cycle+1}/{args.cycles}"
            print(f"=== {tag} ===", flush=True)
            print(f"  cool {args.cool_secs}s", flush=True)
            time.sleep(args.cool_secs)
            if w == 1:
                elapsed, per_call = run_sequential(model, tokens_list)
            else:
                elapsed, per_call = run_parallel(model, tokens_list, workers=w)
            results.append(
                dict(
                    workers=w,
                    cycle=cycle,
                    elapsed=elapsed,
                    per_call=per_call,
                )
            )
            print(
                f"  total={elapsed*1000:.0f}ms  "
                f"per-call: {fmt(per_call)}  "
                f"(n={len(per_call)})",
                flush=True,
            )

    print("\n=== Summary ===", flush=True)
    print(
        f"{'workers':>8} {'cycle':>5} {'total ms':>10} "
        f"{'per-call min':>14} {'per-call med':>14} {'per-call max':>14} "
        f"{'speedup':>8}"
    )
    baseline = next(
        (r["elapsed"] for r in results if r["workers"] == 1 and r["cycle"] == 0), None
    )
    for r in results:
        ms = [t * 1000 for t in r["per_call"]]
        speedup = (baseline / r["elapsed"]) if baseline else 1.0
        print(
            f"{r['workers']:>8d} {r['cycle']+1:>5d} "
            f"{r['elapsed']*1000:>10.0f} "
            f"{min(ms):>14.0f} {statistics.median(ms):>14.0f} "
            f"{max(ms):>14.0f} {speedup:>8.2f}x"
        )


if __name__ == "__main__":
    main()
