"""Interleaved A/B benchmark: fused 8-step VE (1 predict) vs 8-call loop.

Methodology (mobius Trial 15): A and B alternate within one process so
thermal / scheduler drift hits both equally. 10 warmup pairs, then N timed
pairs; report median / p95.

The 8-call loop is timed exactly as the Swift host pays it: 8 separate
predict() invocations with denoised->noisy feedback (feeds rebuilt per
step, like Supertonic3Synthesizer rebuilding current_step each iteration).

Usage:
    python3.11 -m coreml.bench_fused SINGLE_STEP_MODEL FUSED_MODEL \
        [--units CPU_AND_NE ALL] [--n 200] [--steps 8]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np

L_DIM = 144

UNIT_MAP = {
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "ALL": ct.ComputeUnit.ALL,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
}


def _load(path: Path, units):
    if path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(path), compute_units=units)
    return ct.models.MLModel(str(path), compute_units=units)


def _inputs(L: int, T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "noisy_latent": rng.standard_normal((1, L_DIM, L)).astype(np.float32),
        "text_emb": rng.standard_normal((1, 256, T)).astype(np.float32),
        "style_ttl": rng.standard_normal((1, 50, 256)).astype(np.float32),
        "latent_mask": np.ones((1, 1, L), dtype=np.float32),
        "text_mask": np.ones((1, 1, T), dtype=np.float32),
    }


def _time_loop(model, feeds: dict, steps: int) -> float:
    """One full 8-step denoising loop via 8 separate predict() calls (host path)."""
    total = np.full((1,), float(steps), dtype=np.float32)
    noisy = feeds["noisy_latent"]
    t0 = time.perf_counter()
    for k in range(steps):
        out = model.predict({
            **feeds,
            "noisy_latent": noisy,
            "current_step": np.full((1,), float(k), dtype=np.float32),
            "total_step": total,
        })
        noisy = out["denoised_latent"]
    return (time.perf_counter() - t0) * 1000


def _time_fused(model, feeds: dict) -> float:
    t0 = time.perf_counter()
    model.predict(feeds)
    return (time.perf_counter() - t0) * 1000


def _stats(xs):
    a = np.sort(np.asarray(xs))
    return np.median(a), a[int(0.95 * (len(a) - 1))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("single", type=Path)
    p.add_argument("fused", type=Path)
    p.add_argument("--units", nargs="+", default=["CPU_AND_NE", "ALL"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--T", type=int, default=128)
    args = p.parse_args()

    feeds = _inputs(args.L, args.T)

    print(f"single: {args.single.name}   fused: {args.fused.name}")
    print(f"interleaved, warmup={args.warmup} pairs, timed={args.n} pairs, steps={args.steps}\n")
    print(f"{'units':<12} {'config':<22} {'median_ms':>10} {'p95_ms':>10} {'speedup':>8}")
    print("-" * 68)

    for unit_name in args.units:
        units = UNIT_MAP[unit_name]
        single = _load(args.single, units)
        fused = _load(args.fused, units)

        for _ in range(args.warmup):
            _time_loop(single, feeds, args.steps)
            _time_fused(fused, feeds)

        loop_ts, fused_ts = [], []
        for _ in range(args.n):
            loop_ts.append(_time_loop(single, feeds, args.steps))
            fused_ts.append(_time_fused(fused, feeds))

        lm, lp = _stats(loop_ts)
        fm, fp_ = _stats(fused_ts)
        print(f"{unit_name:<12} {'8-call loop':<22} {lm:>10.2f} {lp:>10.2f} {'1.00x':>8}")
        print(f"{unit_name:<12} {'fused 1-call':<22} {fm:>10.2f} {fp_:>10.2f} {lm / fm:>7.2f}x")
        del single, fused


if __name__ == "__main__":
    main()
