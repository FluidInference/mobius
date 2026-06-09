#!/usr/bin/env python3
"""Trial 22 Part A: flowlm_step_ane L=512 vs L=256 cache-bucket benchmark.

The flowlm step model marshals its entire KV cache through the MLModel I/O
boundary every call: 12 tensors of [1, L, 16, 64] fp32 in AND out, plus the
small sequence/position/output tensors. At L=512 that is ~25 MB fp32 twice
per call; per-call medians sit at ~3.0-3.7 ms while the compute itself is a
T=1 step over 6 layers. Hypothesis (a): a large share of the median is
boundary marshalling, which an L=256 bucket should roughly halve.

Usage (from models/tts/pocket_tts):
    uv run python coreml/bench_flowlm_lbucket.py \
        [--build-dir coreml/build/english] [--warmup 10] [--iters 200]

Feed: realistic 136-token conditioning prefix in the caches (position=136),
zeros elsewhere (host zero-fill contract), random latent as `sequence`.
NOTE: this graph has NO bos_emb input (see traceable_flowlm_step_ane.py,
docstring item 6).
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import coremltools as ct
import numpy as np

H, D = 16, 64
PREFIX_LEN = 136
NUM_LAYERS = 6


def make_feed(max_seq_len: int, rng: np.random.Generator) -> dict:
    feed = {"sequence": rng.standard_normal((1, 1, 32)).astype(np.float32)}
    for i in range(NUM_LAYERS):
        k = np.zeros((1, max_seq_len, H, D), dtype=np.float32)
        v = np.zeros((1, max_seq_len, H, D), dtype=np.float32)
        k[:, :PREFIX_LEN] = rng.standard_normal((1, PREFIX_LEN, H, D)).astype(np.float32)
        v[:, :PREFIX_LEN] = rng.standard_normal((1, PREFIX_LEN, H, D)).astype(np.float32)
        feed[f"k_cache{i}"] = k
        feed[f"v_cache{i}"] = v
        feed[f"position{i}"] = np.array([float(PREFIX_LEN)], dtype=np.float32)
    return feed


def cache_io_mb(max_seq_len: int) -> float:
    """fp32 bytes crossing the boundary per call (cache in + cache out)."""
    one_side = NUM_LAYERS * 2 * max_seq_len * H * D * 4  # 12 tensors in
    return 2 * one_side / 1e6  # same 12 come back out


def bench(model_path: str, compute_units, warmup: int, iters: int) -> tuple[float, float]:
    model = ct.models.MLModel(model_path, compute_units=compute_units)
    # Read L from the model spec so one harness covers both buckets.
    spec = model.get_spec()
    shape = None
    for inp in spec.description.input:
        if inp.name == "k_cache0":
            shape = list(inp.type.multiArrayType.shape)
    assert shape is not None, f"k_cache0 not found in {model_path}"
    max_seq_len = int(shape[1])

    rng = np.random.default_rng(0)
    feed = make_feed(max_seq_len, rng)

    for _ in range(warmup):
        model.predict(feed)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(feed)
        times.append((time.perf_counter() - t0) * 1000.0)

    med = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18]
    return med, p95


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=os.path.join(here, "build", "english"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    models = [
        ("L=512", os.path.join(args.build_dir, "flowlm_step_ane.mlpackage"), 512),
        ("L=256", os.path.join(args.build_dir, "flowlm_step_ane_l256.mlpackage"), 256),
    ]
    units = [
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("ALL", ct.ComputeUnit.ALL),
    ]

    print(f"warmup={args.warmup}, iters={args.iters}, prefix={PREFIX_LEN} tokens, fp32 IO")
    print(f"{'model':<8s} {'cache IO/call':>14s} {'units':<12s} {'median':>9s} {'p95':>9s}")
    print("-" * 58)
    rows = {}
    for label, path, L in models:
        if not os.path.exists(path):
            print(f"{label:<8s} MISSING: {path}")
            continue
        for uname, cu in units:
            med, p95 = bench(path, cu, args.warmup, args.iters)
            rows[(label, uname)] = (med, p95)
            print(
                f"{label:<8s} {cache_io_mb(L):>11.1f} MB {uname:<12s} "
                f"{med:>7.2f}ms {p95:>7.2f}ms"
            )

    for uname, _ in units:
        a = rows.get(("L=512", uname))
        b = rows.get(("L=256", uname))
        if a and b:
            print(
                f"\nL=256 saves {a[0] - b[0]:+.2f} ms/call median @ {uname} "
                f"({(1 - b[0] / a[0]) * 100:.1f}%) for {cache_io_mb(512) - cache_io_mb(256):.1f} MB "
                f"less boundary IO"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
