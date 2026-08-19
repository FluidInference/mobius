"""Latency + footprint bench for the CoreML Vocoder (compiled mlmodelc).

Usage:
    .venv/bin/python -m coreml.vocoder.bench --coreml-dir build/coreml-vocoder \
        --frames 282 --compute-units CPU_AND_GPU CPU_AND_NE
"""

import argparse
import resource
import time
from pathlib import Path

import coremltools as ct
import numpy as np


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coreml-dir", default="build/coreml-vocoder")
    parser.add_argument("--frames", type=int, default=282)
    parser.add_argument("--compute-units", nargs="+", default=["CPU_AND_GPU", "CPU_AND_NE"],
                        choices=["ALL", "CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    pkg = Path(args.coreml_dir) / "Vocoder.mlpackage"
    mlc = Path(args.coreml_dir) / "Vocoder.mlmodelc"
    if not mlc.exists():
        t0 = time.perf_counter()
        ct.models.utils.compile_model(str(pkg), str(mlc))
        print(f"compiled {mlc} in {time.perf_counter() - t0:.1f} s")

    mel = np.random.default_rng(0).standard_normal((1, 100, args.frames)).astype(np.float32) * 0.5
    gen_seconds = (args.frames - 1) * 512 / 48000.0
    print(f"frames={args.frames} -> {gen_seconds:.3f} s audio @48k; runs={args.runs} warmup={args.warmup}")

    for cu_name in args.compute_units:
        cu = getattr(ct.ComputeUnit, cu_name)
        rss0 = rss_mb()
        t0 = time.perf_counter()
        model = ct.models.CompiledMLModel(str(mlc), compute_units=cu)
        model.predict({"mel": mel})
        load_ms = (time.perf_counter() - t0) * 1e3

        for _ in range(args.warmup):
            model.predict({"mel": mel})
        ts = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model.predict({"mel": mel})
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        print(f"{cu_name:12s}: {ts.mean():7.2f} ± {ts.std():.2f} ms (min {ts.min():.2f})  "
              f"RTFx {gen_seconds * 1e3 / ts.mean():.0f}x  "
              f"load+first {load_ms:.0f} ms  rss +{rss_mb() - rss0:.0f} MB")
        del model


if __name__ == "__main__":
    main()
