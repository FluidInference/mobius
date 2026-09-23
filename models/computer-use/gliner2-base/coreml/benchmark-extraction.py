"""Selected fixed-bucket end-to-end Core ML extraction latency on this Mac."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import coremltools as ct
import psutil
from gliner2 import Schema

from extraction_runtime import CoreMLBoundaryExtractor

UNITS = {
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_and_neural_engine": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(round(fraction * (len(ordered) - 1)), len(ordered) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--feature-package")
    parser.add_argument("--units", choices=list(UNITS), default="all")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    text = "Alice founded Acme in Toronto in 2020."
    schema = Schema().entities(["person", "organization", "location"])
    started = time.perf_counter()
    runtime = CoreMLBoundaryExtractor(
        args.model_dir,
        precision=args.precision,
        compute_units=UNITS[args.units],
        feature_package=args.feature_package,
    )
    load_ms = (time.perf_counter() - started) * 1000
    for _ in range(args.warmup):
        runtime.extract(text, schema)
    process = psutil.Process()
    latencies = []
    peak_rss = process.memory_info().rss
    for _ in range(args.iterations):
        start = time.perf_counter()
        runtime.extract(text, schema)
        latencies.append((time.perf_counter() - start) * 1000)
        peak_rss = max(peak_rss, process.memory_info().rss)
    report = {
        "purpose": "selected end-to-end entity extraction latency, no benchmark scoring",
        "fixture": text,
        "shape": "L128/W64/Q8/C192",
        "precision": args.precision,
        "compute_units": args.units,
        "feature_package": args.feature_package,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_ms": load_ms,
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "mean_ms": statistics.mean(latencies),
        "peak_process_rss_bytes": peak_rss,
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "coremltools": ct.__version__,
    }
    folder = Path(args.model_dir)
    variant = "w8-embedding" if args.feature_package else "baseline"
    path = folder / f"benchmark-{args.precision}-{args.units}-{variant}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
