"""nanocodec v3 vs v4 — warm-latency + cold-load bench for the
ship/don't-ship decision after #60 Track 1 ABX confirmed v4 is
acoustically transparent.

Per (variant, CU) config:
- 5 cold loads — fresh `CompiledMLModel(...)` constructions, time the
  wall-clock from before/after. Captures palette dequant cost on v4.
- 60 warm predicts (3 trials × 20 calls) — single load, repeated
  `.predict()`. Reports min/median/mean/p95/max + std.
- ANE residency from `coreml-cli --fallback --json` (one shot).
- Disk size from total-bytes of the mlpackage.

Usage:
    uv run python -m experiments.baseline_fp32.bench_v3_v4 \\
        --v3 build/nanocodec_decoder_v3.mlpackage \\
        --v4 build/v4/nanocodec_decoder_v4.mlpackage \\
        --out-json build/fp32/v3_vs_v4.bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import coremltools as ct

_HERE = Path(__file__).resolve().parent
_COREML_DIR = _HERE.parent.parent
if str(_COREML_DIR) not in sys.path:
    sys.path.insert(0, str(_COREML_DIR))

_COREML_CLI_DIR = ("/Users/kikow/brandon/voicelink/FluidAudio/mobius"
                   "/tools/coreml-cli")
_COREML_CLI = ["uv", "run", "--directory", _COREML_CLI_DIR, "coreml-cli"]

_CU_MAP = {
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "cpu_and_neural_engine": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}

_NUM_CODEBOOKS = 8
_T_IN = 24


def _du_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def _human_bytes(n: int) -> str:
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def _seeded_input() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {"tokens": rng.integers(0, 2024, size=(1, _NUM_CODEBOOKS, _T_IN),
                                   dtype=np.int64).astype(np.int32)}


def _stats(samples: List[float]) -> Dict[str, float]:
    a = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(a.size),
        "min_ms": float(a.min()),
        "median_ms": float(np.median(a)),
        "mean_ms": float(a.mean()),
        "p95_ms": float(np.percentile(a, 95)),
        "max_ms": float(a.max()),
        "std_ms": float(a.std()),
    }


def cold_loads(mlmodelc: Path, cu: str, n: int = 5) -> Dict[str, float]:
    """Fresh-construct CompiledMLModel n times, time each."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        m = ct.models.CompiledMLModel(str(mlmodelc), compute_units=_CU_MAP[cu])
        # Force runtime materialization with a single throwaway predict; the
        # CompiledMLModel constructor lazy-initialises on some compute paths.
        m.predict(_seeded_input())
        times.append((time.perf_counter() - t0) * 1000)
        del m
    return _stats(times)


def warm_predicts(mlmodelc: Path, cu: str,
                  trials: int = 3, calls_per_trial: int = 20) -> Dict[str, float]:
    """One load, then `trials × calls_per_trial` warm predicts."""
    m = ct.models.CompiledMLModel(str(mlmodelc), compute_units=_CU_MAP[cu])
    feed = _seeded_input()
    # Warmup: 4 throwaways before timed iterations to flush kernel caches.
    for _ in range(4):
        m.predict(feed)
    samples = []
    for _ in range(trials):
        for _ in range(calls_per_trial):
            t0 = time.perf_counter()
            m.predict(feed)
            samples.append((time.perf_counter() - t0) * 1000)
    return _stats(samples)


def fallback_residency(mlmodelc: Path) -> Dict[str, Any]:
    """coreml-cli --fallback (cpu_and_neural_engine implied)."""
    abspath = str(mlmodelc.resolve())
    proc = subprocess.run(
        _COREML_CLI + [abspath, "--fallback", "--json",
                       "--plan-timeout", "600"],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"coreml-cli failed:\n{proc.stderr}")
    fb = json.loads(proc.stdout)["models"][0]["fallback"]
    return {
        "total_ops": int(fb["total_ops"]),
        "ane_ops": int(fb["ane_ops"]),
        "cpu_ops": int(fb["cpu_ops"]),
        "gpu_ops": int(fb.get("gpu_ops", 0)),
        "ane_pct": float(fb["ane_percent"]),
    }


def _coreml_cli_summary_pct(mlmodelc: Path, cu: str) -> Dict[str, float]:
    """Get the planner-actual placement summary for a specific CU policy."""
    abspath = str(mlmodelc.resolve())
    proc = subprocess.run(
        _COREML_CLI + [abspath, "--units", cu, "--iterations", "1",
                       "--json", "--plan-timeout", "600"],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"coreml-cli --units {cu} failed:\n{proc.stderr}")
    res = next(r for r in json.loads(proc.stdout)["models"][0]["results"]
               if r["compute_units"] == cu)
    return {"ane_pct": float(res["summary"]["ane_percent"]),
            "gpu_pct": float(res["summary"]["gpu_percent"]),
            "cpu_pct": float(res["summary"]["cpu_percent"]),
            "compile_ms": float(res["latency"]["compile_ms"])}


def bench_variant(label: str, mlpackage: Path, mlmodelc: Path) -> Dict[str, Any]:
    print(f"[bench] === {label}: {mlpackage.name} ===", file=sys.stderr)
    size = _du_bytes(mlpackage)
    fb = fallback_residency(mlmodelc)
    summary: Dict[str, Any] = {
        "label": label,
        "mlpackage": str(mlpackage),
        "size_bytes": size,
        "size_human": _human_bytes(size),
        "fallback_residency": fb,
        "by_cu": {},
    }
    for cu in ("cpu_only", "cpu_and_neural_engine", "all"):
        print(f"[bench]   CU={cu}", file=sys.stderr)
        cold = cold_loads(mlmodelc, cu, n=5)
        print(f"            cold load (n=5): "
              f"med={cold['median_ms']:.1f} ms "
              f"min={cold['min_ms']:.1f} max={cold['max_ms']:.1f}",
              file=sys.stderr)
        warm = warm_predicts(mlmodelc, cu, trials=3, calls_per_trial=20)
        print(f"            warm (n=60): "
              f"med={warm['median_ms']:.2f} ms "
              f"p95={warm['p95_ms']:.2f} std={warm['std_ms']:.2f}",
              file=sys.stderr)
        plan = _coreml_cli_summary_pct(mlmodelc, cu)
        summary["by_cu"][cu] = {
            "cold_load_ms": cold,
            "warm_predict_ms": warm,
            "planner_placement_pct": plan,
        }
    return summary


def _md_row(label: str, s: Dict[str, Any]) -> str:
    fb = s["fallback_residency"]
    cu_only = s["by_cu"]["cpu_only"]["warm_predict_ms"]
    cu_ne = s["by_cu"]["cpu_and_neural_engine"]["warm_predict_ms"]
    cu_all = s["by_cu"]["all"]["warm_predict_ms"]
    cold_ne = s["by_cu"]["cpu_and_neural_engine"]["cold_load_ms"]
    return (f"| {label} | {s['size_human']} | "
            f"{fb['ane_pct']:.1f}% (ANE {fb['ane_ops']}/{fb['total_ops']}) | "
            f"{cold_ne['median_ms']:.0f} ms | "
            f"{cu_only['median_ms']:.2f} ms | "
            f"{cu_ne['median_ms']:.2f} ms | "
            f"{cu_all['median_ms']:.2f} ms |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--v3", required=True, type=Path)
    ap.add_argument("--v4", required=True, type=Path)
    ap.add_argument("--mlmodelc-dir", type=Path,
                    default=Path("/tmp/v3v4-bench"),
                    help="Where compiled .mlmodelc files live (or will land).")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    args.mlmodelc_dir.mkdir(parents=True, exist_ok=True)
    for src in (args.v3, args.v4):
        target = args.mlmodelc_dir / src.with_suffix(".mlmodelc").name
        if not target.exists():
            print(f"[bench] compiling {src.name} → {target}", file=sys.stderr)
            proc = subprocess.run(
                ["xcrun", "coremlcompiler", "compile",
                 str(src), str(args.mlmodelc_dir)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"coremlcompiler failed: {proc.stderr}")

    v3_bench = bench_variant(
        "v3", args.v3,
        args.mlmodelc_dir / args.v3.with_suffix(".mlmodelc").name,
    )
    v4_bench = bench_variant(
        "v4", args.v4,
        args.mlmodelc_dir / args.v4.with_suffix(".mlmodelc").name,
    )

    summary = {"v3": v3_bench, "v4": v4_bench}

    print()
    print("| variant | size | ANE % @ best | cold load (cpu+ne, n=5 med) | warm @ cpuOnly | warm @ cpu+ne | warm @ all |")
    print("|---|---|---|---|---|---|---|")
    print(_md_row("v3", v3_bench))
    print(_md_row("v4", v4_bench))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2))
        print(f"\n[bench] wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
