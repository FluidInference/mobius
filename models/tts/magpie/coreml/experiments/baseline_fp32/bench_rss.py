"""nanocodec v3 vs v4 — RSS / memory-footprint bench.

Settles whether v4's 4× on-disk savings (121 → 31 MB) carries through
to runtime RAM. Hypothesis: palette dequant happens at MLModel load
time, so weights expand to fp32 in RSS — making the on-disk win
disappear at runtime.

Per (variant, CU) — measured in a fresh subprocess so the baseline RSS
is clean:

  1. Baseline RSS before any CoreML import or load.
  2. Peak RSS during cold load — background thread samples
     `psutil.Process.memory_info().rss` every 10 ms during
     `CompiledMLModel(...)`. Captures whether palette + dequant briefly
     hold both compressed-form AND fp32 weights simultaneously.
  3. Steady-state RSS post-load — wait 2 s for any deferred allocations
     to settle, then median across 5 samples (50 ms apart).
  4. Steady-state RSS during sustained inference — 60-call warm loop,
     concurrent 100 ms polling thread, report median + p95 across all
     samples taken during the loop.

Driver mode (default): runs all 2 × 3 = 6 (variant, CU) configs as
subprocesses, aggregates into a Markdown table.

Subprocess mode (--measure): used internally — performs one
(variant, CU) measurement, prints JSON to stdout.

Usage:
    uv run python -m experiments.baseline_fp32.bench_rss \\
        --v3 build/nanocodec_decoder_v3.mlpackage \\
        --v4 build/v4/nanocodec_decoder_v4.mlpackage \\
        --mlmodelc-dir /tmp/v3v4-bench \\
        --out-json build/fp32/v3_vs_v4.rss.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

# Late import — psutil and coremltools are loaded only inside the
# measurement subprocess so the driver process stays small and the
# baseline read in the subprocess is clean.

_HERE = Path(__file__).resolve().parent

_CU_NAMES = ("cpu_only", "cpu_and_neural_engine", "all")


def _coreml_units(name: str):
    import coremltools as ct
    return {
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_neural_engine": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }[name]


def _seeded_input():
    import numpy as np
    rng = np.random.default_rng(42)
    return {"tokens": rng.integers(0, 2024, size=(1, 8, 24),
                                   dtype=np.int64).astype(np.int32)}


# ──────────────── poller ────────────────


class _RssPoller:
    """Background thread that records `process.memory_info().rss` every
    `interval_ms`. Use as a context manager."""

    def __init__(self, interval_ms: float):
        self.interval = interval_ms / 1000.0
        self.samples: List[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        import psutil
        proc = psutil.Process()
        self.samples.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    self.samples.append(proc.memory_info().rss)
                except Exception:
                    return
                time.sleep(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def _human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


# ──────────────── single-config measurement ────────────────


def _measure_one(mlmodelc: Path, cu: str) -> Dict[str, Any]:
    """Run baseline → cold-load → post-load → warm-loop measurement
    and return a JSON-serialisable dict.

    Imports coremltools / psutil / numpy *after* the baseline read so
    the baseline reflects pure interpreter overhead.
    """
    import psutil
    proc = psutil.Process()
    baseline_rss = proc.memory_info().rss

    # Heavy imports.
    import coremltools as ct  # noqa: F401  -- triggers fork-friendly init
    import numpy as np  # noqa: F401

    # 1) Cold load with 10 ms RSS polling.
    units = _coreml_units(cu)
    feed = _seeded_input()
    with _RssPoller(interval_ms=10) as cold_poller:
        t0 = time.perf_counter()
        model = ct.models.CompiledMLModel(str(mlmodelc), compute_units=units)
        # Force runtime materialization (CompiledMLModel constructors are
        # lazy on some compute paths). One throwaway predict.
        model.predict(feed)
        cold_load_ms = (time.perf_counter() - t0) * 1000

    cold_samples = cold_poller.samples
    peak_rss = max(cold_samples) if cold_samples else proc.memory_info().rss

    # 2) Settle.
    time.sleep(2.0)

    # 3) Post-load steady state — median across 5 samples 50 ms apart.
    post_samples: List[int] = []
    for _ in range(5):
        post_samples.append(proc.memory_info().rss)
        time.sleep(0.05)
    post_rss_med = int(sorted(post_samples)[len(post_samples) // 2])

    # 4) Sustained inference — 60 calls + 100 ms RSS polling.
    with _RssPoller(interval_ms=100) as warm_poller:
        for _ in range(60):
            model.predict(feed)
    warm_samples = warm_poller.samples or [proc.memory_info().rss]

    warm_arr = sorted(warm_samples)
    warm_median = warm_arr[len(warm_arr) // 2]
    warm_p95 = warm_arr[int(0.95 * (len(warm_arr) - 1))]
    warm_max = warm_arr[-1]

    return {
        "mlmodelc": str(mlmodelc),
        "cu": cu,
        "baseline_rss_bytes": int(baseline_rss),
        "peak_load_rss_bytes": int(peak_rss),
        "cold_load_n_samples": len(cold_samples),
        "cold_load_ms": cold_load_ms,
        "post_load_rss_bytes": int(post_rss_med),
        "warm_inference": {
            "n_samples": len(warm_samples),
            "median_rss_bytes": int(warm_median),
            "p95_rss_bytes": int(warm_p95),
            "max_rss_bytes": int(warm_max),
        },
        "deltas_vs_baseline_bytes": {
            "peak_load": int(peak_rss - baseline_rss),
            "post_load": int(post_rss_med - baseline_rss),
            "warm_median": int(warm_median - baseline_rss),
            "warm_p95": int(warm_p95 - baseline_rss),
        },
    }


# ──────────────── driver ────────────────


def _run_subprocess(mlmodelc: Path, cu: str) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "experiments.baseline_fp32.bench_rss",
           "--measure", "--mlmodelc", str(mlmodelc), "--cu", cu]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=600,
                          cwd=str(_HERE.parent.parent),
                          env={**os.environ, "PYTHONPATH": str(_HERE.parent.parent)})
    if proc.returncode != 0:
        raise RuntimeError(
            f"measurement subprocess failed for ({mlmodelc.name}, {cu}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    last_line = proc.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def _ensure_compiled(src: Path, mlmodelc_dir: Path) -> Path:
    target = mlmodelc_dir / src.with_suffix(".mlmodelc").name
    if target.exists():
        return target
    mlmodelc_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rss] compiling {src.name} → {target}", file=sys.stderr)
    proc = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(src), str(mlmodelc_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed: {proc.stderr}")
    return target


def _md_row(label: str, cu: str, m: Dict[str, Any]) -> str:
    d = m["deltas_vs_baseline_bytes"]
    base = m["baseline_rss_bytes"]
    return (
        f"| {label} | {cu} | {_human_mb(base)} | "
        f"+{_human_mb(d['peak_load'])} | "
        f"+{_human_mb(d['post_load'])} | "
        f"+{_human_mb(d['warm_median'])} | "
        f"+{_human_mb(d['warm_p95'])} |"
    )


def _savings_row(cu: str, v3: Dict[str, Any], v4: Dict[str, Any]) -> str:
    """v3 - v4 in bytes for each delta — positive = v4 saves RAM."""
    fields = ("peak_load", "post_load", "warm_median", "warm_p95")
    parts = [f"| **savings v4 vs v3** | {cu} | — |"]
    for f in fields:
        diff = v3["deltas_vs_baseline_bytes"][f] - v4["deltas_vs_baseline_bytes"][f]
        sign = "−" if diff > 0 else "+"
        parts.append(f" {sign}{_human_mb(abs(diff))} |")
    return "".join(parts)


def driver(args) -> int:
    mlmodelc_dir = args.mlmodelc_dir
    v3_mlmodelc = _ensure_compiled(args.v3, mlmodelc_dir)
    v4_mlmodelc = _ensure_compiled(args.v4, mlmodelc_dir)

    results: Dict[str, Dict[str, Any]] = {"v3": {}, "v4": {}}
    for variant_label, mlmodelc in (("v3", v3_mlmodelc), ("v4", v4_mlmodelc)):
        for cu in _CU_NAMES:
            print(f"[rss] === {variant_label} / {cu} ===", file=sys.stderr)
            m = _run_subprocess(mlmodelc, cu)
            results[variant_label][cu] = m
            print(f"      baseline   = {_human_mb(m['baseline_rss_bytes'])}",
                  file=sys.stderr)
            print(f"      Δ peak     = +{_human_mb(m['deltas_vs_baseline_bytes']['peak_load'])}",
                  file=sys.stderr)
            print(f"      Δ post     = +{_human_mb(m['deltas_vs_baseline_bytes']['post_load'])}",
                  file=sys.stderr)
            print(f"      Δ warm med = +{_human_mb(m['deltas_vs_baseline_bytes']['warm_median'])}",
                  file=sys.stderr)
            print(f"      Δ warm p95 = +{_human_mb(m['deltas_vs_baseline_bytes']['warm_p95'])}",
                  file=sys.stderr)

    # Markdown table.
    print()
    print("| variant | CU | baseline | Δ peak load | Δ post-load | Δ warm median | Δ warm p95 |")
    print("|---|---|---|---|---|---|---|")
    for cu in _CU_NAMES:
        print(_md_row("v3", cu, results["v3"][cu]))
        print(_md_row("v4", cu, results["v4"][cu]))
    print()
    print("| variant | CU | — | savings (peak) | savings (post) | savings (warm med) | savings (warm p95) |")
    print("|---|---|---|---|---|---|---|")
    for cu in _CU_NAMES:
        print(_savings_row(cu, results["v3"][cu], results["v4"][cu]))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2))
        print(f"\n[rss] wrote {args.out_json}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--measure", action="store_true",
                    help="Subprocess mode — run one (variant, CU) "
                         "measurement and emit JSON.")
    ap.add_argument("--mlmodelc", type=Path, default=None,
                    help="(--measure) path to compiled .mlmodelc")
    ap.add_argument("--cu", choices=_CU_NAMES, default=None,
                    help="(--measure) compute unit policy")
    ap.add_argument("--v3", type=Path, default=None)
    ap.add_argument("--v4", type=Path, default=None)
    ap.add_argument("--mlmodelc-dir", type=Path,
                    default=Path("/tmp/v3v4-bench"))
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    if args.measure:
        if args.mlmodelc is None or args.cu is None:
            ap.error("--measure requires --mlmodelc and --cu")
        result = _measure_one(args.mlmodelc, args.cu)
        # Emit single-line JSON so the driver can pick it up reliably.
        print(json.dumps(result))
        return 0

    if args.v3 is None or args.v4 is None:
        ap.error("driver mode requires --v3 and --v4")
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
