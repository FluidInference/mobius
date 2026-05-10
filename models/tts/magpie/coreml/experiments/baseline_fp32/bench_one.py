"""Bench + parity-check one fp32 mlpackage vs the production fp16 mlmodelc.

Mirrors the per-stage shape used by styletts2's `coreml/parity.py` but
limited to the comparison the BASELINE_FP32 table needs:

- Final compute-unit assignment + ANE % via coreml-cli (--fallback --json)
- Warm latency at .cpuAndNeuralEngine and .cpuOnly (3 runs after warmup)
- Cosine + max|delta| of the fp32 mlpackage vs the production fp16
  .mlmodelc on a single representative input

Stage-specific feed builders live in `inputs.py` so each row stays a
one-liner here.

Usage:
    uv run python -m experiments.baseline_fp32.bench_one \\
        --stage text_encoder \\
        --fp32-package build/fp32/text_encoder_fp32.mlpackage \\
        --prod-mlmodelc ~/.cache/fluidaudio/Models/magpie-tts/text_encoder.mlmodelc

Output: a single Markdown table row written to stdout (and JSON to
``build/fp32/<stage>.bench.json``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import coremltools as ct

# Allow running as a script (python bench_one.py …) and as a module
# (-m experiments.baseline_fp32.bench_one).
_HERE = Path(__file__).resolve().parent
_COREML_DIR = _HERE.parent.parent
if str(_COREML_DIR) not in sys.path:
    sys.path.insert(0, str(_COREML_DIR))

from experiments.baseline_fp32.inputs import build_inputs


_COREML_CLI = ["uv", "run", "--directory",
               str(Path("/Users/kikow/brandon/voicelink/FluidAudio/mobius/tools/coreml-cli")),
               "coreml-cli"]


def _du_bytes(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
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


def coreml_cli_fallback(mlmodelc: Path, plan_timeout: float = 600.0) -> dict[str, Any]:
    """Call coreml-cli --fallback --json on an .mlmodelc — returns parsed dict."""
    abspath = str(mlmodelc.resolve())
    proc = subprocess.run(
        _COREML_CLI + [abspath, "--fallback", "--json",
                       "--plan-timeout", str(plan_timeout)],
        capture_output=True, text=True, timeout=plan_timeout * 4,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"coreml-cli failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return json.loads(proc.stdout)


def coreml_cli_bench(mlmodelc: Path, units: str, n: int = 3,
                     plan_timeout: float = 600.0) -> dict[str, Any]:
    """Call coreml-cli at a specific compute-unit policy. Returns parsed JSON."""
    abspath = str(mlmodelc.resolve())
    proc = subprocess.run(
        _COREML_CLI + [abspath, "--units", units, "--json",
                       "--iterations", str(n),
                       "--plan-timeout", str(plan_timeout)],
        capture_output=True, text=True, timeout=plan_timeout * 4,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"coreml-cli failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return json.loads(proc.stdout)


def _model_entry(j: dict[str, Any]) -> dict[str, Any]:
    """coreml-cli output is `{hardware: …, models: [<one entry>, …]}`."""
    return j["models"][0]


def _aggregate_fallback(j: dict[str, Any]) -> tuple[int, int, dict[str, int]]:
    """From coreml-cli --fallback JSON: total ops, ANE ops, per-device breakdown.

    The fallback section reports cpu_and_neural_engine placement (ANE-vs-CPU
    fallback). GPU ops aren't separately tracked here — they show up only in
    the `--units` bench summaries.
    """
    fb = _model_entry(j)["fallback"]
    total = int(fb["total_ops"])
    ane = int(fb["ane_ops"])
    gpu = int(fb.get("gpu_ops", 0))
    cpu = int(fb.get("cpu_ops", max(total - ane - gpu, 0)))
    return total, ane, {"ane": ane, "gpu": gpu, "cpu": cpu}


def _bench_result(j: dict[str, Any], units: str) -> dict[str, Any] | None:
    for r in _model_entry(j)["results"]:
        if r["compute_units"] == units:
            return r
    return None


def _median_predict_ms(j: dict[str, Any], units: str) -> float:
    r = _bench_result(j, units)
    if r is None:
        return float("nan")
    return float(r["latency"]["median_ms"])


def _summary_pct(j: dict[str, Any], units: str) -> dict[str, float]:
    r = _bench_result(j, units)
    if r is None:
        return {}
    s = r["summary"]
    return {"ane_pct": float(s["ane_percent"]),
            "gpu_pct": float(s["gpu_percent"]),
            "cpu_pct": float(s["cpu_percent"])}


def cosine_and_max_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom else float("nan")
    return cos, float(np.max(np.abs(a - b)))


def _load_for_predict(path: Path):
    """Load either a .mlpackage (MLModel) or compiled .mlmodelc (CompiledMLModel)."""
    p = str(path)
    if path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(p, compute_units=ct.ComputeUnit.CPU_ONLY)
    return ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_ONLY)


def parity_check(fp32_pkg: Path, prod_path: Path,
                 stage: str) -> dict[str, Any]:
    """Run identical input through both, return per-output cosine + max|delta|."""
    feed = build_inputs(stage)
    fp32 = _load_for_predict(fp32_pkg)
    prod = _load_for_predict(prod_path)
    fp32_out = fp32.predict(feed)
    prod_out = prod.predict(feed)
    common = sorted(set(fp32_out) & set(prod_out))
    metrics = {}
    for k in common:
        a = np.asarray(fp32_out[k])
        b = np.asarray(prod_out[k])
        if a.shape != b.shape:
            metrics[k] = {"shape_mismatch": [list(a.shape), list(b.shape)]}
            continue
        cos, mxd = cosine_and_max_delta(a, b)
        metrics[k] = {"cosine": cos, "max_abs_delta": mxd, "shape": list(a.shape)}
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["text_encoder", "decoder_prefill", "decoder_step",
                             "local_transformer", "nanocodec_decoder_v3"])
    ap.add_argument("--fp32-package", required=True, type=Path)
    ap.add_argument("--prod-mlmodelc", type=Path, default=None,
                    help="If omitted, parity is skipped.")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--plan-timeout", type=float, default=600.0)
    args = ap.parse_args()

    pkg: Path = args.fp32_package
    if not pkg.exists():
        ap.error(f"fp32 mlpackage missing: {pkg}")

    # Compile to .mlmodelc next to the .mlpackage so coreml-cli can profile it.
    mlmodelc = pkg.with_suffix(".mlmodelc")
    if not mlmodelc.exists():
        print(f"[bench] compiling {pkg.name} → {mlmodelc.name}", file=sys.stderr)
        out_dir = mlmodelc.parent
        proc = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(pkg), str(out_dir)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"coremlcompiler failed: {proc.stderr}")

    size_bytes = _du_bytes(pkg)

    # 1) Fallback / op breakdown.
    print("[bench] coreml-cli --fallback (one-time, slow on first compile)…", file=sys.stderr)
    fb = coreml_cli_fallback(mlmodelc, plan_timeout=args.plan_timeout)
    total_ops, ane_ops, breakdown = _aggregate_fallback(fb)
    ane_pct = (100.0 * ane_ops / total_ops) if total_ops else 0.0

    # 2) Warm latency. coreml-cli --units only takes one value, so we
    # call it twice — but the cold compile cache persists across calls.
    print("[bench] coreml-cli --units cpu_and_neural_engine …", file=sys.stderr)
    bench_ne = coreml_cli_bench(mlmodelc, "cpu_and_neural_engine",
                                n=3, plan_timeout=args.plan_timeout)
    print("[bench] coreml-cli --units cpu_only …", file=sys.stderr)
    bench_cpu = coreml_cli_bench(mlmodelc, "cpu_only",
                                 n=3, plan_timeout=args.plan_timeout)
    ms_ne = _median_predict_ms(bench_ne, "cpu_and_neural_engine")
    ms_cpu = _median_predict_ms(bench_cpu, "cpu_only")
    pct_ne = _summary_pct(bench_ne, "cpu_and_neural_engine")
    pct_cpu = _summary_pct(bench_cpu, "cpu_only")

    # 3) Parity (optional).
    parity = None
    if args.prod_mlmodelc is not None and args.prod_mlmodelc.exists():
        print(f"[bench] parity vs {args.prod_mlmodelc} …", file=sys.stderr)
        t0 = time.time()
        parity = parity_check(pkg, args.prod_mlmodelc, args.stage)
        print(f"[bench]   parity took {time.time() - t0:.1f}s", file=sys.stderr)

    summary = {
        "stage": args.stage,
        "fp32_package": str(pkg),
        "prod_mlmodelc": str(args.prod_mlmodelc) if args.prod_mlmodelc else None,
        "size_bytes": size_bytes,
        "size_human": _human_bytes(size_bytes),
        "total_ops": total_ops,
        "ane_ops": ane_ops,
        "ane_pct": round(ane_pct, 1),
        "device_breakdown": breakdown,
        "warm_predict_ms": {"cpu_and_neural_engine": ms_ne, "cpu_only": ms_cpu},
        "device_pct_at_cpu_and_neural_engine": pct_ne,
        "device_pct_at_cpu_only": pct_cpu,
        "parity_vs_prod": parity,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2))
        print(f"[bench] wrote {args.out_json}", file=sys.stderr)

    # Markdown row.
    cu = breakdown
    cu_s = f"ANE {cu['ane']} / GPU {cu['gpu']} / CPU {cu['cpu']}"
    if parity:
        cos_min = min(v["cosine"] for v in parity.values()
                      if isinstance(v, dict) and "cosine" in v)
        mxd_max = max(v["max_abs_delta"] for v in parity.values()
                      if isinstance(v, dict) and "max_abs_delta" in v)
        parity_s = f"cos {cos_min:.5f} / max|d| {mxd_max:.2e}"
    else:
        parity_s = "—"
    print(f"| {args.stage} | {pkg.name} | {summary['size_human']} | "
          f"{total_ops} | {ane_pct:.1f}% | {cu_s} | "
          f"{ms_ne:.2f} ms | {ms_cpu:.2f} ms | {parity_s} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
