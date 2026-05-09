"""Trial 10e Option 1 — T_mel cap (find empirical ANE-accepting bucket).

Step 1c proved the ANE blocker is a 16,384-element width-axis limit on
intermediate tensors after `generator.ups[1]`, not an op-type
rejection. Conv2d rewrite is orthogonal (Trial 10c at fp16 + Conv2d
hits the same widths 16,414 / 16,390 at T_mel = 294).

This script finds the largest T_mel that ANE will accept and confirms
parity vs the PyTorch wrapper at that cap. Mirrors `iteration_3`'s
production fp16 1D conversion shape exactly except for fixed T_mel.

Per-stage shape math:
    T_mel input  → x_pre = (1, 512, T_mel)
    ups[0]: stride 10, kernel 20  →  T ≈ T_mel × 10
    ups[1]: stride 5,  kernel 10  →  T ≈ T_mel × 50  ← ANE width check fires here
    ups[2]: stride 3,  kernel 6   →  T ≈ T_mel × 150
    ups[3]: stride 2,  kernel 4   →  T ≈ T_mel × 300

Empirical ANE width budget on this M2 / macOS 26.5 (from Step 1):
    T_mel = 294 → ANE pads/tiles post-ups[1] tensor to 16,414 / 16,390 → REJECT
    T_mel = ?   → must produce post-ups[1] internal layout < 16,384

Crops the captured T_mel = 294 fixture to the candidate T_mel, traces +
converts the standard `decoder_upsample` wrapper (NOT the Conv2d-
rewritten variant — Trial 10b/10c showed Conv2d is orthogonal to the
width budget) at fp16 fixed shapes, saves to coreml/packages/.

For each candidate this script:
    1. converts at fixed T_mel
    2. probes with .cpuAndNeuralEngine to capture any
       'Tensor width' / 'ANECCompile() FAILED' messages
    3. runs `coreml-cli --fallback --json` to capture ANE residency %
    4. runs the wrapper in PyTorch on the same cropped inputs to get
       gold reference, computes cosine + max|Δ| vs CoreML output
    5. benches warm-avg on CPU_ONLY and CPU_AND_NE

Run:
    cd models/tts/styletts2
    uv run python coreml/experiments/trial10e1_t_mel_cap.py \\
        --candidates 280,290,292,293,294
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coreml.exporters import convert as _convert  # noqa: F401  (installs MIL patches)
from coreml._runtime import HERE, build_runtime, stage_example_inputs
from coreml.wrappers import build_wrapper

import coremltools as ct

PACKAGES_DIR = HERE / "coreml" / "packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
COREML_CLI = str(_HERE.parent.parent.parent / "tools" / "coreml-cli")
HOP = 300  # decoder hop_factor
WIDTH_RE = re.compile(r"Tensor width goes beyond limit supported \((\d+) > (\d+)")
ANE_FAIL_RE = re.compile(r"ANECCompile\(\)? FAILED")


@dataclass
class CapResult:
    t_mel: int
    pkg_path: Path
    ane_widths: list[int]            # widths that exceeded 16,384 (empty = ANE accepted)
    ane_compile_failed: bool
    ane_pct: float                    # from coreml-cli --fallback --json
    parity_cos: float                 # vs PyTorch wrapper at same cropped inputs
    parity_max_abs: float
    bench_cpu_only_ms: float
    bench_cpu_and_ne_ms: float


def _crop_inputs(example_inputs: tuple, t_mel: int) -> tuple:
    """Crop the captured T_mel = 294 fixture to a smaller T_mel along the
    time axis. ref is unchanged; x_pre and har_source are sliced."""
    x_pre, ref, har = example_inputs
    return (
        x_pre[:, :, :t_mel].contiguous(),
        ref.contiguous(),
        har[:, :, :t_mel * HOP].contiguous(),
    )


def _convert_at_t_mel(
    wrapper: torch.nn.Module,
    cropped_inputs: tuple,
    t_mel: int,
) -> Path:
    """Trace + convert the wrapper at fixed T_mel. fp16, ALL units (so the
    converter doesn't reject upfront), 1D Conv (matches production iteration_3
    shape — the smallest-change-from-prod variant)."""
    x_pre, ref, har = cropped_inputs
    out_path = PACKAGES_DIR / f"decoder_upsample_trial10e1_fp16_tmel{t_mel}.mlpackage"

    print(f"[trial10e1] T_mel={t_mel}: tracing wrapper at fixed shape ...", flush=True)
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, cropped_inputs, check_trace=False, strict=False
        )

    print(
        f"[trial10e1] T_mel={t_mel}: ct.convert (fp16, fixed, 1D) ...",
        flush=True,
    )
    inputs = [
        ct.TensorType(name="x_pre", shape=tuple(x_pre.shape), dtype=np.float32),
        ct.TensorType(name="ref", shape=tuple(ref.shape), dtype=np.float32),
        ct.TensorType(name="har_source", shape=tuple(har.shape), dtype=np.float32),
    ]
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"[trial10e1] T_mel={t_mel}:   ct.convert took {time.time() - t0:.1f}s")

    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    return out_path


def _probe_ane(pkg_path: Path, cropped_inputs: tuple) -> tuple[list[int], bool, float]:
    """Load mlpackage with .cpuAndNeuralEngine, run one predict, capture
    stderr. Return (widths_exceeded, ane_compile_failed, predict_ms)."""
    # Dereference any HF-style symlinks so the compiler can copy.
    deref = Path(tempfile.gettempdir()) / pkg_path.name
    if deref.exists():
        shutil.rmtree(deref)
    shutil.copytree(pkg_path, deref, symlinks=False)

    probe = (
        "import sys, numpy as np, time\n"
        "import coremltools as ct\n"
        f"m = ct.models.MLModel({str(deref)!r}, compute_units=ct.ComputeUnit.CPU_AND_NE)\n"
        "x_pre = np.load(sys.argv[1])\n"
        "ref = np.load(sys.argv[2])\n"
        "har = np.load(sys.argv[3])\n"
        "feed = {'x_pre': x_pre, 'ref': ref, 'har_source': har}\n"
        "t0 = time.time()\n"
        "out = m.predict(feed)\n"
        "ms = (time.time() - t0) * 1000\n"
        "print(f'PREDICT_MS={ms:.1f}', file=sys.stderr)\n"
    )

    # Stash inputs to npy for the subprocess.
    x_pre, ref, har = cropped_inputs
    tmpdir = Path(tempfile.mkdtemp(prefix="trial10e1_"))
    xp = tmpdir / "x_pre.npy"; np.save(xp, x_pre.detach().numpy().astype(np.float32))
    rp = tmpdir / "ref.npy"; np.save(rp, ref.detach().numpy().astype(np.float32))
    hp = tmpdir / "har.npy"; np.save(hp, har.detach().numpy().astype(np.float32))

    proc = subprocess.run(
        ["uv", "run", "python", "-c", probe, str(xp), str(rp), str(hp)],
        capture_output=True, text=True, timeout=600,
        env={**__import__("os").environ, "MLLOG": "1", "OS_ACTIVITY_MODE": "info"},
        cwd=str(_HERE),
    )
    stderr = proc.stderr
    widths = sorted({int(m.group(1)) for m in WIDTH_RE.finditer(stderr)})
    failed = bool(ANE_FAIL_RE.search(stderr))
    pred_ms = 0.0
    for line in stderr.splitlines():
        if line.startswith("PREDICT_MS="):
            pred_ms = float(line.split("=", 1)[1])
    shutil.rmtree(tmpdir, ignore_errors=True)
    return widths, failed, pred_ms


def _coreml_cli_residency(pkg_path: Path) -> float:
    """Compile mlpackage to mlmodelc, run coreml-cli --fallback --json,
    return ANE residency %. Returns NaN if compute_plan timed out."""
    out = Path(ct.utils.compile_model(str(pkg_path.resolve())))
    cli_input = Path(tempfile.mkdtemp(prefix="trial10e1_cli_")) / out.name
    shutil.copytree(out, cli_input)

    proc = subprocess.run(
        ["uv", "run", "coreml-cli", str(cli_input),
         "--fallback", "--json", "-u", "cpu_and_neural_engine",
         "--plan-timeout", "300"],
        capture_output=True, text=True, timeout=900,
        cwd=COREML_CLI,
    )
    text = proc.stdout
    last_brace = text.rfind("}")
    if last_brace < 0:
        return float("nan")
    try:
        d = json.loads(text[:last_brace + 1])
    except json.JSONDecodeError:
        return float("nan")
    fb = d["models"][0]["fallback"]
    return float(fb["ane_percent"])


def _bench_warm(pkg_path: Path, cropped_inputs: tuple, units: ct.ComputeUnit, n: int = 8) -> float:
    """Load with explicit units, warm 3, time n predicts, return median ms."""
    deref = Path(tempfile.gettempdir()) / f"bench_{pkg_path.name}"
    if deref.exists():
        shutil.rmtree(deref)
    shutil.copytree(pkg_path, deref, symlinks=False)
    m = ct.models.MLModel(str(deref), compute_units=units)
    x_pre, ref, har = cropped_inputs
    feed = {
        "x_pre": x_pre.detach().numpy().astype(np.float32),
        "ref": ref.detach().numpy().astype(np.float32),
        "har_source": har.detach().numpy().astype(np.float32),
    }
    for _ in range(3):
        m.predict(feed)
    timings = []
    for _ in range(n):
        t0 = time.time()
        m.predict(feed)
        timings.append((time.time() - t0) * 1000.0)
    timings.sort()
    return timings[len(timings) // 2]


def _eager_reference(wrapper: torch.nn.Module, cropped_inputs: tuple) -> np.ndarray:
    with torch.no_grad():
        out = wrapper(*cropped_inputs)
    return out.detach().numpy()


def _cos_max(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    md = float(np.max(np.abs(a - b)))
    return cos, md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=str, default="280,292,293,294",
        help="Comma-separated T_mel values to probe. Default brackets the "
        "expected boundary.",
    )
    parser.add_argument(
        "--skip-bench", action="store_true",
        help="Skip warm-avg bench (probe + parity only).",
    )
    args = parser.parse_args()

    candidates = [int(x) for x in args.candidates.split(",")]
    print(f"[trial10e1] candidates: {candidates}")

    rt = build_runtime()
    wrapper = build_wrapper("decoder_upsample", rt.model)
    full_inputs = stage_example_inputs("decoder_upsample", rt)
    print(
        f"[trial10e1] base example_inputs: x_pre={tuple(full_inputs[0].shape)} "
        f"ref={tuple(full_inputs[1].shape)} har={tuple(full_inputs[2].shape)}"
    )

    results: list[CapResult] = []

    for t_mel in candidates:
        cropped = _crop_inputs(full_inputs, t_mel)
        pkg = _convert_at_t_mel(wrapper, cropped, t_mel)

        # PyTorch reference at this T_mel
        ref_out = _eager_reference(wrapper, cropped)

        widths, failed, pred_ms = _probe_ane(pkg, cropped)
        ane_pct = _coreml_cli_residency(pkg)

        # Parity vs PyTorch
        deref = Path(tempfile.gettempdir()) / f"parity_{pkg.name}"
        if deref.exists():
            shutil.rmtree(deref)
        shutil.copytree(pkg, deref, symlinks=False)
        m = ct.models.MLModel(str(deref), compute_units=ct.ComputeUnit.CPU_ONLY)
        x_pre, ref_t, har = cropped
        cm_out = m.predict({
            "x_pre": x_pre.detach().numpy().astype(np.float32),
            "ref": ref_t.detach().numpy().astype(np.float32),
            "har_source": har.detach().numpy().astype(np.float32),
        })
        cm_arr = np.asarray(list(cm_out.values())[0])
        cos, md = _cos_max(ref_out, cm_arr)

        cpu_ms = float("nan")
        ne_ms = float("nan")
        if not args.skip_bench:
            cpu_ms = _bench_warm(pkg, cropped, ct.ComputeUnit.CPU_ONLY)
            ne_ms = _bench_warm(pkg, cropped, ct.ComputeUnit.CPU_AND_NE)

        results.append(CapResult(
            t_mel=t_mel,
            pkg_path=pkg,
            ane_widths=widths,
            ane_compile_failed=failed,
            ane_pct=ane_pct,
            parity_cos=cos,
            parity_max_abs=md,
            bench_cpu_only_ms=cpu_ms,
            bench_cpu_and_ne_ms=ne_ms,
        ))

        print()
        print(f"[trial10e1] === T_mel = {t_mel} ===")
        print(f"  widths exceeded:     {widths!r}")
        print(f"  ANECCompile failed:  {failed}")
        print(f"  ANE residency:       {ane_pct:.1f}%")
        print(f"  parity vs PyTorch:   cos={cos:.6f}  max|d|={md:.3e}")
        print(f"  bench CPU_ONLY:      {cpu_ms:.1f} ms (median, warm)")
        print(f"  bench CPU_AND_NE:    {ne_ms:.1f} ms (median, warm)")
        print(f"  pkg:                 {pkg.relative_to(_HERE)}")

    print()
    print("[trial10e1] === summary ===")
    print(
        f"  {'T_mel':>5} | {'ANE accept':>10} | {'ANE %':>6} | {'cos':>10} | "
        f"{'CPU only':>9} | {'CPU+NE':>9} |"
    )
    for r in results:
        accept = "yes" if not r.ane_widths and not r.ane_compile_failed else "no"
        print(
            f"  {r.t_mel:>5} | {accept:>10} | {r.ane_pct:>5.1f}% | "
            f"{r.parity_cos:>10.6f} | {r.bench_cpu_only_ms:>7.1f}ms | "
            f"{r.bench_cpu_and_ne_ms:>7.1f}ms |"
        )


if __name__ == "__main__":
    main()
