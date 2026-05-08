"""Trial 10: decoder_upsample, FIXED shapes, fp32, ANE probe.

Why
---
Trial 8 (iteration_2/3) put `decoder_upsample` on `ct.ComputeUnit.ALL`
and saw warm latency oscillate 322-759 ms — strong signal CoreML's ANE
planner was *trying* to place the HiFi-GAN ConvTranspose1d ups stack
but bailing intermittently. The mlpackage's time axes are RangeDim
(`T_FRAME2` 2..4096, `HAR_LEN` 600..1228800), and ANE generally rejects
dynamic shapes for ConvTranspose stacks.

This trial collapses both time axes to the trace-default fixed shapes
(T_FRAME=147 → x_pre [1,512,294], har_source [1,1,88200]) and
re-converts at fp32. fp32-first: only one variable changes vs the
known-good iteration_2 baseline (precision stays fp32, shape becomes
fixed). If ANE accepts the graph and warm latency is competitive with
or beats the current 304 ms CPU number, Trial 10 promotes to a fixed
single-bucket placement; the next step is EnumeratedShapes with N
fixed buckets to recover token-length flexibility.

Output
------
* `coreml/packages/decoder_upsample_trial10_fp32_fixed.mlpackage`

Bench
-----
Loads the saved package three times under CPU_ONLY, CPU_AND_NE, and
ALL; runs warmup + 8 timed predicts on the same fixed input; reports
load + min/med/avg/max ms and parity vs the eager wrapper output.

Run
---
    cd models/tts/styletts2
    uv run python coreml/trial10_decoder_upsample_fixed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Importing convert installs `_patch_coreml_int_cast` and
# `_register_aten_aliases` — both are required for the decoder graph to
# convert cleanly (HiFi-GAN uses `torch.multiply`; the LSTM weight-norm
# strip emits `aten::Int` chains).
from coreml import convert as _convert  # noqa: F401  (side-effect import)
from coreml._runtime import HERE, build_runtime, stage_example_inputs
from coreml.wrappers import build_wrapper

PACKAGES_DIR = HERE / "coreml" / "packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = PACKAGES_DIR / "decoder_upsample_trial10_fp32_fixed.mlpackage"


def _metric(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    return {
        "shape": tuple(a.shape),
        "mse": float(np.mean(diff * diff)),
        "max_abs_delta": float(np.max(np.abs(diff))),
        "rms_a": float(np.sqrt(np.mean(a * a))),
        "rms_b": float(np.sqrt(np.mean(b * b))),
        "cos": cos,
    }


def convert_fixed():
    import coremltools as ct

    print("=== Trial 10 convert: decoder_upsample fp32 fixed-shape ===")
    rt = build_runtime()
    wrapper = build_wrapper("decoder_upsample", rt.model)
    example_inputs = stage_example_inputs("decoder_upsample", rt)

    x_pre, ref, har = example_inputs
    print(f"  x_pre      = {tuple(x_pre.shape)} {x_pre.dtype}")
    print(f"  ref        = {tuple(ref.shape)} {ref.dtype}")
    print(f"  har_source = {tuple(har.shape)} {har.dtype}")

    with torch.no_grad():
        eager_out = wrapper(*example_inputs)
    print(f"  eager wav  = {tuple(eager_out.shape)} {eager_out.dtype}")

    print("  tracing ...")
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, example_inputs, check_trace=False, strict=False
        )

    print("  ct.convert (fp32, fixed shapes, units=ALL at convert time) ...")
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
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  ct.convert: {time.time() - t0:.1f}s")

    if OUT_PATH.exists():
        import shutil

        shutil.rmtree(OUT_PATH)
    mlmodel.save(str(OUT_PATH))
    print(f"  saved {OUT_PATH.relative_to(HERE)}")
    return example_inputs, eager_out


def bench(example_inputs, eager_out):
    import coremltools as ct

    feed = {
        "x_pre": example_inputs[0].detach().numpy().astype(np.float32),
        "ref": example_inputs[1].detach().numpy().astype(np.float32),
        "har_source": example_inputs[2].detach().numpy().astype(np.float32),
    }
    eager_np = eager_out.detach().numpy().astype(np.float32)

    placements = [
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("ALL", ct.ComputeUnit.ALL),
    ]

    print("\n=== Trial 10 bench (fp32 fixed-shape decoder_upsample) ===")
    for name, units in placements:
        print(f"\n  --- {name} ---")
        t0 = time.time()
        m = ct.models.MLModel(str(OUT_PATH), compute_units=units)
        load_ms = (time.time() - t0) * 1000.0

        for _ in range(3):
            m.predict(feed)

        timings = []
        for _ in range(8):
            t1 = time.time()
            out = m.predict(feed)
            timings.append((time.time() - t1) * 1000.0)
        timings.sort()
        out_arr = np.asarray(list(out.values())[0])
        met = _metric(eager_np, out_arr)
        med = timings[len(timings) // 2]
        avg = sum(timings) / len(timings)
        bimodal_gap = timings[-1] - timings[0]
        print(
            f"  load={load_ms:6.0f}ms  warm: min={timings[0]:6.1f} med={med:6.1f} "
            f"avg={avg:6.1f} max={timings[-1]:6.1f}  spread={bimodal_gap:5.1f} ms"
        )
        print(
            f"  parity vs eager: cos={met['cos']:.6f}  max|d|={met['max_abs_delta']:.3e}"
        )


def main():
    example_inputs, eager_out = convert_fixed()
    bench(example_inputs, eager_out)


if __name__ == "__main__":
    main()
