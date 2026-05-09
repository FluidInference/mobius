"""Trial 6 — convert the fused f0n_predictor + har_source stage.

Builds `coreml/packages/fused_f0n_har_source{,_fp16}.mlpackage`.

Why
---
`f0n_predictor` produces `f0`, which `har_source` consumes immediately.
Standalone, that intermediate `f0 [1, F0_LEN]` array is marshalled out
of one mlmodel and into the next on every utterance, plus a second
`predict()` round-trip is paid. Fuse them into one call. Three outputs
are exposed because both `f0` and `n` are still consumed by
`decoder_pre` downstream.

Pseudocode (matches `coreml/fusions.md` Trial 6):

    class FusedF0NHarSource(nn.Module):
        def __init__(self, predictor, decoder):
            self.f0n_wrap = F0NPredictorWrapper(predictor)
            self.har_wrap = HarSourceWrapper(decoder)

        def forward(self, en, s):
            f0, n = self.f0n_wrap(en, s)        # [1, F0_LEN], [1, F0_LEN]
            har   = self.har_wrap(f0)           # [1, 1, F0_LEN * 300]
            return f0, n, har

CoreML inputs / outputs
-----------------------
| name | shape                | dtype | meaning |
|------|----------------------|-------|---------|
| en   | `[1, 640, T_FRAME]`  | f32   | aligned predictor input (asr-shifted, hidden=640) |
| s    | `[1, 128]`           | f32   | predictor encoder half of `ref_s` |

| output  | shape              | role                       |
|---------|--------------------|----------------------------|
| `f0`    | `[1, F0_LEN]`      | f0 contour (-> decoder_pre, internal har) |
| `n`     | `[1, F0_LEN]`      | n contour (-> decoder_pre)  |
| `har`   | `[1, 1, HAR_LEN]`  | har source (-> decoder_upsample) |

`F0_LEN = 2 * T_FRAME`, `HAR_LEN = 300 * F0_LEN = 600 * T_FRAME`.

T_FRAME and its derived axes are exposed as `ct.RangeDim` so the
package accepts any sentence length.

Run
---
    cd models/tts/styletts2
    uv run python coreml/exporters/fuse_f0n_har_source.py
    uv run python coreml/exporters/fuse_f0n_har_source.py --precision fp16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coreml.exporters import convert as _convert  # noqa: F401  (installs MIL patches)
from coreml._runtime import HERE, build_runtime, stage_example_inputs
from coreml.wrappers import F0NPredictorWrapper, HarSourceWrapper

PACKAGES_DIR = HERE / "coreml" / "packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fused module
# ---------------------------------------------------------------------------


class FusedF0NHarSource(nn.Module):
    """One graph for f0n_predictor -> har_source.

    Reuses the existing per-stage wrappers unchanged; the only new
    surface is the three-output forward.
    """

    def __init__(self, predictor: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.f0n_wrap = F0NPredictorWrapper(predictor)
        self.har_wrap = HarSourceWrapper(decoder)
        self.eval()

    def forward(
        self, en: torch.Tensor, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0, n = self.f0n_wrap(en, s)
        har = self.har_wrap(f0)
        return f0, n, har


# ---------------------------------------------------------------------------
# Convert + bench
# ---------------------------------------------------------------------------


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
        "cos": cos,
    }


def convert_fused(*, precision: str) -> tuple[Path, tuple, tuple]:
    import coremltools as ct

    print(f"=== Trial 6 convert: fused_f0n_har_source ({precision}) ===")
    rt = build_runtime()

    # Use the same captured (en, s) inputs that f0n_predictor stage uses.
    en, s = stage_example_inputs("f0n_predictor", rt)
    print(f"  en = {tuple(en.shape)} {en.dtype}")
    print(f"  s  = {tuple(s.shape)} {s.dtype}")

    fused = FusedF0NHarSource(rt.model.predictor, rt.model.decoder)

    # Reference: standalone f0n -> har eager chain. Must match fused
    # eager bit-equivalently because the wrappers are reused unchanged.
    ref_f0n = F0NPredictorWrapper(rt.model.predictor)
    ref_har = HarSourceWrapper(rt.model.decoder)
    with torch.no_grad():
        ref_f0, ref_n = ref_f0n(en, s)
        ref_h = ref_har(ref_f0)
        f0_e, n_e, h_e = fused(en, s)

    for name, a, b in [("f0", ref_f0, f0_e), ("n", ref_n, n_e), ("har", ref_h, h_e)]:
        m = _metric(a.numpy(), b.numpy())
        print(
            f"  eager parity {name:>3}: cos={m['cos']:.6f} "
            f"max|d|={m['max_abs_delta']:.3e}"
        )
        if m["max_abs_delta"] > 1e-4:
            raise SystemExit(
                f"ABORT: fused {name} does not match standalone reference."
            )

    print("  tracing ...")
    fused.eval()
    with torch.no_grad():
        traced = torch.jit.trace(fused, (en, s), check_trace=False, strict=False)

    if precision not in ("fp16", "fp32"):
        raise ValueError(f"precision must be fp16 or fp32, got {precision!r}")
    ct_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )

    # T_FRAME range matches convert.py's f0n_predictor + har_source.
    T_FRAME = ct.RangeDim(lower_bound=1, upper_bound=2048, default=int(en.shape[2]))
    descs = [
        ct.TensorType(
            name="en",
            shape=ct.Shape(shape=(1, en.shape[1], T_FRAME)),
            dtype=np.float32,
        ),
        ct.TensorType(name="s", shape=tuple(s.shape), dtype=np.float32),
    ]

    print(f"  ct.convert ({precision}) ...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=descs,
        convert_to="mlprogram",
        compute_precision=ct_precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  ct.convert: {time.time() - t0:.1f}s")

    suffix = "_fp16" if precision == "fp16" else ""
    out_path = PACKAGES_DIR / f"fused_f0n_har_source{suffix}.mlpackage"
    if out_path.exists():
        import shutil

        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    print(f"  saved {out_path.relative_to(HERE)}")
    return out_path, (en, s), (ref_f0, ref_n, ref_h)


def bench(out_path: Path, inputs: tuple, refs: tuple) -> None:
    import coremltools as ct

    en, s = inputs
    ref_f0, ref_n, ref_h = refs
    feed = {
        "en": en.detach().numpy().astype(np.float32),
        "s": s.detach().numpy().astype(np.float32),
    }
    placements = [
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("ALL", ct.ComputeUnit.ALL),
    ]

    print("\n=== Trial 6 bench (fused_f0n_har_source) ===")
    for name, units in placements:
        print(f"\n  --- {name} ---")
        t0 = time.time()
        m = ct.models.MLModel(str(out_path), compute_units=units)
        load_ms = (time.time() - t0) * 1000.0

        for _ in range(3):
            m.predict(feed)

        timings = []
        for _ in range(8):
            t1 = time.time()
            out = m.predict(feed)
            timings.append((time.time() - t1) * 1000.0)
        timings.sort()
        # Spec output order: f0, n, har (same as eager fused.forward).
        out_vals = list(out.values())
        med = timings[len(timings) // 2]
        avg = sum(timings) / len(timings)
        spread = timings[-1] - timings[0]
        print(
            f"  load={load_ms:6.0f}ms  warm: min={timings[0]:6.1f} med={med:6.1f} "
            f"avg={avg:6.1f} max={timings[-1]:6.1f}  spread={spread:5.1f} ms"
        )
        for tag, ref, ml in zip(("f0", "n", "har"), (ref_f0, ref_n, ref_h), out_vals):
            met = _metric(ref.detach().numpy(), np.asarray(ml))
            print(
                f"  parity {tag:>3}: cos={met['cos']:.6f}  max|d|={met['max_abs_delta']:.3e}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16"],
        help="output mlpackage precision (default fp32)",
    )
    parser.add_argument("--no-bench", action="store_true", help="skip the bench loop")
    args = parser.parse_args()

    out_path, inputs, refs = convert_fused(precision=args.precision)
    if not args.no_bench:
        bench(out_path, inputs, refs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
