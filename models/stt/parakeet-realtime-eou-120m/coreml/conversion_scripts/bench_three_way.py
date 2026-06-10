#!/usr/bin/env python3
"""Per-RNNT-step interleaved benchmark: ref pair vs traced fused vs MIL-lean fused.

Single process, single harness — resolves the cross-harness speedup
discrepancy (traced 1.23x measured in its own Python harness on M4 Pro vs
lean 1.71x/step measured in the Swift harness on M5) on equal footing.

Interleaved A/B/C (ref, traced, lean per iteration), 10 warmup + 200 timed,
median/p95, CPU_ONLY and CPU_AND_NE. Inputs: zero LSTM state, blank last
token, one REAL encoder frame (from the wer_three_way encoder cache) so the
argmax/softmax paths see production-like values.

Usage:
    python bench_three_way.py \
        --model-dir "$HOME/Library/Application Support/FluidAudio/Models/parakeet-eou-streaming/160ms" \
        --traced .../decoder_joint_decision_fp16.mlpackage \
        --lean /tmp/eou_fused/decoder_joint_decision_fused.mlpackage \
        --enc-frame /tmp/eou_enc_cache/<file>.npy
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import coremltools as ct
import numpy as np

BLANK_ID = 1026

CU = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--traced", type=Path, required=True)
    ap.add_argument("--lean", type=Path, required=True)
    ap.add_argument("--enc-frame", type=Path, default=None, help=".npy of encoder frames; first frame used")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()

    if args.enc_frame is not None:
        enc = np.load(args.enc_frame)[0][None, :, None].astype(np.float32)  # [1, 512, 1]
    else:
        enc = (np.random.default_rng(0).standard_normal((1, 512, 1)) * 2.0).astype(np.float32)

    h = np.zeros((1, 1, 640), dtype=np.float32)
    c = np.zeros((1, 1, 640), dtype=np.float32)
    tok = np.array([[BLANK_ID]], dtype=np.int32)
    tlen = np.array([1], dtype=np.int32)

    for cu_name, cu in CU.items():
        dec = ct.models.CompiledMLModel(str(args.model_dir / "decoder.mlmodelc"), compute_units=cu)
        joint = ct.models.CompiledMLModel(str(args.model_dir / "joint_decision.mlmodelc"), compute_units=cu)
        traced = ct.models.MLModel(str(args.traced), compute_units=cu)
        lean = ct.models.MLModel(str(args.lean), compute_units=cu)

        def step_ref():
            d = dec.predict({"targets": tok, "target_length": tlen, "h_in": h, "c_in": c})
            joint.predict({"encoder_step": enc, "decoder_step": d["decoder"].astype(np.float32)})

        def step_traced():
            traced.predict({"targets": tok, "target_length": tlen, "h_in": h, "c_in": c, "encoder_step": enc})

        def step_lean():
            lean.predict({"targets": tok, "h_in": h, "c_in": c, "encoder_step": enc})

        variants = [("ref", step_ref), ("traced", step_traced), ("lean", step_lean)]

        for _ in range(args.warmup):
            for _, fn in variants:
                fn()

        times: dict[str, list[float]] = {name: [] for name, _ in variants}
        for _ in range(args.runs):
            for name, fn in variants:
                t0 = time.perf_counter()
                fn()
                times[name].append((time.perf_counter() - t0) * 1e3)

        ref_med = statistics.median(times["ref"])
        for name, _ in variants:
            xs = sorted(times[name])
            med = statistics.median(xs)
            p95 = xs[int(len(xs) * 0.95) - 1]
            print(
                f"{cu_name:<11} {name:<7} median {med:6.4f} ms  p95 {p95:6.4f} ms  "
                f"speedup vs ref {ref_med / med:.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
