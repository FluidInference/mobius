"""fp32 baseline + tail-fp16 probe harness — diagnostic-only.

Two related but distinct diagnostic tracks live here:

1. **fp32 stage-by-stage baseline** (`bench_one.py`, `inputs.py`).
   Captures per-stage parity, ANE residency, model size, and warm
   latency for every Magpie pipeline model re-converted at
   `compute_precision=ct.precision.FLOAT32`. Output: rows in
   `coreml/BASELINE_FP32.md` + JSON captures in
   `coreml/build/fp32/<stage>.bench.json`.

2. **Option 1 tail-fp16 probe** (`tail_fp16_probe.py`,
   `tail_fp16_audibility.py`). Custom MIL `AbstractGraphPass` that
   casts only late-stage HiFi-GAN ops to fp16 while keeping the early
   stages fp32 — the one configuration Phase F.2's op-class sweep
   didn't cover. Trial 11 in `PERF.md` recorded the v1 (smallest
   region) result as a definitive negative; v2/v3 skipped per the
   stopping rule.

Outputs land in `coreml/build/fp32/` and are not uploaded to HuggingFace
or used in production. The fp32 mlpackages here have no `MagpieModelStore`
loader path; they exist only for the diagnostic comparison.
"""
