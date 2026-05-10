# Magpie TTS — fp32 baseline (diagnostic)

Per-stage profile of every Magpie pipeline model re-converted at
`compute_precision=ct.precision.FLOAT32`, captured to give us a parity
reference before any further optimization. **Diagnostic only** — these
mlpackages are not uploaded to HuggingFace and do not change the
production `MagpieModelStore` selection (which keeps fp16 + the v3
fp32 nanocodec exactly as-is).

Hardware: **Apple M2, 16 GB, macOS 26.5, coremltools 9.0**. Each row
captured by `experiments/baseline_fp32/bench_one.py` against the
compiled `.mlmodelc` in `build/fp32/`.

Conventions:

- **size** — disk size of the fp32 `.mlpackage`.
- **ops** — `total_ops` from `coreml-cli --fallback`.
- **ANE %** — `ane_ops / total_ops` at `cpu_and_neural_engine` placement.
- **CU breakdown** — op count assigned to ANE / GPU / CPU at the same policy.
- **warm @ cpu+ne / cpu** — median of 3 timed iterations after warmup, `cpu_and_neural_engine` and `cpu_only`.
- **parity vs prod** — `(min cosine, max max|delta|)` across all output tensors comparing the fp32 mlpackage (CPU) vs the production fp16 `.mlmodelc` (CPU) on a deterministic seeded input.

Per-stage JSON: `build/fp32/<stage>.bench.json`.

## Results

| stage | mlpackage | size | ops | ANE % | CU breakdown | warm @ cpu+ne | warm @ cpu | parity vs prod |
|---|---|---|---|---|---|---|---|---|
| text_encoder | `text_encoder_fp32.mlpackage` | 386.1 MB | 225 | 0.0 % | ANE 0 / GPU 0 / CPU 225 | 78.38 ms | 78.88 ms | cos 0.99852 / max\|d\| 4.43e-01 |
| decoder_prefill | `decoder_prefill_fp32.mlpackage` | 321.0 MB | 541 | 0.0 % | ANE 0 / GPU 0 / CPU 541 | 112.76 ms | 121.27 ms | cos 0.99997 / max\|d\| 7.58e-02 |

## Per-stage notes

### text_encoder

- **0 % ANE residency.** All 225 ops fall back to CPU. `coreml-cli --fallback`
  reports the rejection reason as **"Invalid output tensor format: fp32"** for
  209 of 225 ops. The fp32 graph keeps the encoder output in `Float32` end-to-end,
  but the M2 ANE only accepts fp16 at the I/O boundary, so even kernels that
  *could* run on ANE are dropped by the planner.
- **Parity vs prod fp16.** Cosine 0.998, max|d| 0.44 on a 1×256×768 hidden state
  with empirical range ~[-13, +14]. The 0.44 max|d| is roughly 1.5 % of the
  output's dynamic range — within the expected envelope for fp16 vs fp32 on a
  12-layer transformer. This is the target the production fp16 model is
  paying for in numerics, not a regression.
- **Warm latency.** 78 ms at both `.cpuAndNeuralEngine` and `.cpuOnly` (identical
  because everything runs on CPU regardless). Production fp16 baseline (per
  `PERF.md`) is 12.4 ms at `.cpuAndNeuralEngine`, 98 % ANE → fp32 costs ~6.3×.

### decoder_prefill

- **0 % ANE residency.** Same structural rejection as text_encoder — every one of
  the 541 ops marked "Invalid output tensor format: fp32". The 24 KV-cache
  outputs all leave the graph as `MultiArray (Float32 …)` so the planner can't
  hand any of the deeper graph to ANE.
- **Parity vs prod fp16.** Cosine 0.99997, max|d| 0.076 across all 24 KV-cache
  outputs (each shaped 2 × 1 × 512 × 12 × 64). Tighter than text_encoder because
  the prefill outputs span a narrower dynamic range (KV cache scaling pre-softmax).
- **Warm latency.** 112.8 ms at `.cpuAndNeuralEngine` vs 121.3 ms at `.cpuOnly`.
  The 8 ms gap on a CPU-only graph likely reflects different threading defaults
  in the coreml-cli loader's two policies, not actual ANE work — both rows show
  100 % CPU placement in `summary`. Production fp16 baseline: 17.1 ms /
  93.9 % ANE → fp32 costs ~6.6× wall-clock.
