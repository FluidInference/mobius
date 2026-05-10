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
| decoder_step | `decoder_step_fp32.mlpackage` | 395.9 MB | 771 | 0.0 % | ANE 0 / GPU 0 / CPU 771 | 90.39 ms | 87.59 ms | cos 0.99461 / max\|d\| 5.89e-01 |
| local_transformer | `local_transformer_fp32.mlpackage` | 33.6 MB | 423 | 0.0 % | ANE 0 / GPU 0 / CPU 423 | 1.45 ms | 1.74 ms | cos 0.93664 / max\|d\| 1.00e+03¹ |
| nanocodec_decoder_v3 | `nanocodec_decoder_v3.mlpackage` (production) | 121.0 MB | 1820 | 0.0 % | ANE 0 / GPU 0 / CPU 1820 | 379.56 ms | 315.56 ms | — (already production) |

¹ The cosine/max|d| numbers for `local_transformer` are misleading because
the model output is **`codes` (Int32, 8 codebook indices)** rather than a
continuous tensor. Two fp32-vs-fp16 paths can sample different discrete
indices given identical uniforms (drift from any internal fp16 round-off
shifts top-k boundaries); max|d|≈1000 just means some indices differ by
~1000 (a 2024-codebook). The bench harness measures these uniformly across
all stages — interpret with the dtype context in mind.

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

### decoder_step

- **0 % ANE residency.** All 771 ops on CPU at `.cpuAndNeuralEngine`.
  Same structural rejection as text_encoder / prefill — fp32 outputs at
  the I/O boundary force the entire 12-layer transformer body off ANE.
  Note the production fp16 `decoder_step.mlmodelc` lands 97.3 % ANE
  (per `PERF.md`); fp32 forfeits all of it.
- **Parity vs prod fp16.** Cosine 0.99461, max|d| 0.589 across the 38
  output tensors (logits + 24 KV-cache writes + 12 position scalars).
  Lower cosine than the encoder/prefill rows because the AR-step graph
  spans both the logits head (large dynamic range, ~[−40, +40]) and
  the KV writes (narrow). Driven by random KV-cache inputs at
  position=110 (mid-loop) per the seeded inputs feed.
- **Warm latency.** 90.39 ms at `.cpuAndNeuralEngine` vs 87.59 ms at
  `.cpuOnly` — both CPU. Production fp16 baseline: 15.7 ms / 97.3 %
  ANE → fp32 costs ~5.8× per step. Magpie's first chunk is 24 AR
  steps, so the per-utterance wall would scale by 24× the gap.

### local_transformer

- **0 % ANE residency** at fp32. Production fp16 LT lands **55.3 % ANE**
  per live `coreml-cli --fallback` (228 / 412 ops; the 73.9 % at
  `PERF.md:62/153` is stale — flagged in `OPTIONS.md`'s side-finding).
  At fp32 the planner forfeits all of it.
- **Parity vs prod fp16 — discrete-output caveat.** `codes` is `Int32 (8,)`,
  not a continuous tensor. Cosine and max|d| are reported by the bench
  harness uniformly, but they're not the right metric here: two paths
  with internal fp16 round-off drift can sample *different* top-k
  indices given identical uniforms, and the cosine on int32 codebook
  IDs is geometric, not perceptual. The 0.937 cosine / 1e3 max|d| just
  means a handful of codebooks chose different IDs. A real LT
  audibility test would compare downstream NanoCodec outputs.
- **Warm latency.** 1.45 ms at `.cpuAndNeuralEngine` vs 1.74 ms at
  `.cpuOnly` — both CPU. Tiny graph (33.6 MB / 423 ops); the absolute
  numbers are dwarfed by the surrounding decoder_step. Production fp16:
  ~1.6 ms at 55.3 % ANE.

### nanocodec_decoder_v3 (production fp32 — profile-only)

- **Already fp32 in production.** Per `PERF.md`, v3 is the **default**
  `MagpieNanocodecPrecision.fp32` build — full-fp32 weights are
  required for clean voiced speech (Phase F.2 audibility envelope; v2
  fp16 audibly noisy). This row is profile-only against the existing
  shipping artifact, not a new conversion.
- **0 % ANE residency.** Same fp32-output structural rejection as the
  upstream stages, plus the dilated-conv `W ≤ 16 384` ANE compiler
  ceiling on `T_out = 24 576` (Phase C+ subgraph probe). Even an fp16
  v3 wouldn't land on ANE under production input shape.
- **Warm latency.** 379.56 ms at `.cpuAndNeuralEngine` vs 315.56 ms at
  `.cpuOnly` — `.cpuAndNeuralEngine` is *slower* because the planner
  attempts ANE placement, fails, and falls back through CPU. This is
  exactly why `MagpieModelStore.swift` pins NanoCodec to `.cpuOnly`
  per `PERF.md:75-86`. **Never bench this stage at `.cpuAndNeuralEngine`
  in production.**
- **Parity skipped.** Candidate IS the production model; same-model
  comparison would be cosine 1.0 / max|d| 0.0 by construction.
