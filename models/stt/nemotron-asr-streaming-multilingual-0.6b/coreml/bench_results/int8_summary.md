# int8 Weight-Only Quantization — Trade-off Report

Encoder weight-only int8 quantization via
`coremltools.optimize.coreml.linear_quantize_weights` with default
`OpLinearQuantizerConfig(mode=linear_symmetric, dtype=int8, granularity=PER_CHANNEL, weight_threshold=2048)`.

Hardware: Apple M2, 16 GB, macOS 26.5.

## 1. Size

| Component | fp16 | int8 | Δ |
|---|---:|---:|---:|
| encoder.mlpackage | 1184.5 MB | 594.0 MB | **−50%** |
| Other components | unchanged (decoder/joint/preproc not quantized) | | |

## 2. Encoder Parity (vs fp32 PyTorch reference)

| Metric (encoded) | fp16 | int8 | Δ |
|---|---:|---:|---:|
| cosine (median) | 0.999992 | 0.996577 | −3.4e-3 |
| max\|Δ\| (median) | 5.28e-3 | 1.06e-1 | ~20× worse |
| relL2 (median) | — | 8.27e-2 | |
| Worst chunk cos | — | 0.9639 (ja_jp ch1) | |

Cache state (channel + time) stays tight (cos > 0.99 throughout); int8
drift does not catastrophically compound across the streaming window.

## 3. FLEURS WER/CER (n=50/lang smoke, forced mode)

| Lang | Metric | fp16 (n=100) | fp16 (full) | **int8 (n=50)** | Δ vs fp16 n=100 |
|---|---|---:|---:|---:|---:|
| en_us | WER | 11.18 | 12.09 | **10.41** | −0.77 |
| cmn_hans_cn | CER | 24.89 | 24.54 | **27.79** | **+2.90** |
| ja_jp | CER | 17.93 | 16.86 | **18.27** | +0.34 |
| es_419 | WER | 9.33 | 9.01 | **6.91** | −2.42 |
| fr_fr | WER | 16.62 | 15.18 | **15.28** | −1.34 |

Notes:
- 4/5 langs are within noise (n=50 sample variance dominates).
- **cmn_hans_cn shows a +2.90 CER regression (~12% relative)** — this is
  the only one that looks meaningful and matches the parity result
  (cmn cache_channel cos is lowest among the three sampled languages).
- Need full-FLEURS run on int8 to confirm the cmn regression isn't sample
  variance.

## 4. ANE Residency

Default scheduling (per `MLComputePlan`):

| Build | ANE ops | CPU ops | ANE % |
|---|---:|---:|---:|
| fp16 | 1635 / 1680 | 45 | **97.3%** |
| int8 | 1637 / 1976 | 339 | **82.8%** |

The 296 extra CPU ops are all `ios16.constexpr_affine_dequantize` (one
per quantized weight const). They are cheap (`est. CPU cost: ~3 ms`
aggregate) but the default ALL scheduler reads "many CPU ops" and routes
the whole encoder to GPU.

## 5. Latency (single chunk, after warmup)

| ComputeUnit | fp16 | int8 | Δ |
|---|---:|---:|---:|
| `.all` (default) | 41.77 ms (GPU) | **407.27 ms (GPU)** | **10× slower** ⚠️ |
| `.cpuAndGPU` | 45.24 ms | 259.31 ms | 5.7× slower |
| `.cpuOnly` | 134.19 ms | 42.60 ms | **3.1× faster** |
| `.cpuAndNeuralEngine` | 63.53 ms | **16.74 ms** | **3.8× faster** ✓ |

Cold compile: 12.3 s → **6.6 s** (47% faster).

### End-to-end RTFx (FLEURS benchmark, defaults)

The FLEURS harness uses `MLComputeUnits.ALL`, so it lands on GPU for both
builds:

| Build | RTFx (avg) |
|---|---:|
| fp16 | ~8.0× |
| int8 | ~2.2× (slower because default routes to GPU and int8 GPU path is awful) |

## 6. Recommendation

int8 is a **size + ANE-only** win, not a default win:

- **Pro**: 50% encoder size reduction, 3.8× faster inference on
  `.cpuAndNeuralEngine`, 47% faster cold compile.
- **Con**: Default `.all` scheduler picks the GPU path on int8 and is
  10× slower than fp16. Requires the Swift integration to **explicitly**
  set `MLComputeUnits.cpuAndNeuralEngine` for the encoder.
- **Con**: Measurable accuracy regression on Mandarin (+2.9 CER on
  smoke); needs full-FLEURS confirmation before shipping.

### Suggested next steps before shipping int8

1. Run full FLEURS (≥10 langs, all utts) on int8 to quantify real WER
   drift, especially cmn_hans_cn.
2. In FluidAudio Swift, pin encoder `MLModelConfiguration.computeUnits`
   to `.cpuAndNeuralEngine` for the int8 build (publish as a separate
   `build_int8` artifact so users can A/B).
3. Consider per-channel quantization with a higher
   `weight_threshold` or skipping the embedding/prompt MLP if the
   Mandarin regression localizes there.

## Artifacts

- `bench_results/parity_encoder_int8.{json,log}` — encoder parity
- `bench_results/fleurs_int8_smoke.{json,log}` — n=50/lang WER/CER
- `bench_results/encoder_int8_fallback.log` — ANE op assignment
- `bench_results/encoder_int8_bench.log` — per-compute-unit latency
- `quantize_int8.py` — quantization script
