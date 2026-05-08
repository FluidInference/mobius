# StyleTTS2 → CoreML iteration_3

Mixed-precision build on top of iteration_2: 7 stages flipped to fp16
weight precision, 1 stage kept at fp32 to avoid an audible-quality
regression. Disk halved, pipeline-stage sum cut 24–41 % cool.

## Pipeline (8 stages, 8 dispatches)

```
text_encoder           → CPU_ONLY      fp16   11 MB
bert                   → ALL           fp16   12 MB
ref_encoder            → CPU_AND_GPU   fp16   53 MB
fused_diffusion_sampler → ALL          fp16   47 MB   ← Trial 4
duration_predictor     → CPU_ONLY      fp16   15 MB
fused_f0n_har_source   → CPU_ONLY      fp32   32 MB   ← Trial 6 (kept fp32: cumsum drift)
decoder_pre            → CPU_AND_NE    fp16   64 MB
decoder_upsample       → CPU_ONLY      fp16   40 MB
```

Total: **274 MB**, 8 mlpackages, 8 dispatches per utterance.

## Performance

Warm pipeline-stage sum (sum of per-stage timings reported by
`coreml.inference`), 3-iter sweep with 8 s cooldown, M-series Mac:

| Build           | min  | avg  | max   |
|-----------------|------|------|-------|
| iteration_2 fp32| 782  | 898  | 1075  |
| iteration_3     | **460** | **683** | 1110 (thermal) |

Cool-run delta: **−322 ms (−41 %)** at min, **−215 ms (−24 %)** at avg.
The max bucket bunches because pipeline-wide variance dominates any
config — same pattern observed in Trial 8b benches.

Per-stage savings observed end-to-end:

| stage                   | fp32 ms | fp16 ms | Δ        |
|-------------------------|---------|---------|----------|
| fused_diffusion_sampler | 18.3    | 14.7    | −3.6 ms  |
| decoder_pre             | 35      | 7       | −28 ms   |
| decoder_upsample        | 593–638 | 284–325 | **−309 ms** |

## Mixed precision rationale

| Stage                   | fp16 verdict        | Why                                     |
|-------------------------|---------------------|-----------------------------------------|
| text_encoder            | adopt               | clean A/B                               |
| bert                    | adopt               | clean A/B                               |
| ref_encoder             | adopt               | clean A/B                               |
| fused_diffusion_sampler | adopt               | parity 4.66e-3, A/B clean                |
| duration_predictor      | adopt               | clean A/B                               |
| fused_f0n_har_source    | **drop**            | har computes sin(2π·cumsum(f0)) over 88 200 samples; fp16 cumsum drifts ~10 bits, audible phase distortion in second half |
| decoder_pre             | adopt               | parity tight, A/B clean                  |
| decoder_upsample        | adopt               | A/B clean; previously feared "+240 ms" regression on `ALL` did not reproduce on `CPU_ONLY` placement (this is the 8b-winning placement) |

Drift evidence comes from per-stage CoreML parity vs eager fp32 plus
direct A/B listening of three configurations:

```
sanity_fp16_mixed.wav         (5 fp16 / 3 fp32)   — clean
sanity_fp16_plus_decpre.wav   (6 fp16 / 2 fp32)   — clean
sanity_fp16_plus_decup.wav    (7 fp16 / 1 fp32)   — clean   ← this build
sanity_fp16_plus_f0n.wav      (8 fp16)            — degraded second half
```

## Storage

| Artifact                             | iteration_2 | iteration_3 |
|--------------------------------------|-------------|-------------|
| Total                                | 514 MB      | **274 MB** (−47 %) |
| largest stage                        | decoder_pre 128 MB | decoder_pre 64 MB |
| smallest stage                       | text_encoder 21 MB | text_encoder 11 MB |

## Usage

Same wiring as iteration_2 — `_STAGE_PRECISION` in `coreml/inference.py`
selects fp16 / fp32 per stage. No code changes, only the manifest values
flip:

```python
_STAGE_PRECISION: dict[str, str] = {
    "text_encoder":             "fp16",
    "bert":                     "fp16",
    "ref_encoder":              "fp16",
    "fused_diffusion_sampler":  "fp16",
    "diffusion_unet":           "fp32",  # legacy fallback
    "duration_predictor":       "fp16",
    "fused_f0n_har_source":     "fp32",  # cumsum drift
    "f0n_predictor":            "fp32",  # legacy fallback
    "har_source":               "fp32",  # legacy fallback
    "decoder_pre":              "fp16",
    "decoder_upsample":         "fp16",
}
```

CLI overrides still work:

```bash
# Re-run any stage at fp32 to A/B
python -m coreml.inference --fp32 decoder_upsample

# Drop back to iteration_2 wholesale
python -m coreml.inference --fp32
```

## Skipped trials this iteration

| Stage                    | Reason for staying fp32                              |
|--------------------------|------------------------------------------------------|
| fused_f0n_har_source     | har_source cumsum drift over 88 200-sample window    |

Other quantization tiers (int8 weight-only, int4 palettization) deferred
to a future iteration — fp16 already pays for itself on disk and warm
latency.
