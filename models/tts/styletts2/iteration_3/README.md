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

## Token-axis buckets (Trial 11)

The `bert` and `fused_diffusion_sampler` packages reject `ct.RangeDim`
on the token axis (HF Albert + cross-attn produce ops MIL refuses with
"data-dependent shapes were disabled"). The default packages above
hard-code T = 57, which caps prompts at ~37 chars.

**RangeDim removal: DEAD END.** Multiple attempts to lift the token
axis to `ct.RangeDim` for these two stages were closed out (HF Albert's
embedding-layer shape ops + the U-Net's cross-attention `reshape_like`
chains both bottom out in MLProgram's "data-dependent shapes were
disabled" guard). Token-axis bucketing is the production answer; the
buckets below are not a stopgap.

To support longer prompts without RangeDim, this iteration ships
**three additional fixed-T variants** of each constrained stage:

| File                                              | Compute      | Size  |
|---------------------------------------------------|--------------|-------|
| `bert_fp16_t64.mlpackage`                         | ALL          | 12 MB |
| `bert_fp16_t128.mlpackage`                        | ALL          | 12 MB |
| `bert_fp16_t256.mlpackage`                        | ALL          | 12 MB |
| `fused_diffusion_sampler_fp16_t64.mlpackage`      | ALL          | 48 MB |
| `fused_diffusion_sampler_fp16_t128.mlpackage`     | ALL          | 48 MB |
| `fused_diffusion_sampler_fp16_t256.mlpackage`     | ALL          | 48 MB |
| **Sub-total (extra over the 8 defaults)**         |              | **180 MB** |

The original `bert_fp16.mlpackage` / `fused_diffusion_sampler_fp16.mlpackage`
(T = 57) remain in the manifest as the default fast path — every
sentence that fits T = 57 should keep using them. The bucketed variants
are loaded on demand for longer prompts.

Loader policy (Python — Swift consumer lives in FluidAudio):

```
real_n = #espeak tokens
if   real_n <=  57: use *_fp16.mlpackage          (default)
elif real_n <=  64: use *_fp16_t64.mlpackage
elif real_n <= 128: use *_fp16_t128.mlpackage
elif real_n <= 256: use *_fp16_t256.mlpackage
else: error (extend the bucket ladder)
```

Pad the token + attention_mask tensors with zeros to the chosen
bucket's T. `bert` honours `attention_mask`, so contamination at
padded positions is bounded; the sampler attends to bert output, so
it inherits the same masking.

Per-bucket end-to-end inference verified by `coreml/inference_buckets.py
--all` (writes `coreml/out_t{64,128,256}.wav`):

| Bucket | Prompt                                     | Tokens | Audio  | Pipeline |
|--------|--------------------------------------------|--------|--------|----------|
| 64     | "Hello there. How are you today?"          | 36     | 2.42 s |  494 ms  |
| 128    | "StyleTTS 2 is a text to speech model."    | 57     | 3.60 s |  414 ms  |
| 256    | longer paragraph (see `inference_buckets.py`) | 154 | 8.37 s | 4933 ms  |

T = 256 cost is dominated by `decoder_upsample` at 4.5 s / 4.9 s
(real-time-ish CPU_ONLY at 24 kHz × 8.4 s output). Bucket-swap cost
itself is a few ms; the rest of the pipeline scales with output
frame count, not bucket size.

**Total iteration_3 footprint with buckets: 451 MB** (274 MB defaults
+ 180 MB buckets), or skip the T = 57 defaults entirely and ship only
buckets to save ~60 MB.

### Build / refresh the bucketed packages

```bash
cd models/tts/styletts2

# Build buckets (writes to coreml/packages/, run once)
uv run python coreml/exporters/build_buckets.py \
    --buckets 64,128,256 --stages bert,sampler --precision fp16

# Stage into iteration_3 + compile
for T in 64 128 256; do
  for stage in bert fused_diffusion_sampler; do
    cp -R "coreml/packages/${stage}_fp16_t${T}.mlpackage" \
          "iteration_3/packages/${stage}_fp16_t${T}.mlpackage"
    xcrun coremlcompiler compile \
      "iteration_3/packages/${stage}_fp16_t${T}.mlpackage" \
      "iteration_3/compiled/"
  done
done

# Validate
uv run python coreml/inference_buckets.py --all --output-dir coreml
```

### HuggingFace upload manifest

Upload the entire `iteration_3/packages/` tree (14 mlpackages):

```
iteration_3/packages/
├── text_encoder_fp16.mlpackage
├── bert_fp16.mlpackage                              ← T=57 default
├── bert_fp16_t64.mlpackage                          ← bucket
├── bert_fp16_t128.mlpackage                         ← bucket
├── bert_fp16_t256.mlpackage                         ← bucket
├── ref_encoder_fp16.mlpackage
├── fused_diffusion_sampler_fp16.mlpackage           ← T=57 default
├── fused_diffusion_sampler_fp16_t64.mlpackage       ← bucket
├── fused_diffusion_sampler_fp16_t128.mlpackage      ← bucket
├── fused_diffusion_sampler_fp16_t256.mlpackage      ← bucket
├── duration_predictor_fp16.mlpackage
├── fused_f0n_har_source.mlpackage                   ← fp32 (cumsum drift)
├── decoder_pre_fp16.mlpackage
└── decoder_upsample_fp16.mlpackage
```

Total: **451 MB** (12 fp16 stages + 1 fp32 stage + 1 cumsum-sensitive
stage). Compiled `.mlmodelc` siblings live next to the packages in
`iteration_3/compiled/` — same file count, same total size.
