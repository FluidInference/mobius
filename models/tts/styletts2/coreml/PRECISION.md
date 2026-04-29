# StyleTTS2 Precision

How the StyleTTS2 CoreML artifacts are placed across compute units, and
why the decoder is fp32 while the rest of the pipeline is fp16.

References:
- Upstream: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2),
  LibriTTS multi-speaker checkpoint (`Models/LibriTTS/epochs_2nd_00020.pth`).
- Decoder fp32 rationale: PHASE6_FP16_DECODER.md.
- Conversion log: TRIALS.md.

---

## TL;DR

Four stages, three precisions. fp16 everywhere except the decoder.

| Artifact                                | Precision | Compute unit | Buckets                              | Called      |
|-----------------------------------------|-----------|--------------|--------------------------------------|-------------|
| `styletts2_text_predictor_{B}.mlpackage` | fp16     | ANE          | `B ∈ {32, 64, 128, 256, 512}` token | 1× per utt  |
| `styletts2_diffusion_step_512.mlpackage` | fp16     | CPU+GPU      | 1 bucket (512)                       | 5× per utt  |
| `styletts2_f0n_energy.mlpackage`        | fp16      | ANE          | dynamic                              | 1× per utt  |
| `styletts2_decoder_{M}.mlpackage`       | **fp32**  | CPU+GPU      | `M ∈ {256, 512, 1024, 2048, 4096}` mel | 1× per utt |

On-disk size (LibriTTS checkpoint): **~1.3 GB**. Warm RTFx **4.32×** on
M-series Mac. Log-mel cosine vs PyTorch fp32: **0.9687**.

---

## Why the decoder is fp32 and everything else is fp16

The HiFi-GAN decoder's SineGen harmonic source accumulates phase via
`cumsum × 2π × hop=300`, reaching magnitudes ~4000 mid-frame. fp16
precision at that magnitude (~4) is much larger than the per-sample
phase increment (~0.05 rad), so the sine output collapses to a few
discrete values and the synthesizer produces audibly robotic audio.
Full analysis in PHASE6_FP16_DECODER.md.

The other three stages (text_predictor, diffusion_step, f0n_energy) do
not have a phase-accumulator and run cleanly at fp16 with cosine ≥ 0.99
versus the PyTorch fp32 reference per stage.

---

## Why not int8?

Selective int8 PTQ on the text_predictor was tried (recipe:
`coremltools.optimize.coreml.linear_quantize_weights`,
`weight_threshold=200_000`, `mode=linear_symmetric`,
`granularity=per_channel`) and dropped before ship:

- The Apple Silicon ANE has no exposed int8 GEMM. int8 weights are
  dequantized to fp16 on read; the only win is DRAM→SRAM bandwidth.
- The bandwidth payoff was ~3 MB per bucket (~15 MB total across the
  five buckets) — small relative to the rest of the bundle.
- Per-channel scales on the smaller projections were fragile across the
  bucket family; some buckets needed a higher `weight_threshold` than
  others to preserve parity, and the ship matrix was not worth the
  validation cost.

The diffusion step was never a quantization candidate: it iterates 5×
per utterance through an ODE-style sampler, so quantization noise
compounds (PocketTTS issue #7's failure mode). f0n_energy is 6 MB,
nothing to save.

The historical recipe (`scripts/optimize/quantize_text_predictor_int8.py`)
is kept in tree as reference; it is not part of the build/ship pipeline.

---

## Compute-unit placement

The split below was found by sweeping `compute_units` per stage and
measuring warm-cache RTFx:

| Stage          | Best unit   | Why                                                            |
|----------------|-------------|----------------------------------------------------------------|
| text_predictor | **ANE**     | BiLSTM + projections; small per-call cost; ANE-friendly.       |
| diffusion_step | **CPU+GPU** | Attention block at `(1, 512, dim)` — ANE rejects parts of the graph and falls back, paying double. CPU+GPU gives a single consistent path. |
| f0n_energy    | **ANE**     | Tiny LSTM + linear; ANE-friendly.                              |
| decoder       | **CPU+GPU** | HiFi-GAN's transposed-conv upsampling stalls on ANE; CPU+GPU is faster end-to-end. |

Sweep result: **RTFx 1.61× → 3.80× → 4.32×** (worst-uniform-CU →
best-per-stage → after warmup of every bucket on app launch).

---

## Bucket strategy

| Stage          | Bucket axis     | Buckets shipped                  | Notes                                |
|----------------|-----------------|----------------------------------|--------------------------------------|
| text_predictor | input tokens    | 32, 64, 128, 256, 512            | All fp16.                            |
| diffusion_step | bert_dur frames | **512 only**                     | Pruned 32/64/128/256 (saved 192 MB). |
| f0n_energy    | dynamic shape   | n/a                              | Single package.                      |
| decoder       | mel frames      | 256, 512, 1024, 2048, 4096       | All fp32 (fp16 → robotic audio).     |

The diffusion bucket prune is documented in TRIALS.md Phase 4. We never
observed a per-utterance `bert_dur` length below the 512 bucket in any
LibriTTS or vendored StyleTTS2 sample, so the smaller buckets were dead
weight. The non-linear cost ladder (per-step duration: B=32 66 ms,
B=256 143 ms, B=512 152 ms) means jumping all short utterances to the
512 bucket adds at most ~430 ms per utterance — well below the ~1.4 s
saved by avoiding cold-bucket recompilation in steady state.

---

## Build and ship

```bash
cd models/tts/styletts2
uv run python scripts/00_fetch_weights.py
uv run python scripts/01_export_text_predictor.py     # 5 fp16 buckets
uv run python scripts/02_export_diffusion_step.py     # 1 fp16 bucket (512)
uv run python scripts/03_export_f0n_energy.py
uv run python scripts/04_export_decoder.py            # 5 fp32 buckets
uv run python scripts/99c_e2e_optimized.py            # warm-cache RTFx + log-mel parity
```

After `99c_e2e_optimized.py` finishes the `coreml/` directory holds the
shippable artifacts: 5× fp16 text_predictor, 1× fp16 diffusion_step,
1× fp16 f0n_energy, 5× fp32 decoder.

Speaker-ID forensics (TRIALS.md Phase 5) confirm the decoder/precision
recipe is **not** what makes outputs sound "robotic" relative to
upstream PyTorch — the clone-fidelity ceiling is architectural in
StyleTTS2 itself; PyTorch fp32 only achieves ECAPA-TDNN cosine 0.29 to
its own reference clip, and our shipped pipeline reaches 0.18.
