# StyleTTS2 Precision and Quantization

How the StyleTTS2 CoreML artifacts are quantized and placed across compute
units, why specific stages are deliberately left at fp16, and why this
mixed-precision recipe preserves quality at the model's voice-clone ceiling.

References:
- Upstream: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2),
  LibriTTS multi-speaker checkpoint (`Models/LibriTTS/epochs_2nd_00020.pth`).
- Quantization recipe family: `coremltools.optimize.coreml.linear_quantize_weights`
  with `weight_threshold` (same primitive used by PocketTTS — see
  `models/tts/pocket_tts/coreml/PRECISION.md`).

---

## TL;DR

We ship four stages. Three are fp16, one is selectively int8.

| Artifact                                | Precision           | Compute unit | Buckets                      | Called           |
|-----------------------------------------|---------------------|--------------|------------------------------|------------------|
| `styletts2_text_predictor_{B}.mlpackage` | **selective int8** | ANE          | `B ∈ {32, 64, 128, 256, 512}` token | 1× per utt      |
| `styletts2_diffusion_step_512.mlpackage` | fp16               | CPU+GPU      | 1 bucket (512)               | 5× per utt       |
| `styletts2_f0n_energy.mlpackage`        | fp16                | ANE          | dynamic                      | 1× per utt       |
| `styletts2_decoder_{M}.mlpackage`       | **fp32**            | CPU+GPU      | `M ∈ {256, 512, 1024, 2048, 4096}` mel | 1× per utt |

Final on-disk size (LibriTTS checkpoint): **~1.4 GB**. Decoder is fp32
because fp16 produces robotic audio (the SineGen harmonic source's
cumsum-then-multiply phase chain saturates fp16 precision; see
PHASE6_FP16_DECODER.md). Other stages (text_predictor, diffusion_step,
f0n_energy) ship at int8/fp16 as documented. Warm RTFx **4.32×** on
M-series Mac. Log-mel cosine vs PyTorch fp32: **0.9687**.

---

## Why selective and not blanket

StyleTTS2 is a four-stage pipeline; only one stage is large enough that
8-bit weights help, and only some compute placements actually run on
ANE without graph rejection.

| Stage                   | Bytes (fp16) | int8 win? | ANE OK? | Decision               |
|-------------------------|--------------|-----------|---------|------------------------|
| text_predictor (5 buckets) | ~178 MB     | yes (−89 MB) | yes  | **int8 + ANE**         |
| diffusion_step (per bucket) | ~48 MB      | tiny — Mish/AdaIN-1d, mostly conv | no — large attention block falls off ANE  | fp16 + CPU+GPU         |
| f0n_energy              | ~6 MB        | none — small  | yes  | fp16 + ANE             |
| decoder (5 buckets)     | ~210 MB each (fp32) | conv-heavy, low payoff; **fp16 produces robotic audio** (SineGen phase saturation, see PHASE6_FP16_DECODER.md) | no — HiFi-GAN upsampling stalls ANE | **fp32** + CPU+GPU |

This mirrors the PocketTTS rule: "quantize the big GEMM-heavy stage,
leave conv stacks and small/iterative stages alone."

---

## Method: weight-only PTQ via `coremltools.optimize.coreml`

```python
# scripts/optimize/quantize_text_predictor_int8.py
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

op_cfg = OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int8",
    granularity="per_channel",
    weight_threshold=200_000,   # only weights with ≥200k elements
)
config = OptimizationConfig(global_config=op_cfg)

mlmodel = ct.models.MLModel(f"styletts2_text_predictor_{B}.mlpackage")
quantized = linear_quantize_weights(mlmodel, config=config)
quantized.save(f"styletts2_text_predictor_{B}_int8.mlpackage")
```

What this produces:

- Only the BiLSTM linear / projection weights with ≥200k elements are
  stored as int8 with a per-output-channel fp16 scale. Embedding tables,
  small projections, normalization scales, and bias vectors stay fp16.
- At inference the ANE dequantizes to fp16 on read. **Activations stay
  fp16.** The win is bandwidth (DRAM→SRAM), not GEMM throughput — the
  ANE has no exposed int8 path.
- `weight_threshold=200_000` is load-bearing. Drop it and the converter
  pulls in tiny matrices (LayerNorm-adjacent projections, the duration
  predictor head, the BERT pooling) and parity collapses (we observed
  log-mel cosine drop from 0.9998 → < 0.9 in early sweeps).

Same numerical recipe as PocketTTS `flowlm_stepv2`, just expressed
through the CoreML-graph API (`ct.optimize.coreml.linear_quantize_weights`)
instead of the torch-graph API (`ct.optimize.torch...PostTrainingQuantizer`).
The CoreML graph already has the structure we want; we don't need to go
through PyTorch a second time.

|                       | Upstream torch.ao dynamic    | Ours (CoreML weight-only PTQ) |
|-----------------------|------------------------------|-------------------------------|
| Weight storage        | n/a (StyleTTS2 has no int8 upstream) | int8                  |
| Activation precision  | n/a                          | fp16                          |
| GEMM kernel           | n/a                          | fp16 × fp16 → fp16            |
| Backend               | n/a                          | Apple Silicon ANE fp16        |
| Speedup source        | n/a                          | weight bandwidth              |

---

## Why the excluded layers stay fp16

### Diffusion step (the iterative one)

ADPM2 + Karras schedule runs **5 sampler steps** per utterance, each
calling `styletts2_diffusion_step_512.mlpackage`. The denoiser is a
DiT-style transformer over `(1, 512, dim)` style tokens. Two reasons not
to int8 it:

1. **Quantization noise compounds.** Same lesson as VoxCPM Trial 19:
   int8 caches feeding back through 5 iterations of an ODE-style solver
   accumulates drift. PocketTTS issue #7 documents the same failure mode
   ("the iterative LSD denoiser amplifies quantization error step over
   step").
2. **It's already small.** One bucket × ~48 MB. The bandwidth payoff is
   ~24 MB; the parity risk is the entire utterance.

### F0N energy

6 MB. Per-frame F0 + voicedness regression. Quantizing per-channel
collapses small-output projections; quantizing the LSTM hidden-state
projection injects pitch noise that audibly chirps. Trivial size, leave
alone.

### Decoder (HiFi-GAN)

Convolutional upsampler (300× hop). PocketTTS PRECISION.md documents
why we don't quantize Mimi's conv VAE; the same argument applies to
HiFi-GAN: conv kernels have less weight mass than transformer FFNs
(lower bandwidth payoff per parameter), and upsampling is more
sensitive to quantization than matmul because each int8 weight is
re-used across the upsample stride.

### Embedding / projection / LayerNorm in `text_predictor`

`weight_threshold=200_000` excludes them automatically. Per-channel
int8 on small matrices either degenerates to per-tensor (output dim
< 32) or quantizes parameters that have already been carefully
normalized (LayerNorm γ/β).

---

## Compute-unit placement

The split below was found by sweeping `compute_units` per stage and
measuring warm-cache RTFx:

| Stage          | Best unit   | Why                                                            |
|----------------|-------------|----------------------------------------------------------------|
| text_predictor | **ANE**     | BiLSTM + projections; small per-call cost; ANE handles it cleanly. |
| diffusion_step | **CPU+GPU** | Attention block at `(1, 512, dim)` — ANE rejects parts of the graph and falls back, paying double. CPU+GPU gives a single consistent path. |
| f0n_energy    | **ANE**     | Tiny LSTM + linear; ANE-friendly.                              |
| decoder       | **CPU+GPU** | HiFi-GAN's transposed-conv upsampling stalls on ANE; CPU+GPU is faster end-to-end. |

Sweep result: **RTFx 1.61× → 3.80× → 4.32×** (worst-uniform-CU →
best-per-stage → after warmup of every bucket on app launch).

---

## Bucket strategy

| Stage          | Bucket axis     | Buckets shipped                  | Notes                                |
|----------------|-----------------|----------------------------------|--------------------------------------|
| text_predictor | input tokens    | 32, 64, 128, 256, 512            | All 5 quantized.                     |
| diffusion_step | bert_dur frames | **512 only**                     | Pruned 32/64/128/256 (saved 192 MB). |
| f0n_energy    | dynamic shape   | n/a                              | Single package.                      |
| decoder       | mel frames      | 256, 512, 1024, 2048, 4096       | All fp32 (fp16 → robotic audio; see PHASE6_FP16_DECODER.md). |

The diffusion bucket prune is documented in TRIALS.md Phase 4. The
short-version: we never observed a per-utterance bert_dur length below
the 512 bucket in any LibriTTS or vendored StyleTTS2 sample, so the
smaller buckets were dead weight. The non-linear cost ladder (per-step
duration: B=32 66 ms, B=256 143 ms, B=512 152 ms) means jumping all
short utterances to the 512 bucket adds at most ~86 ms × 5 steps ≈ 430
ms per utterance — well below the ~1.4 s saved by avoiding cold-bucket
recompilation in steady state.

---

## Why this works

Three reasons the recipe is robust:

1. **The hot tensor really is the text_predictor BiLSTM.** It's the
   only stage with ≥200k-element weights that runs at scale across
   the bucket family. fp16 → int8 saves 89 MB out of ~178 MB
   (50%), without touching anything iterative.

2. **The iterative stage stays fp16.** PocketTTS issue #7's failure
   mode (int8 noise compounding through a denoiser loop) cannot
   trigger because `diffusion_step` is fp16 end to end.

3. **Per-channel granularity matches the geometry.** The BiLSTM
   linears have output dim ≥ 256, so per-channel scales preserve
   each row's dynamic range. The 200k threshold automatically
   excludes everything where per-channel would degenerate.

Empirical: log-mel cosine vs PyTorch fp32 reference is **0.9998** for
text_predictor int8 with all 5 diffusion buckets, **0.9687** after
adding the diffusion-bucket prune (a separate, fp16-only effect).
Speaker-ID forensics (TRIALS.md Phase 5) confirm quantization is
**not** what makes outputs sound "robotic" — the clone-fidelity
ceiling is architectural in StyleTTS2 itself; PyTorch fp32 only
achieves ECAPA-TDNN cosine 0.29 to its own reference clip.

---

## Build and ship

```bash
cd models/tts/styletts2
uv run python scripts/00_fetch_weights.py
uv run python scripts/01_export_text_predictor.py     # 5 fp16 buckets
uv run python scripts/02_export_diffusion_step.py     # 1 fp16 bucket (512)
uv run python scripts/03_export_f0n_energy.py
uv run python scripts/04_export_decoder.py            # 5 fp32 buckets
uv run python scripts/optimize/quantize_text_predictor_int8.py  # fp16 → int8 ×5
uv run python scripts/99c_e2e_optimized.py            # warm-cache RTFx + log-mel parity
```

After `99c_e2e_optimized.py` finishes the `coreml/` directory holds the
shippable artifacts: 5× int8 text_predictor, 1× fp16 diffusion_step,
1× fp16 f0n_energy, 5× fp32 decoder. The fp16 text_predictor packages
are kept as build intermediates (gitignored, not shipped) so we can
re-quantize without re-tracing.
