# Cohere Transcribe — Encoder Bucket Plan (ANE-friendly variable-length input)

Status: design, not implemented. Follow-up to the INT8-encoder / FP16-decoder
hybrid pipeline landed on
`docs/cohere-transcribe-coreml-decoder-fix`.

## Problem

The current `cohere_encoder.mlmodelc` has a **fixed 35-second input**:

```
n_mels      = 128
max_frames  = 3500      # 35 s at hop_length=160, sr=16000
input shape = (1, 128, 3500)
```

Every audio sample — regardless of its real duration — is zero-padded to 3500
frames before the encoder runs. The downstream mask (`feature_length`) prevents
the padded region from contributing to the output, but the compute has already
been spent.

Measured on FLEURS (14 Cohere-supported languages,
`benchmark_results/cohere_max10`, 140 samples):

| Stat | Value |
|---|---|
| Mean audio duration | ~11.0 s |
| Padded chunk length | 35.0 s (fixed) |
| Wasted compute fraction | ~69% |
| Macro RTFx (INT8 encoder, FP16 decoder, ANE) | 1.62× |
| FLEURS samples ≤ 5 s | ~15% |
| FLEURS samples ≤ 10 s | ~55% |
| FLEURS samples ≤ 20 s | ~90% |
| FLEURS samples > 20 s | ~10% |

Most of our inference cost is zero propagation. We'd like to only pay for the
audio actually present.

## Constraints

From `mobius/CLAUDE.md`:

> Fixed input shapes only (no dynamic dimensions)

This matches observed ANE behaviour:

| CoreML shape mode | ANE? | Notes |
|---|---|---|
| Fixed (`ct.TensorType(shape=(...))`) | ✅ full ANE | one size per `.mlmodelc` |
| Enumerated (`EnumeratedShapes`) | ⚠️ partial | ANE support is brittle, mobius team has seen silent CPU fallback |
| Range (`RangeDim`) | ❌ no | GPU/CPU only; kills the speedup we're trying to achieve |

"Maximum ANE residency" rules out dynamic/range shapes. We also don't want to
depend on enumerated-shapes behaving correctly across Xcode / macOS versions.

## Proposal: four fixed-shape buckets

Export and ship **four separate `.mlmodelc` encoders**, each with a different
`target-frames` value. At inference time, pick the smallest bucket that fits
the actual audio.

```
cohere_encoder_q8_f500.mlmodelc    →  5.0 s  (XS)
cohere_encoder_q8_f1000.mlmodelc   → 10.0 s  (S)
cohere_encoder_q8_f2000.mlmodelc   → 20.0 s  (M)
cohere_encoder_q8_f3500.mlmodelc   → 35.0 s  (L, today's model)
```

### Bucket sizing rationale

| Bucket | Frames | Max audio | FLEURS hit rate | Expected avg compute vs L |
|:--:|--:|--:|--:|--:|
| XS | 500 | 5.0 s | ~15% | 0.14× |
| S | 1000 | 10.0 s | ~55% | 0.29× |
| M | 2000 | 20.0 s | ~90% | 0.57× |
| L | 3500 | 35.0 s | 100% | 1.00× (today) |

Expected macro compute reduction on FLEURS (weighted by bucket hit rate):

```
0.15 × 0.14  +  0.40 × 0.29  +  0.35 × 0.57  +  0.10 × 1.00
= 0.021 + 0.116 + 0.200 + 0.100
= 0.437
```

**Expected ~2.3× speedup in encoder wall-clock on FLEURS avg audio**, lifting
macro RTFx from ~1.6× to somewhere in the 3–4× range (decoder is the remaining
tax).

Real-world streaming traffic (VoiceLink) will skew even shorter than FLEURS —
most dictations are < 10 s — so S-bucket hit rate in production may be higher
than 40%.

### Why four buckets?

- Two (S + L) leaves too much padding on 10–20 s audio (~30% of FLEURS).
- Eight is overkill: Conformer encoder cost is roughly linear in frames after a
  constant prelude, so gains from further splitting decay fast.
- Four hits the knee: XS and S cover short dictation / streaming chunks, M
  covers long utterances common in meetings / lectures, L stays as the
  correctness anchor (guaranteed to fit anything Cohere's pre-training saw).

### Disk / RAM cost

Current single INT8 encoder `.mlmodelc` on disk: ~350 MB (quantized from ~700 MB
FP16). The Conformer weights are shared; the only thing that varies per bucket
is the input tensor shape and a few shape-specialized kernels. Rough estimate:

| Item | Size | Notes |
|---|---:|---|
| `.mlmodelc` on disk, per bucket | ~350 MB | weights dominate, dense shapes marginal |
| 4 buckets on disk | ~1.4 GB | acceptable for a desktop/server app |
| ANE kernel cache, per bucket | ~40–80 MB | one-time compile per bucket |
| Resident RAM, 1 bucket loaded | ~350 MB | one encoder at a time |
| Resident RAM, all 4 preloaded | ~1.4 GB | optional; see "Loading strategy" below |

iOS deployments will want to gate this: ship only L + S, or lazy-load buckets
on first use. Not a blocker for the macOS / desktop use case that drove this
work.

## Swift-side routing

Sketch (not wired yet; goes in the FluidAudio PR follow-up once bucket models
exist):

```swift
static let bucketFrames = [500, 1000, 2000, 3500]

func selectBucket(audioSamples: Int) -> Int {
    // audioSamples / hopLength, rounded up, then find smallest bucket ≥ that.
    let neededFrames = (audioSamples + CohereAsrConfig.MelSpec.hopLength - 1)
        / CohereAsrConfig.MelSpec.hopLength
    return bucketFrames.first(where: { $0 >= neededFrames }) ?? bucketFrames.last!
}
```

### Loading strategy

Three options, from lightest to heaviest:

1. **Lazy + singleton**: keep at most one bucket loaded; swap on mismatch. RAM
   stays at ~350 MB, but consecutive short-then-long audio pays bucket swap
   cost (~200 ms ANE cache warm-up).
2. **L always resident + opportunistic smaller**: keep L as a guaranteed
   fallback, load S/M on demand. Good for mixed workloads.
3. **Preload all 4**: ~1.4 GB RAM but zero swap cost. Fine for servers and
   desktops, probably not for iOS.

Recommendation: start with (1) for the benchmark, measure swap cost, escalate
to (2) if it dominates.

### Cross-bucket correctness

The bucket choice is transparent to the decoder: encoder outputs are
`(1, N, 1024)` where N scales with `target_frames`. The FP16 decoder already
handles variable-length encoder outputs via cross-attention. No decoder
changes needed.

One thing to verify during bucket export: downsampling factor must match
across buckets so `N / target_frames` is constant. The Conformer uses 4×
subsampling → `500→125, 1000→250, 2000→500, 3500 → ~438`. All integer, all
consistent. Safe.

## Export workflow change

### Today

`export-encoder-ios18.py` hardcodes `max_frames = 3500` (line 79).

Output filename: `cohere_encoder.mlpackage`, one per run.

### After this plan

`export-encoder-ios18.py` accepts `--target-frames N`.

```bash
for frames in 500 1000 2000 3500; do
    uv run python export-encoder-ios18.py \
        --target-frames "$frames" \
        --output-dir "build/f${frames}"
done
```

Output filename encodes the bucket: `cohere_encoder_f500.mlpackage`,
`cohere_encoder_f1000.mlpackage`, etc. The existing INT8 quantization
(`tools/quantize_to_int8.py`) and the `compile_encoder_to_mlmodelc.py` step
run per-bucket unchanged; they don't care about the input shape.

A companion change to `export-encoder-ios18.py` lands alongside this doc —
see commit `feat(export): parametrize encoder target-frames` on the same
branch.

## Validation plan (not this PR)

Before shipping buckets we need parity + speedup measurements:

1. **Correctness**: run all 4 bucket exports through
   `compare-models.py` on 20 FLEURS samples per language. Output hidden
   states must match the L-bucket baseline within FP16 tolerance for audio
   that fits in the smaller bucket. Any drift indicates a numerical issue
   with the shape-dependent kernel variant.

2. **Latency**: profile each bucket with `coreml-cli` to confirm full ANE
   residency. If any bucket shows CPU fallback (e.g., shape-dependent
   reshape ops refusing to compile), debug before shipping.

3. **End-to-end**: re-run `fluidaudiocli cohere-mixed-benchmark` with a
   bucket-aware FluidAudio build on FLEURS and compare WER + RTFx vs the
   current single-L baseline. WER must be flat within noise; RTFx should
   rise to the predicted 3–4× macro.

4. **Bucket-swap cost**: measure wall-clock of 100 alternating short↔long
   inferences to quantify option-(1) swap overhead and decide if option (2)
   preloading is needed.

## Out of scope

- Dynamic encoder (we explicitly rejected `RangeDim` above).
- Encoder-only distillation (would reduce compute across all buckets but is a
  training problem, not an export problem).
- Decoder changes. The decoder KV cache + 108-token limit is a separate
  bottleneck (see `RESEARCH_INSIGHTS.md` §1) and should be attacked on its
  own.
- FluidAudio Swift routing layer. Included as a sketch only; the full
  implementation is a follow-up PR in FluidAudio once bucket `.mlmodelc`
  files exist.

## Open questions

1. Should the buckets live in the same HuggingFace repo as a 4-file bundle,
   or as 4 separate revisions? Leaning "same repo, parallel filenames" so
   the FluidAudio downloader can fetch the set in one call.
2. Do we want a 250-frame (2.5 s) XXS bucket for real-time streaming? That
   would cover VAD-segmented short utterances in `StreamingAsrManager` more
   efficiently. Probably yes, but defer to after the initial four land.
3. Bucket boundary at 1000 / 2000 frames — should we snap to the nearest
   Conformer-friendly value (multiple of the 4× subsampling stride)? 1000
   and 2000 are both already divisible by 4, so we're fine.

## References

- Cohere Transcribe official 35 s window:
  https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- `RESEARCH_INSIGHTS.md` — decoder bottleneck analysis (paper [4]).
- `mobius/CLAUDE.md` — ANE fixed-shapes rule.
- FLEURS benchmark results: `FluidAudio/benchmark_results/cohere_max10/`
  (and forthcoming `cohere_full/`).
