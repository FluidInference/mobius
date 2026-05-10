# Magpie TTS — Performance & Optimization Log

Single source of truth for Magpie TTS latency tuning on Apple Silicon.
All numbers on **Apple M2 / macOS 26.5 / coremltools 9.0** unless noted.
"TTFA" = wall-clock from `synthesizeStream(...)` call to first
`MagpieAudioChunk` yield, warm path, release build, seed 42.

> **Layout note** — production conversion scripts live at the top of
> `coreml/`. Diagnostic / dead-end scripts (Trial 2 int8 quant, Trial 4a
> N=2 unroll, STATEFUL MLState variant) live in
> [`experiments/`](experiments/); the separate NanoCodec sweep track is
> in [`nanocodec_experiments/`](nanocodec_experiments/). All path refs
> below assume you `cd` into `coreml/`.

---

## TL;DR — current state

| Lever | Status | Δ TTFA | Notes |
|---|---|---|---|
| #1 Streaming first-chunk cap 50→24 | ✓ shipped | **−978 ms** (−38.7%) | pure Swift, no model touched |
| #3 Local Transformer fusion | ✓ shipped | **−161 ms** (−12.5%) on short input | one ANE call replaces 8 Swift CPU passes |
| #2 int8 weight quant on `decoder_step` | ✗ dead | — | EOS classifier breaks on long inputs |
| #5 `speakerContextLength` 110→64 | ✗ dead | +1.7 s regression | naive truncation breaks voice cloning |
| #7 First-chunk cap 24→16 | ✗ dead | 0 ms | no win, UX regression |
| #10a NanoCodec v4 (palette) | ✗ dead | — | slower than v3 across all policies on M2 |
| #10b NanoCodec split first call | ✗ dead | — | first call still required, second-chunk gap problem |
| #10c NanoCodec mixed-precision | ✗ dead | — | Phase F.2 sweep — fp32 weights required for clean audio |
| #4 AR loop unroll N=2 | ✗ dead | +30 ms regression | sampler tail (cumsum/topk/equal/softmax) forces ANE↔CPU thrash; 64.77 ms vs 34.6 ms target |
| #8 QAT/calibration int8 via NeMo | ⏸ deferred | est. −400 ms | multi-day project, only remaining ceiling lever |

**Net stack**: ~830 ms warm TTFA on M2 (Levers #1 + #3 active). All graph-level
levers exhausted on the Magpie 357M open-weight checkpoint; further wins require
QAT, model swap, or platform-side improvements.

---

## Architecture context (verified from NeMo source)

NeMo's open-source `MagpieTTSModel` (in `nemo/collections/tts/models/magpietts.py`)
is **batch-only**. Public inference surface is `infer_batch(...)` and
`generate_long_form_speech(...)` — both synchronous, both return a complete
audio tensor at the end. Zero `yield`/`async`/streaming primitives across
all 4 NeMo Magpie modules (`magpietts.py`, `magpietts_inference/inference.py`,
`magpietts_inference/utils.py`, `magpietts_modules.py`).

Streaming exists only in:
- **NVIDIA Riva / MagpieTTS NIM** (closed enterprise)
- **Our Swift `synthesizeStream(...)`** in `MagpieSynthesizer.swift:252` —
  custom producer/consumer wrapper. Sentence-chunks the text (same as
  NeMo's long-form), but **yields each chunk's audio as it finishes**
  instead of concatenating at the end.

Lever #1's first-chunk cap exists to fake streaming for single-sentence
input where there's only one chunk to yield.

---

## TTFA breakdown (warm, M2, 24-frame first chunk, seed 42)

Empirical per-model latency from `coreml-cli` with `cpu_and_neural_engine`
policy (matches Swift production config in `MagpieModelStore.swift:75-99`):

| Model | Predict | ANE % | Calls per first-chunk | Total |
|---|---|---|---|---|
| `text_encoder.mlmodelc` | 12.4 ms | 98.1% | 1 | 12 ms |
| `decoder_prefill.mlmodelc` | 17.1 ms | 93.9% | 1 | 17 ms |
| `decoder_step.mlmodelc` | 15.7 ms | 97.3% | 24 | 377 ms |
| `local_transformer.mlmodelc` | 1.6 ms | 73.9% | 24 | 38 ms |
| `nanocodec_decoder_v3.mlmodelc` | 182.3 ms | 0% (CPU) | 1 | 182 ms |
| **Compute total** | | | | **626 ms** |
| Measured warm TTFA | | | | **~830 ms** |
| **Implied Swift overhead** | | | | **~204 ms (24%)** |

Where the Swift overhead goes: 24 AR steps × 2 dispatch round-trips
(decoder_step + local_transformer) = 48 boundary crossings, plus
audio-embed lookup (vDSP), MLMultiArray allocations, KV-state I/O,
producer/consumer task hopping.

### Compute-unit selection (already optimal in Swift)

| Model | `all` Predict | `cpu_and_neural_engine` Predict | Penalty if `.all` |
|---|---|---|---|
| text_encoder | 17.9 ms | 12.4 ms | +5.5 ms |
| decoder_prefill | **70.7 ms** | 17.1 ms | **+53.6 ms** |
| decoder_step | 16.8 ms | 15.7 ms | +1.1 ms |
| local_transformer | 3.6 ms | 1.6 ms | +2.0 ms |
| nanocodec_v3 | 337.8 ms | 182.3 ms (CPU forced) | +155.5 ms |

Swift uses `.cpuAndNeuralEngine` per model in `MagpieModelStore.swift` —
verified correct. CoreML's `.all` chooser would mis-route prefill to GPU
(+54 ms) and NanoCodec to GPU (+155 ms). **No change needed** — flagging
in case a downstream consumer overrides the default.

---

## Model inventory

4 logical models + 4 NanoCodec build variants (only one active at runtime):

| File | Role | Compute | Status |
|---|---|---|---|
| `text_encoder.mlmodelc` | phoneme/IPA → encoder hidden states | ANE | required |
| `decoder_prefill.mlmodelc` | speaker-context prefill, batched 110-step | ANE | optional fast path |
| `decoder_step.mlmodelc` | AR transformer body, called per code-frame | ANE 97.3% | required |
| `local_transformer.mlmodelc` | fused 8-codebook sampler (Lever #3) | ANE 73.9% | optional, shipped |
| `nanocodec_decoder.mlmodelc` (v1) | T=256 monolithic, fp16 | CPU | legacy fallback |
| `nanocodec_decoder_v2.mlmodelc` | T=24 chunked, fp16 | ~43% ANE | noisy on voiced speech |
| `nanocodec_decoder_v3.mlmodelc` | T=24 chunked, fp32 | CPU | **default**, audibly clean |
| `nanocodec_decoder_v4.mlmodelc` | T=24 chunked, fp32 + 8-bit palette | CPU | acoustically transparent vs v3, 4× smaller (31 MB) |

---

## Trials & verdicts (chronological)

### Trial 1 — Streaming first-chunk cap 50 → 24 frames ✓ SHIPPED

`MagpieChunker.streamingFirstChunkCap` reduced from **50 → 24** codec frames
(matches NanoCodec's v2/v3 sliding-window receptive field).

| | Before | After | Δ |
|---|---|---|---|
| TTFA | 2 525 ms | 1 547 ms | **−38.7%** |
| Audio quality | clean | clean | unchanged |

Commit `b93f3b099` (FluidAudio). Pure Swift change, no model touched.
`24` is hard floor: NanoCodec v2/v3 use 24-frame chunk shape.

### Trial 2 — Post-training int8 weight quant on `decoder_step` ✗ DEAD-END

Three configs tried, all break end-to-end EOS termination on long-context
streaming inputs.

Script: `experiments/quantize_decoder_step_int8.py`

| Config | Short TTFA (3 words) | Long-input EOS |
|---|---|---|
| fp16 (baseline) | 1.48 s | ✓ @ 197 codes |
| int8 per-tensor | 6.84 s (regression) | **never fires**, runaway @ 500 |
| int8 per-channel | 0.87 s (−41%) ✓ | ✓ chunks 0-3, **runaway chunk 4** |
| per-channel + skip LM head | 0.92 s ✓ | **still runaway chunk 4** |

**Diagnosis**: drift accumulates in the int8 transformer **body** (12 layers
× ~300 generated codes of int8-perturbed activations), not localized to LM
head. Real fix requires QAT or activation-calibrated int8 export (Lever #8).

### Trial 3 — Local Transformer fusion ✓ SHIPPED

Standalone fused mlpackage (`local_transformer.mlpackage`) — separate from
`decoder_step` to avoid NeMo retrace. Bakes 8 unrolled LT iterations +
per-codebook softmax / cumsum / sample into a single CoreML graph.

```
before: decoder_step → 1×768 hidden → Swift LT (8 sequential CPU passes) → 8 ints
after:  decoder_step → 1×768 hidden → local_transformer.mlmodelc (1 ANE call) → 8 ints
```

| Build | ANE residency | Per-step |
|---|---|---|
| `local_transformer.mlpackage` (fp32) | 73.9% | 1.78 ms |

End-to-end TTFA, M2 release, 3 trials:

| Input | Swift LT (baseline) | Fused LT (ANE) | Δ |
|---|---|---|---|
| "Hello from Magpie." (3 words) | 1.283 s | **1.122 s** | **−161 ms (−12.5%)** |
| 14-word EN sentence | 4.243 s | 4.146 s | within noise |

Wiring: `MagpieFusedLocalSampler.swift`, `MagpieSynthesizer.swift`. CFG
path falls back to Swift `MagpieLocalSampler`. Unit tests in
`MagpieFusedLocalSamplerTests.swift` (env-gated by `FLUIDAUDIO_RUN_MAGPIE_LT_FUSED=1`).

Commits: `4b57ab27e` (Swift), `f3ba62e` (mobius mlpackage).

#### Quality validation (post-merge audio diff, M2)

Trial 3 originally cited "audio quality clean → clean unchanged" based on
listening only. Quantified post-hoc by an end-to-end same-seed diff:

Setup: `magpie text` CLI run twice with `--seed 42 --speaker 0 --language en`
on prompt _"The quick brown fox jumps over the lazy dog. Quality validation
in progress."_  Force-Swift run was produced by replacing
`local_transformer.mlmodelc` with an empty placeholder directory (causes
`MLModel(contentsOf:)` to throw → `MagpieModelStore.loadModel` returns nil
because the model is loaded with `required: false` →
`MagpieSynthesizer` falls back to `MagpieLocalSampler`). The empty
directory prevents `MagpieResourceDownloader` from re-fetching it.

| Metric | Fused (ANE) | Swift (CPU) |
|---|---|---|
| Codes / frames | 155 | 172 |
| Audio duration | 7.448 s | 8.238 s (+11%) |
| Sampler / step | 4.3 ms | 5.9 ms |
| RMS | 0.116 | 0.121 |
| Spectral centroid | 2023 Hz | 2040 Hz (ratio 0.992) |
| F0 mean (voiced) | 178.5 Hz | 195.2 Hz (+9%) |
| F0 std | 82.4 Hz | 80.5 Hz |
| **DTW-aligned 13-MFCC cosine** | mean **0.9935** / median **0.9955** |

The two outputs are different draws of the same content (different code
counts → AR divergence from the first frame), but after DTW time-warp the
phoneme content is **>99% similar**. Speaker timbre matches within 1%
(spectral centroid ratio). Prosody differs (+9% mean F0 for Swift) but F0
range / std is nearly identical, so it's the same speaker placing emphasis
slightly differently — not a different voice or wrong words.

This is the expected behavior of two non-equivalent samplers drawing from
the same conditional distribution: the fused graph runs fp16 on ANE with
top-k=80 cumsum / `cdf >= u`, Swift runs fp32 on CPU with full-vocab
cumsum / `u * totalF` / `cdf > u`. They were never bit-equivalent and
`MagpieFusedLocalSampler.swift:73-76` documents the difference. The
shipped lever is "produces clean speech", not "produces identical
samples."

A separate `MagpieFusedLocalSamplerParityTests` was written that probes
bit-level parity on synthetic Gaussian hiddens. Per-codebook agreement
~13–24% — flat synthetic logit distributions are pathologically sensitive
to fp16 ε perturbations, but the test does not invalidate Trial 3 because
real decoder hiddens produce peaky distributions where fp16 drift is
imperceptible (this audio diff is the proof).

Subjective listen on the same prompt: fused was preferred over Swift
(more natural cadence on this seed). N=1, so not a systematic finding —
would need a multi-prompt MOS-style evaluation to claim consistent
improvement. The relevant claim is that fused did **not** degrade
quality, which both the objective metrics and the listening test
confirm.

Commits: `4b57ab27e` (Swift), `f3ba62e` (mobius mlpackage), this PERF
update closes the quality validation loop.

#### End-to-end latency A/B (post-quality-validation)

After quality was confirmed, measured Trial 3 fused LT vs Swift LT
fallback head-to-head with the same text / seed / speaker. Toggle
mechanism: `mv local_transformer.mlmodelc /tmp/...` + `mkdir`
empty-dir placeholder so `MagpieResourceDownloader.ensureAssets`
skips re-download (fileExists=true) and `MagpieModelStore.loadModel`
gracefully returns nil → Swift sampler engaged.

**Test rig**: M2 MacBook Air, release build, `magpie text` CLI.
Text: "The quick brown fox jumps over the lazy dog. Pack my box with
five dozen liquor jugs." (~8s audio, ~170 codes), seed=42, speaker=0,
en, cfg=1.0 (no CFG → useFusedSampler=true when LT loaded).

**Methodology lesson**: cycle 1 ran ON-runs sequentially then
OFF-runs sequentially with no cooldown. OFF appeared **2.3× faster**
end-to-end (3.85s vs 8.81s). This was a thermal artifact — the ON
runs hit accumulated thermal throttling, the OFF runs benefited from
warm ANE caches without paying the throttle. Cycle 2 alternated
ON / OFF with 30s cooldowns and a 90s pre-cool. Always alternate when
benchmarking AR-loop CoreML on the same SoC, or you measure thermals
not models.

##### Cycle 2 — alternating, batch mode

| Run | Mode | Total | AR loop | decoder ms/step | sampler ms/step | nanocodec |
|---|---|---|---|---|---|---|
| A1 | ON  | 3.94s | 2.32s | 11.4 | **2.1** | 3.10s |
| B1 | OFF | 3.84s | 2.35s | 11.4 | 2.4 | 2.96s |
| A2 | ON  | 4.98s | 2.33s | 11.5 | **2.1** | 4.86s |
| B2 | OFF | 6.53s | 2.87s | 13.7 | 3.2 | 6.88s |
| A3 | ON  | 5.86s | 2.44s | 12.1 | **2.2** | 6.48s |
| B3 | OFF | 3.86s | 2.35s | 11.4 | 2.5 | 2.99s |

- Per-step sampler: fused **~2.1 ms**, Swift **~2.4 ms** → fused wins
  ~0.3 ms / step.
- Per-step decoder: identical at 11.3-11.5 ms regardless of LT path
  (the standalone profile previously reported 15.3 ms — the standalone
  number was inflated by being measured with both decoder_step and LT
  loaded).
- AR loop totals essentially equal (~2.32s ON vs ~2.35s OFF on healthy
  thermal).
- Total wall-clock dominated by `nanocodec_decoder_v3.mlmodelc`,
  which varied 3-7s purely on thermal state.

##### Cycle 2 — alternating, streaming mode (TTFA)

| Run | Mode | TTFA | Total | First chunk |
|---|---|---|---|---|
| S_A1 | ON  | **1.287s** | 5.30s | 22 codes (1.10s audio) |
| S_B1 | OFF | 3.130s | 8.92s | 29 codes (1.43s audio) |
| S_A2 | ON  | **1.306s** | 3.81s | 22 codes |
| S_B2 | OFF | 4.036s | 12.37s | 29 codes |
| S_A3 | ON  | **1.309s** | 3.72s | 22 codes |
| S_B3 | OFF | 1.463s | 4.35s | 29 codes |

- ON TTFA rock-stable at ~1.30s.
- OFF TTFA thermal-sensitive (1.46s steady-state, 3-4s during
  thermal accumulation).
- Steady-state TTFA win (S_A3 vs S_B3): **~155 ms**.
- Steady-state total win: **~630 ms**.
- First chunk size differs (22 vs 29 codes) because fused and Swift
  samplers use different RNGs (`MagpieMT19937` vs `MagpieSamplerRng`)
  with the same seed → slightly different EOS step. This complicates
  the apples-to-apples but doesn't invalidate the TTFA delta —
  per-frame TTFA also favors fused.

##### Honest verdict on Trial 3's magnitude

The original Trial 3 claim was "~4 ms/step → 1.78 ms/step LT call,
~2.2 ms / step saved." That was the **standalone CoreML predict()**
profile. The full Swift-side call path (fused vs Accelerate) is much
closer:

- Fused LT (CoreML, ANE): ~2.1 ms / step from inside the AR loop
- Swift LT (Accelerate vDSP, CPU): ~2.4 ms / step

Real per-step win: **~0.3 ms**, not ~2.2 ms. For 170 steps that's
~50 ms saved on the AR loop, plus a TTFA improvement that compounds
because the first chunk hits Trial 1's 24-frame cap. Total observed
wins: ~155 ms TTFA, ~630 ms total streaming wall-clock at steady
thermal state.

Trial 3 is still a net positive (no regression, slight win on every
step) and the quality is at-least-equivalent. It is **not** the
dramatic 2× speedup the standalone profile suggested. The dominant
end-to-end cost is `nanocodec_decoder_v3` which Trial 3 does not
touch and which exhibits ~3× thermal swings — that's where the next
optimization lever should aim, not at further LT fusion.

### Trial 4 — AR loop unroll into fixed-N graph ✗ DEAD-END

Trace `decoder_step` with N unrolled iterations + N LT samplers + (N-1)
audio-embed lookups inside one CoreML graph. Single ANE submission per N
steps; KV cache stays internal between unrolled steps.

**Hypothesis (pre-flight rationale)**:

The Trial 3 fused LT graph successfully unrolls **8 × 1-layer LT** at 73.9%
ANE. By contrast, decoder_step is a **12-layer transformer body**.
- N=2 unroll → 24 transformer-layer ops + 2 LT + 1 audio-embed → ~5× LT graph size
- N=8 unroll → 96 transformer-layer ops + 8 LT + 7 audio-embed → ~20× LT graph size

Boundary cost was estimated at ~4 ms per dispatch (200 ms Swift overhead / 48
dispatches in first chunk). Hypothetical unroll savings:

| N | Dispatches saved | Est. Δ TTFA | ANE compile risk |
|---|---|---|---|
| 1 (fuse decoder+LT only) | 24 | −50 to −75 ms | low |
| 2 | 36 | −75 to −100 ms | moderate |
| 4 | 42 | −100 to −130 ms | high |
| 8 | 45 | −130 to −160 ms | very high |

Pre-flight kill criteria: ship if N=2 stays > 90% ANE; abandon if < 80% ANE.

#### Trial 4a — N=2 pre-flight ✗ DEAD

**Implementation** (`traceable/traceable_decoder_step_n2.py`,
`experiments/convert_decoder_step_n2.py`):

- Reuse `TraceableDecoderStep` for both iterations (parameters shared, KV
  state passed through)
- Embed `_FusedLT` (mirrors `FusedLocalTransformer`): 8 unrolled 1-layer LT
  iterations + per-codebook top-k+CDF sampling
- Inter-iter audio-embed lookup: `audio_embed_2 = mean_cb (one-hot @
  audio_emb_full[cb])` — mask-multiply pattern, mirrors Swift `fillAudioEmbed`
  exactly
- 38 inputs (audio_embed + enc + mask + 36 KV state + 5 sampler params)
- 38 outputs (codes_1 + codes_2 + 36 new KV state)
- Conversion: 35.4 s, mlprogram fp16 iOS17, 215 MB weights

**Profile (M2, coreml-cli, all three models re-measured in same session)**:

| Model | `.all` | `.cpu_only` | `.cpu_and_gpu` | `.cpu_and_neural_engine` |
|---|---|---|---|---|
| decoder_step | 17.19 ms | 18.59 ms | 23.45 ms | **16.26 ms** (97.3% ANE) |
| local_transformer | 3.07 ms | 1.09 ms | 3.86 ms | **1.57 ms** (73.9% ANE) |
| fused_decoder_step_n2 | **64.77 ms** (88.7% ANE) | 26.23 ms | 28.64 ms | **ANEF compile fails** |

**Head-to-head, per policy** (N=2 fused vs 2 × sequential):

| Policy | 2 × sequential | N=2 fused | Δ |
|---|---|---|---|
| `.all` | 40.52 ms | 64.77 ms | **+24.25 ms (1.60×)** |
| `.cpu_only` | 39.36 ms | 26.23 ms | −13.13 ms (0.67×) |
| `.cpu_and_gpu` | 54.62 ms | 28.64 ms | −25.98 ms (0.52×) |
| `.cpu_and_neural_engine` (prod) | 35.66 ms | err | N=2 cannot run |

Two notable findings:

1. **N=2 fused IS faster than 2 sequential calls on CPU-only and CPU+GPU**
   (dispatch amortization works as predicted). It just can't pay off on ANE
   because the sampler tail forces partition thrash.

2. **N=2 still loses under `.all` and is unrunnable under production
   `.cpu_and_neural_engine`**. Swift's production code routes individual
   models to `.cpu_and_neural_engine` (16.26 + 1.57 = 17.83 ms per iter,
   35.66 ms × 2). The lever was always going to lose vs that — N=2 can't
   even compile under that policy.

**Root cause** (`coreml-cli --fallback`): **858 of 2315 ops fall back to CPU
(37.1%)** despite 88.7% aggregate residency. The doubled sampler tail dominates
the partition cuts:

- `cumsum × 32`, `topk × 16`, `equal × 38`, `greater_equal × 18`, `softmax × 30`
- `linear × 92`, `reduce_mean × 56`, `real_div × 42`, `sqrt × 28`, `gelu × 14`

Aggregate ANE residency reads as 88.7% because the CPU ops are individually
cheap, but the **partition boundary cost dominates wall-clock** — every
sampler block forces a round-trip through ANE→CPU→ANE. State I/O is also
huge (12 layers × 2 × 1×512×12×64 fp16 = ~19 MB cache in + ~19 MB cache out
per call, doubled for the 2 iterations).

The kill threshold of "<80% ANE" turned out to be the wrong metric — even at
88.7% residency the latency regressed 1.87×. The true gating signal would have
been per-op fallback rate or partition count.

**Verdict**: lever dead. N=4/N=8 cannot save this — they linearly scale the
sampler-tail penalty and state I/O. **Do not pursue**.

Files retained for reference (do not delete):
- `traceable/traceable_decoder_step_n2.py`
- `experiments/convert_decoder_step_n2.py`
- `build/fused_decoder_step_n2.mlpackage` (215 MB)
- `compiled/build/fused_decoder_step_n2.mlmodelc` (1.7 MB + weights)

Already-known dead-end (separate experiment): stateful variant
(`traceable_decoder_step_stateful.py`) forces CPU+GPU via MLState, regresses 2.2×.

**Lesson**: ANE residency percentage is a misleading proxy for an unroll
graph's quality — partition count + per-op fallback rate matter more.
Sampler tails (cumsum, topk, equal, softmax with comparisons) are CPU-only
and proliferate linearly with N.

#### Sub-trial 4b — N=1 fused (decoder_step + LT, no unroll)

After Trial 3 quality validation closed, profiled a smaller variant:
just `decoder_step + local_transformer` fused into one graph (12 layers
+ 1 LT, no audio_embed handoff, no AR unroll). Hypothesis: too big to
ANE-compile was the killer for N=2; N=1 should compile and save the
per-step `decoder_step → LT` dispatch boundary.

`mobius/models/tts/magpie/coreml/traceable/traceable_decoder_step_n1.py`,
`convert_decoder_step_n1.py`. Resulting `fused_decoder_step_n1.mlmodelc`
is 191 MB.

Same-session profile (M2, coreml-cli):

| Path | Policy | Predict | ANE % |
|---|---|---|---|
| `decoder_step.mlmodelc` | `.cpu_and_neural_engine` | 15.29 ms | 97.3% |
| `local_transformer.mlmodelc` | `.cpu_and_neural_engine` | 1.69 ms | 73.9% |
| **Sum (current shipped)** | — | **17.0 ms** | — |
| `fused_decoder_step_n1.mlmodelc` | `.cpu_and_neural_engine` | **20.45 ms** | 94.5% |
| `fused_decoder_step_n1.mlmodelc` | `.all` | 42.60 ms | 91.8% |

**Verdict: N=1 fusion is a 3.45 ms / step regression.** ANE residency
drops from 97.3% (clean decoder_step) to 94.5% (fused) because the LT
sampler tail's CPU ops now contaminate the same graph. The dispatch
boundary savings (which would have to exceed 3.45 ms to come out ahead)
are smaller than that — observed AR loop overhead in production is
< 2 ms / step beyond kernel time.

Useful negative datum: **fusion at any granularity at the
decoder_step + LT interface costs more in graph-compile pessimization
than it saves in dispatch boundary on M2 ANE.** The two models are
better off as separate dispatches. Closes Trial 4 across the unroll
spectrum (N=1 regresses, N=2 won't compile under prod policy).

### Trial 5 — `speakerContextLength` 110 → 64 ✗ DEAD-END

Pure-reconvert path tried. **Naive truncation breaks voice cloning fidelity.**

```bash
uv run python convert_decoder_prefill.py --t-ctx 64 \
  --output build/decoder_prefill_tctx64.mlpackage
xcrun coremlcompiler compile build/decoder_prefill_tctx64.mlpackage compiled/build/
uv run python truncate_speaker_embeddings.py \
  --constants-dir ~/.cache/fluidaudio/Models/magpie-tts/constants --t-ctx 64
```

`truncate_speaker_embeddings.py` takes **last 64 frames** of each
(5, 110, 768) speaker embedding. Per-speaker `speaker_{0..4}.npy` files
must also be truncated (Swift's `MagpieConstantsStore` validates each
shape against `(speaker_context_length, d_model)`).

Result on M2 release, "Hello from Magpie.":

| | T_ctx=110 (baseline) | T_ctx=64 |
|---|---|---|
| TTFA | 0.829 s | **2.520 s** (regression) |
| First chunk | 24 codes (1.19 s "Hello") | 83 codes (3.93 s "Hello") |
| EOS | ✓ at natural boundary | runs ~3.5× past natural EOS |

Model trained on T_ctx=110 — truncating destroys the prosodic state the
EOS classifier relies on. Real fix paths (none simple): re-extract from
source audio at T=64 via NeMo's speaker encoder, or train a learned
pooling. Cache restored from `~/.cache/fluidaudio/Models/magpie-tts/.backup-tctx110/`.

Artifacts kept: `truncate_speaker_embeddings.py`,
`build/decoder_prefill_tctx64.mlpackage`.

### Trial 6 — NanoCodec / AR pipelining in Swift (low-impact for TTFA)

NanoCodec runs CPU, AR runs ANE → free parallelism if scheduled to overlap.
Already implemented via producer/consumer split in `synthesizeStream(...)`.
Helps **aggregate RTFx**, not TTFA (first chunk has nothing to overlap with).

### Trial 7 — First-chunk cap 24 → 16 frames ✗ DEAD-END

A/B on M2 release, fused LT path, 5 runs each:

| Input | Cap | TTFA median | First chunk |
|---|---|---|---|
| "The quick brown fox..." | 24 | **1.585 s** | 22 codes ("The quick"), 1.10 s |
| "The quick brown fox..." | 16 | 1.636 s | 7 codes ("The"), 0.41 s |

1. **TTFA didn't drop** — with chunker tweak + LT fusion, the first chunk's
   AR loop is no longer dominant; 8 fewer steps don't move the needle.
2. **0.41 s first chunk is too short for streaming UX** — visible playback
   stutter as the player waits for chunk 1.

24 frames is the sweet spot. Recorded so we don't re-litigate.

### Trial 8 — QAT / calibration int8 via NeMo ⏸ DEFERRED

Real fix for the int8 runaway in Trial 2. Recovers per-step int8 win cleanly.
Multi-day project: NeMo install (~1-2 GB), calibration data, training-time
work. **Highest ceiling, highest cost.** Theoretical TTFA floor ~300 ms on M2.

### Trial 9 — Pipeline-level fusion via `MLPipeline` (low-EV)

Wire `text_encoder → decoder_prefill` into a single mlpackage. Saves one
Swift dispatch per synth (~1-2 ms). Doesn't help the AR loop. Not worth it.

### Trial 10 — NanoCodec investigation ✗ DEAD-END (all paths)

NanoCodec at 182 ms / 22% of TTFA looked like a target after Trial 4
profiling identified it as the 2nd-largest single term. Three paths
investigated; all closed.

#### Trial 10a — NanoCodec v4 (palette-quantized fp32) ✗ DEAD

`coreml-cli` profile, M2 release, `compiled/build/v4/nanocodec_decoder_v4.mlmodelc`:

| Variant | `.all` | `.cpuOnly` | `.cpuAndGPU` | `.cpuAndNeuralEngine` | ANE % | Audibility |
|---|---|---|---|---|---|---|
| v2 (fp16) | 86 ms | 60.8 ms | 133 ms | **38.96 ms** | 43.4% | **noisy on voiced speech** |
| v3 (fp32, current) | 337 ms | **182 ms** | 318 ms | (CPU forced) | 0% | clean ✓ |
| v3_int8pal | 305 ms | 233 ms | 316 ms | 170 ms | 0% | (untested) |
| v4 (palette) | **307 ms** | 518 ms | 322 ms | 447 ms | 0% | (untested) |

v4 is **slower than v3 on every compute-unit policy** on M2. Palette
quantization shrinks the model (~31 MB vs ~120 MB) but adds dequant
overhead per CPU op without unlocking ANE (still fp32 compute precision
end-to-end). v3_int8pal is the only marginal speedup (~12 ms vs v3 on
`.cpuAndNeuralEngine`) but likely re-introduces the same fp16 noise
budget Phase F.2 already proved is audibly broken — would need an A/B
listening test to confirm.

#### Trial 10b — Split first NanoCodec call ✗ DEAD

Hypothesis: emit first audio after 8-12 codec frames instead of 24,
overlapping later NanoCodec calls with chunk-2 AR work. Reading
`MagpieNanocodec.swift:23` and `nanocodec_experiments/results/STATUS.md`
Phase C v2 chunked-parity:

```
T_in=8  → SNR -1.38 dB  (FAILS)
T_in=16 → SNR  5.40 dB  (FAILS)
T_in=24 → SNR 11.46 dB  ≈ Taylor5 floor (CLEAN, current)
```

The model's input window is fixed at T=24; you can't shrink it without
re-training. Lowering `streamingFirstChunkCap` below 24 codec frames
would still trigger one NanoCodec call (~182 ms), so wins are limited
to AR loop savings (~157-235 ms TTFA). But:

1. **First NanoCodec call would right-edge-replicate** for caps below 24.
   Phase C v2 step 7 only validated **left-edge** replication for
   utterance start. Right-edge is an audio-quality regression risk.
2. **Steady-state UX worse** — at cap=8 the first chunk produces only
   ~372 ms of audio, while chunk 2's AR loop (cap=24's worth) takes
   ~470 ms. Audible gap right after the first audio pop.

Net: theoretical −235 ms TTFA at the cost of (a) untested audio
quality, (b) structurally worse playback continuity. Not worth shipping.

#### Trial 10c — NanoCodec ANE / mixed-precision ✗ DEAD (Phase F)

Already exhausted in `nanocodec_experiments/results/STATUS.md` Phase F:

| Approach | Result |
|---|---|
| Whole-graph fp32 | clean (current production) |
| op_type filter (convs only fp32) | audibly noisy at 48 dB SNR |
| op_type filter (activations / Snake only fp32) | identical to fp16 |
| scope filter (per-stage / per-location) | not supported by coremltools op_selector |

The only remaining path to clean ANE-resident NanoCodec is **model-level
retraining or QAT** — out of scope. Production stays on v3 fp32 CPU.

#### Trial 10d — Parallel sliding-window decode ✗ DEAD

**Hypothesis.** `MagpieNanocodec.decode()` runs ~20 sequential
24-frame window calls per ~8 s utterance, all on a single CPU core
(`compute_units = .cpuOnly`). The window iterations are data-independent
(each reads its own input slice, writes its own output slice). If the
CoreML CPU backend is single-core per call, dispatching N independent
`predict()` calls concurrently from a thread pool should scale roughly
N× on a 4 P-core M2.

**Bench.** `nanocodec_experiments/parallel_window_bench.py` —
loads `nanocodec_decoder_v3.mlpackage` with `ComputeUnit.CPU_ONLY`,
fabricates 20 distinct (8 codebook × 24 frame) int32 inputs, dispatches
them sequentially or via `ThreadPoolExecutor`. 4 warmup calls, 15 s
cooldown between trials, 2 cycles per `workers` count.

```
 workers cycle  total ms  per-call min  per-call med  per-call max
       1     1      5849           147           240           773
       1     2      2346           116           116           135   ← warm floor
       2     1      2406           116           236           266
       2     2      3612           143           303           467
       4     1      2367           117           249          1049
       4     2      3138           134           509           821
```

**Result.** Adding workers does *not* reduce wall-clock. The fastest
configuration of all six trials is **1 worker, cycle 2** (2346 ms total,
116 ms median per call) — exactly the warm sequential baseline. 2-way
and 4-way parallel land at 2406 / 2367 ms (within 1-3 % of the warm
floor — i.e. noise). All cycle 2 results past the first thermal cycle
degrade further (3138-3612 ms) regardless of worker count, indicating
heat is the dominant variable, not concurrency.

**Why it fails.** Either (a) Apple's CoreML CPU backend already
multi-threads internally inside a single `predict()` call and saturates
available cores, or (b) there is a global serialization lock around
the prediction path. Either way the per-call latency floor (~117 ms
warm) is what we get; it does not divide across cores.

This is consistent with Trial 10b's earlier finding that splitting
the model into smaller pieces does not unlock parallelism — the
per-call cost is set by the model graph, not by Swift-side dispatch.

**Verdict.** Sliding-window parallelism is dead. The 20 × ~117 ms
NanoCodec budget (~2.3 s of an 8 s utterance) is a hard floor on
M2 fp32 CPU. Cannot be improved by Swift-level concurrency.

**Trial 10 closed.** All graph-level and dispatch-level levers on
NanoCodec are exhausted (10a fp32 palette, 10b split, 10c ANE/mixed,
10d parallel). The only remaining path is model-level retraining
(QAT int8 via NeMo) — out of scope here.

---

## Lever ranking (post-Trial 10 closure)

Updated priority order after Trial 4 and Trial 10d closed (all
graph-level *and* dispatch-level NanoCodec levers exhausted):

| Rank | Lever | Est. TTFA win | Effort | Risk |
|---|---|---|---|---|
| 1 | v3_int8pal NanoCodec A/B listening test | 12 ms (if clean) | 30 min | quality risk |
| 2 | Cache-aware encoder warmup | one-shot first-call | 2-3 hr | low |
| 3 | #8 QAT int8 via NeMo | ~400 ms | multi-day | very high |
| 4 | Architectural: route to Kokoro/PocketTTS where multilingual not needed | ~700 ms | 1-2 hr | none |
| ✗ | ~~Parallel sliding-window decode~~ | ~~est. 3-4× nanocodec~~ — Trial 10d: 0 % wall-clock win on M2 | — | — |

---

## Estimated TTFA ceilings

| Stack | TTFA (warm M2) |
|---|---|
| Pre-tuning baseline | ~2 525 ms |
| + chunker tweak (#1) | ~1 547 ms |
| + LT fusion (#3) | **~830 ms** ← current (all graph levers exhausted) |
| ~~+ AR unroll N=2~~ | ~~est. 735 ms~~ — actual: regresses; lever dead |
| + QAT int8 (#8) | ~300 ms (theoretical floor — only remaining lever) |

**NanoCodec floor confirmed at 182 ms** (22% of TTFA). Cannot be reduced
without re-training; full fp32 is required for clean audio output.

For comparison: **Kokoro TTFA < 100 ms** (parallel non-AR, ~20× RTFx). Any
amount of optimization on a 357M autoregressive transformer with 21.5 fps
emission rate will not approach a parallel TTS at M2 ANE throughput. The
right architectural answer is to **route latency-sensitive callers to Kokoro
or PocketTTS** and use Magpie when its multilingual / built-in-speaker
features are specifically required.

---

## Alternative open-weight Magpie checkpoints

| Variant | Source | Architecture | Speed |
|---|---|---|---|
| `nvidia/magpie_tts_multilingual_357m` v2512 (Jan 2026) | HF, **currently shipped** | 357M | baseline |
| `nvidia/magpie_tts_multilingual_357m` v2602 (Mar 2026) | HF | 357M (+Hindi, +Japanese) | **same** |
| `magpie-tts-flow` | NIM API only | different arch | n/a (cloud) |
| `magpie-tts-zeroshot` | NIM API only, voice cloning removed from open release | n/a | n/a (cloud) |
| `magpie-tts-multilingual` NIM | NIM API only, "optimized batch and latency pipeline" | n/a | n/a (cloud) |

**No faster open-weight Magpie exists.** v2602 is a content upgrade
(2 more languages) at the same architecture/size/speed. Speed-class
upgrades are NIM-only.

---

## Repro

### Profile any model

```bash
cd mobius/tools/coreml-cli
uv run coreml-cli ~/.cache/fluidaudio/Models/magpie-tts/decoder_step.mlmodelc
uv run coreml-cli ~/.cache/fluidaudio/Models/magpie-tts/decoder_step.mlmodelc --fallback
```

### End-to-end TTFA measurement

```bash
cd FluidAudio
swift build -c release
.build/release/fluidaudio tts \
  --text "Hello from Magpie." --engine magpie --stream \
  --speaker 0 --output /tmp/out.wav --seed 42
```

### Reconvert decoder_step (int8)

```bash
cd mobius/models/tts/magpie/coreml

# Bring fp16 decoder_step.mlpackage from HF (avoid NeMo install)
huggingface-cli download FluidInference/magpie-tts-multilingual-357m-coreml \
  decoder_step.mlpackage --local-dir build/upstream/

# Quantize (per-channel, skip LM head — best int8 we found)
uv run python experiments/quantize_decoder_step_int8.py \
  --input build/upstream/decoder_step.mlpackage \
  --output build/decoder_step_int8_pc_skiphead.mlpackage \
  --granularity per_channel \
  --skip-ops linear_60_cast_fp16

uv run python compile_mlmodelc.py
```

### Reconvert decoder_prefill (T_ctx=N)

```bash
uv run python convert_decoder_prefill.py --t-ctx 64 \
  --output build/decoder_prefill_tctx64.mlpackage
xcrun coremlcompiler compile build/decoder_prefill_tctx64.mlpackage compiled/build/
uv run python truncate_speaker_embeddings.py \
  --constants-dir ~/.cache/fluidaudio/Models/magpie-tts/constants --t-ctx 64
```

### Convert fused Local Transformer

```bash
uv run python convert_local_transformer.py \
  --output build/local_transformer.mlpackage
xcrun coremlcompiler compile build/local_transformer.mlpackage compiled/build/
# See UPLOAD_LOCAL_TRANSFORMER.md for HF upload steps.
```

### Hot-swap a candidate model into the cache

```bash
# Backup first
cp -R ~/.cache/fluidaudio/Models/magpie-tts/decoder_step.mlmodelc \
      ~/.cache/fluidaudio/Models/magpie-tts/.backup-decoder_step.mlmodelc

# Swap candidate in
cp -R compiled/build/decoder_step_candidate.mlmodelc \
      ~/.cache/fluidaudio/Models/magpie-tts/decoder_step.mlmodelc

# Re-run end-to-end TTFA, then restore from backup if regressed.
```
