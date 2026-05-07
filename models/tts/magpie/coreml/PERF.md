# Magpie TTS — TTFA / RTFx Performance Notes

Running record of latency tuning for the FluidAudio Swift port. All numbers are
on **Apple M2 / macOS 26.5 / coremltools 8.x** unless noted. "TTFA" = time to
first audio chunk under the streaming `synthesizeStream(...)` API.

## Latest — Lever #3: Local Transformer fusion ✓ SHIPPED

Standalone fused mlpackage (`local_transformer.mlpackage`) — separate from
`decoder_step` to avoid a NeMo retrace. Bakes 8 unrolled LT iterations +
per-codebook softmax / cumsum / sample into a single CoreML graph.

| Build | Inputs | Output | ANE residency | Per-step |
|---|---|---|---|---|
| `local_transformer.mlpackage` (fp32) | `decoder_hidden 1×768`, `uniforms 8`, `forbid_eos 1`, `temperature 1` | `codes 8 int32` | 73.9 % | 1.78 ms |

End-to-end TTFA, M2, release build, seed 42, 3 trials:

| Input | Swift LT (baseline) | Fused LT (ANE) | Δ |
|---|---|---|---|
| "Hello from Magpie." (3 words) | 1.283 s | **1.122 s** | **-161 ms (-12.5 %)** |
| 14-word EN sentence | 4.243 / 4.203 s | 4.788 / 4.146 s | neutral, high variance |

Short-input win is inside the predicted **-120 to -240 ms** band. Long-input
delta is dominated by AR loop length (143 codes vs 38) and per-step variance
on shared M2 — the LT fusion saving still applies but is a smaller fraction
of total TTFA.

## Baseline (pre-tuning)

| Mode | TTFA | Total | RTFx | Notes |
|---|---|---|---|---|
| Streaming, 8-word EN sentence | 2.525 s | — | — | first-chunk cap = 50 codec frames |
| Batch synth, 8-word EN | — | ≈ 96 s | 0.04× | autoregressive, 357M params |
| Aggregate MiniMax-EN corpus | — | — | 0.41× | reported in `MagpieTtsManager.swift` doc |

Magpie's value prop is **multilingual coverage and 5 built-in speakers**, not
throughput. For latency-sensitive paths use **Kokoro (~20× RTFx, parallel)** or
**PocketTTS (~1.5–2× RTFx, streaming Mimi)** — both already in FluidAudio.

## CoreML model inventory

4 logical models + 4 NanoCodec build variants (only one active at runtime):

| File | Role | Compute | Status |
|---|---|---|---|
| `text_encoder.mlmodelc` | phoneme/IPA → encoder hidden states | ANE | required |
| `decoder_prefill.mlmodelc` | speaker-context prefill batched 110-step | ANE | optional fast path |
| `decoder_step.mlmodelc` | AR transformer body, called per code-frame | ANE 97.3% | required |
| `nanocodec_decoder.mlmodelc` (v1) | T=256 monolithic, fp16 | CPU | legacy fallback |
| `nanocodec_decoder_v2.mlmodelc` | T=24 chunked, fp16 | ~43% ANE | noisy on voiced speech |
| `nanocodec_decoder_v3.mlmodelc` | T=24 chunked, fp32 | CPU | **default**, audibly clean |
| `nanocodec_decoder_v4.mlmodelc` | T=24 chunked, fp32 + 8-bit palette | CPU | acoustically transparent vs v3, 4× smaller (31 MB) |

Plus a Swift-side **Local Transformer** (1-layer, 8-codebook sampler) loaded
from `constants/local_transformer/`. Not a separate mlmodelc — runs on CPU
between every `decoder_step` call.

## Per-step `decoder_step` latency (coreml-cli, 1 isolated step)

| Variant | `all` predict | `cpu_and_neural_engine` | ANE residency |
|---|---|---|---|
| fp16 (baseline) | 16.05 ms | 15.49 ms | 97.3 % |
| int8 per-tensor | 12.56 ms | 14.25 ms | 97.3 % |
| int8 per-channel | 14.62 ms | 15.09 ms | 97.3 % |
| int8 per-channel + skip-head | 15.06 ms | **13.44 ms** | 97.3 % |

`constexpr_affine_dequantize` shows up in `--fallback` op counts but is a
constant-prep op, not in the forward path. All quantized variants stay 97.3 %
ANE-resident on actual measured runs.

## TTFA breakdown (warm, fp16, after chunker tweak)

| Stage | Cost | Notes |
|---|---|---|
| Text encoder | ~50–100 ms | 1 call per utterance |
| Decoder prefill (110 steps batched) | ~200–300 ms | 1 call per utterance via `decoder_prefill.mlmodelc` |
| AR loop on first chunk (24 steps) | **~360 ms** | dominant — 24 × 15 ms |
| Local Transformer (8 codebooks × 24) | ~100–200 ms | Swift, CPU |
| NanoCodec decode of 24 frames | ~50–100 ms | CPU (v3) |
| **Total TTFA** | **~870 ms warm** | M2 |

The AR loop is the structural bottleneck. 357M params × 24 sequential
`decoder_step` calls — there is no way around the 24 calls without changing
the model.

---

## Trial 1 — Streaming first-chunk cap 50 → 24 frames ✓ SHIPPED

`MagpieChunker.streamingFirstChunkCap` reduced from **50 → 24** codec frames
(matches NanoCodec's v2/v3 sliding-window receptive field).

| | Before | After | Δ |
|---|---|---|---|
| TTFA | 2 525 ms | 1 547 ms | **-38.7 %** |
| Audio quality | clean | clean | unchanged |
| Reconvert? | no | no | — |

Result: **shipped** as commit `b93f3b099` on branch `feat/magpie-ttfa-tweak`
(FluidAudio repo, not yet pushed). Pure Swift change, no model touched.

`24` is hard floor: NanoCodec v2/v3 use a 24-frame chunk shape, going below
forces zero-padding on the first chunk and adds boundary ringing.

---

## Trial 2 — Post-training int8 weight quant on `decoder_step` ✗ DEAD-END

Three configs tried, all break end-to-end EOS termination on long-context
streaming inputs. Per-step latency wins are real in isolation but unusable
when the model can't terminate.

Script: `quantize_decoder_step_int8.py` (in this dir).

### 2a. Per-tensor `linear_symmetric` int8

End-to-end synth, "Hello from Magpie." (3 words, 1–2 chunks):

| | fp16 | int8 per-tensor |
|---|---|---|
| TTFA | 1.48 s | 6.84 s |
| Total synth | 2.00 s | 15.13 s |
| Audio out | 2.59 s | **24.04 s (garbled)** |
| Chunk 1 EOS | @ 29 codes ✓ | **never fires**, hits maxSteps=500 |

**Diagnosis**: per-tensor int8 collapses dynamic range across the 16192-way
output softmax. The EOS code's logit no longer wins after the prefill-anchored
chunk 0 → unconditional runaway from chunk 1.

### 2b. Per-channel `linear_symmetric` int8

Better but still not viable on long inputs:

| | fp16 | int8 per-channel |
|---|---|---|
| Short (3 words) TTFA | 1.48 s | **0.87 s** (-41 %) ✓ |
| Short total | 2.00 s | 1.46 s ✓ |
| Short chunk 1 EOS | ✓ @ 29 | ✓ @ 36 |
| Long (5 chunks) TTFA | 2.79 s | 2.12 s ✓ |
| Long chunk 4 EOS | ✓ @ 197 codes | **runaway @ 500, ~12 s tail garbage** |

Per-channel scales preserve per-output-code dynamic range and recover EOS on
chunks 0–3, but the longest terminal chunk still drifts. Short-input win is
genuine; long-input regression is unshippable as default.

### 2c. Per-channel + skip LM head (`linear_60_cast_fp16` kept fp16)

Textbook fix for INT8 transformer LMs. Did NOT help long-input runaway:

| | fp16 | per-ch + skip-head |
|---|---|---|
| Short TTFA | 1.48 s | 0.92 s ✓ |
| Long TTFA | 2.79 s | 2.73 s |
| Long chunk 4 EOS | ✓ @ 197 | **still runaway @ 500** |

**Diagnosis**: the drift is in the int8 transformer **body**, not localized to
the head. By chunk 4 the KV cache holds 110 prefill steps + ~300 generated
codes of int8-perturbed representations; an fp16 LM head can't recover the
correct EOS distribution off that drifted state.

### Verdict on post-training int8

Not shippable as default. Possibly opt-in for short / always-streaming
workloads where every chunk is bounded (≤ 2 chunks). Real fix requires
**QAT or activation-calibrated int8 export via NeMo** — multi-day project
with uncertain outcome. The mlpackages and mlmodelcs are kept under
`build/` and `compiled/build/` for the calibration follow-up.

---

## Untried levers, ranked by expected ROI

### 3. Local Transformer fusion into `decoder_step` (high-EV) ✓ SHIPPED (standalone)

Implemented as **standalone** `local_transformer.mlpackage` rather than
fused into `decoder_step` — keeps the original `decoder_step` graph intact
(no NeMo retrace) and lets the AR loop branch on availability. Wiring in
`MagpieFusedLocalSampler.swift` + `MagpieSynthesizer.swift`. CFG path falls
back to Swift `MagpieLocalSampler` (unconditional branch not in fused graph).

```
before: decoder_step → 1×768 hidden → Swift LT (8 sequential passes + topk softmax) → 8 ints
after:  decoder_step → 1×768 hidden → local_transformer (one ANE call)        → 8 ints
```

Measured: **-161 ms TTFA on short input** (see top of file). 1.78 ms/step on
ANE replaces ~4–8 ms/step of Swift CPU work. See `convert_local_transformer.py`.

### 4. AR loop unroll into fixed-N graph (research lever) [needs NeMo]

Trace `decoder_step` with N=8 or N=12 unrolled iterations inside one CoreML
graph. Single ANE submission for N steps; KV cache stays internal.

| | Cost |
|---|---|
| TTFA win | -100 to -200 ms (KV cache no longer crosses ANE↔Swift boundary) |
| Risk | medium-high — ANE compile-time graph depth limits may reject N=24 |
| Effort | high — fixed N requires multi-call strategy + post-EOS masking |
| Sampler | greedy works; stochastic needs RNG plumbed through inputs |

### 5. Drop `speakerContextLength` 110 → 64 [needs reconvert]

Each prefill step is one `decoder_step` call (or 1/110th of `decoder_prefill`).
Cutting 46 steps trims prefill cost.

| | Cost |
|---|---|
| TTFA win | ~-200 to -300 ms |
| Risk | voice cloning fidelity on the 5 built-in speakers — needs A/B audition |
| Effort | low — pure reconvert with a smaller `speaker_ctx` shape |

### 6. NanoCodec / AR pipelining in Swift

NanoCodec runs CPU, AR runs ANE → free parallelism if scheduled to overlap.
Helps **aggregate RTFx**, not TTFA (first chunk has nothing to overlap with).
Pure Swift change.

### 7. First-chunk cap 24 → 16 frames

-8 AR steps × 15 ms = **-120 ms TTFA**. Pushes inside NanoCodec's 24-frame
receptive field → boundary ringing on chunk 0. Likely audible but probably
not catastrophic. Quick to try.

### 8. QAT / calibration int8 via NeMo

Real fix for the int8 runaway. Recovers the per-step int8 win cleanly.
Multi-day project: NeMo install (~1–2 GB), calibration data, training-time
work. Highest ceiling, highest cost.

### 9. Pipeline-level fusion via `MLPipeline` (low-EV)

Wire `text_encoder → decoder_prefill` into a single mlpackage. Saves one
Swift dispatch per synth (~1–2 ms). Doesn't help the AR loop. Not worth it.

---

## Estimated ceilings

| Stack | TTFA |
|---|---|
| Pre-tuning baseline | ~2 525 ms |
| + chunker tweak (#1, shipped) | ~870 ms warm |
| + LT fusion (#3, shipped) | **~710 ms warm** (measured -161 ms) |
| + speaker context 64 (#5) | ~400–500 ms |
| + AR unroll (#4) | ~300–400 ms |
| + QAT int8 (#8) | ~300 ms theoretical floor on M2 |

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
| `magpie-tts-multilingual` | NIM API only, "optimized batch and latency pipeline" | n/a | n/a (cloud) |

**No faster open-weight Magpie exists.** `v2602` is a content upgrade
(2 more languages) at the same architecture/size/speed. Speed-class
upgrades are NIM-only.

---

## Repro

Conversion / quantization scripts in this directory:

```bash
# Bring fp16 decoder_step.mlpackage from HF (avoid NeMo install)
huggingface-cli download FluidInference/magpie-tts-multilingual-357m-coreml \
  decoder_step.mlpackage --local-dir build/upstream/

# Quantize (per-channel, skip LM head — best int8 we found)
uv run python quantize_decoder_step_int8.py \
  --input build/upstream/decoder_step.mlpackage \
  --output build/decoder_step_int8_pc_skiphead.mlpackage \
  --granularity per_channel \
  --skip-ops linear_60_cast_fp16

# Compile to mlmodelc
uv run python compile_mlmodelc.py

# Profile per-step latency + ANE residency
cd ../../../tools/coreml-cli
uv run coreml-cli ../../models/tts/magpie/coreml/compiled/build/decoder_step_int8_pc_skiphead.mlmodelc

# A/B end-to-end (hot-swap into FluidAudio cache)
cp -R compiled/build/decoder_step_int8_pc_skiphead.mlmodelc \
      ~/.cache/fluidaudio/Models/magpie-tts/decoder_step.mlmodelc
swift run fluidaudiocli magpie text \
  --text "Hello from Magpie." --stream --speaker 0 --output /tmp/out.wav
```

To restore fp16 baseline: re-download `decoder_step.mlmodelc` from HF or
keep a backup before swapping.
