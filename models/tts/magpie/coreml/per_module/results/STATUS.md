# Magpie ANE overhaul — status

Hardware: Apple M2, 16 GB, macOS 26.5. fp16, iOS17 deployment target.

Cross-phase status of the Magpie TTS ANE residency overhaul. The detailed
findings + tables live in `PHASE_A.md`; this is the single-page index.

## Phase 0 — Baseline (done)

Production CoreML pipeline benchmarked via `fluidaudiocli magpie bench`.

| Component | Static ANE | Runtime | Notes |
|---|---|---|---|
| `text_encoder` | 94–99 % | clean | — |
| `decoder_prefill` | 94–99 % | clean | — |
| `decoder_step` | **99 %** | (was pinned `.cpuAndGPU`) | runtime fixed in Phase D — see below |
| `nanocodec_decoder` | **0 %** (1149 ops) | all CPU | `ANECCompile() FAILED` whole-graph |

Numbers: 47.4 ms/decoder step, 0.44× RTFx end-to-end, 23.49 s nanocodec
per sentence.

### Final production state (post-Phase F)

| Component | Compute | Median predict | ANE % | Source build |
|---|---|---|---|---|
| `text_encoder` | ANE | 12.32 ms | 98.1 | fp16 |
| `decoder_prefill` | ANE | 17.55 ms | 93.9 | fp16 |
| `decoder_step` | ANE | 17.0 ms | 97.3 | fp16 (Phase D pinned `.cpuAndNeuralEngine`) |
| `nanocodec_decoder_t24_v2` (default) | **CPU only** | 142.51 ms / 24-frame call | 0 | **fp32** (Phase F: fp16 audibly noisy) |
| `nanocodec_decoder_t24` (opt-in) | CPU+ANE | 38.4 ms / 24-frame call | ~43 | fp16 (audibly noisy on voiced speech) |

End-to-end RTFx ~1.3× median on the fp32 default; AR loop terminates on
EOS deterministically.

### Nanocodec t24 build naming

Two t24 builds ship side-by-side; both share the same I/O contract
(24-frame input / 24576 audio samples per call) so `MagpieNanocodec`
auto-dispatches by shape.

| File on disk | Precision | Selector |
|---|---|---|
| `nanocodec_decoder_t24_v2.mlmodelc` | fp32 | `MagpieNanocodecPrecision.fp32` (default) |
| `nanocodec_decoder_t24.mlmodelc` | fp16 | `MagpieNanocodecPrecision.fp16` (opt-in) |

`MagpieModelStore.init(..., nanocodecPrecision:)` selects which to load;
if the requested precision is missing, it falls back to the other t24
build with a warning, then to the monolithic CPU-only `nanocodec_decoder.mlmodelc`.

The v2 suffix records the post-Phase F switch from fp16-default to
fp32-default. Older docs and trial scripts may still reference the
intermediate `nanocodec_decoder_t24_fp32.mlpackage` build name; the
on-disk artifact has been renamed to `_v2`.

## Phase A — Per-module ANE diagnostics (done)

`per_module/analyze.py` converts ~14 isolated nn.Modules and reports static
ANE residency.

Confirmed:
- **Snake (sin) is op-level rejected by ANE** — `Conv1d → Snake → Conv1d`
  lands 0 % ANE; identical block with polynomial Snake lands 100 %.
- **weight_norm is NOT a blocker** — both unfolded and folded land 100 %
  identical 2-op graphs.
- **Rank-4 one-hot KV cache write is NOT a blocker** — full causal
  self-attn with KV write at decoder shapes lands 100 % ANE statically.

→ Production `decoder_step` runtime fallback is therefore not op-level. It
   is a runtime recompile churn. Investigated separately in Phase D.

Full table + raw data: `PHASE_A.md`, `ledger.json`, `raw/*.json`.

## Phase B — Snake replacement in convert_nanocodec.py (done)

`convert_nanocodec.py` updated: `_snake_plain` now uses a clamped 5th-order
Taylor expansion of `sin²(α·x)` (`SnakeTaylor5Clipped`):

```python
ax = clamp(α · x, -π/2, π/2)
sin² ≈ ax² - ax⁴/3 + 2·ax⁶/45
return x + sin² / α
```

This was the minimum needed to remove `ios17.sin` from the converted graph.
Audio parity is **insufficient** at this approximation: **11.56 dB SNR**
vs the reference sin² Snake. Replacement candidates for Phase C+ v2:
LUT-via-conv (preferred), Padé approximant, or `mod π` + Taylor5.

## Phase C — Re-convert + analyze (done; result: still 0 % ANE)

Patched nanocodec (1821 ops, no sin / no pow) re-profiled with
`coreml-cli --fallback`:

| Build | Total ops | ANE % | Failure |
|---|---|---|---|
| baseline (sin²) | 1149 | 0.0 | `ANECCompile() FAILED` |
| Taylor5Clipped patch | 1821 | 0.0 | `ANECCompile() FAILED` |

Snake fix was necessary but not sufficient. Whole-graph compile failed.

## Phase C+ — Subgraph probe / threshold finding (done; root cause)

`per_module/nano_subgraph_probe.py` builds synthetic HiFi-GAN-style
decoders progressively from a single ResBlock up through the full 5-stage
decoder, holding topology constant while varying the input time dim.

| Spec | T_in | T_out | total ops | ANE % | failure mode |
|---|---|---|---|---|---|
| `body_5stage` | 8 | 4096 | 1006 | 99.0 | 10 dilated convs CPU |
| `body_5stage_T16` | 16 | 8192 | 1006 | **98.8** | 10 dilated + 2 ops `W=16386 ∉ [1, 16384]` |
| `body_5stage_T20` | 20 | 10240 | 1006 | **0.0** | **ANECCompile() FAILED** |
| `body_5stage_T24` | 24 | 12288 | 1006 | 0.0 | failed |
| `body_5stage_T32` | 32 | 16384 | 1006 | 0.0 | failed |
| `body_5stage_T64` | 64 | 32768 | 1006 | 0.0 | failed |
| `body_5stage_T128` | 128 | 65536 | 1006 | 0.0 | failed |
| `full_decoder_T8` | 8 | 2048 | 1019 | **99.0** | 10 dilated convs CPU |
| `full_decoder` (T=256) | 256 | 262144 | 1019 | 0.0 | ANECCompile() FAILED |

### Root cause

ANE compiler imposes a hard **W ≤ 16384** dimension limit on the
space-to-batch lowering of dilated convs (HiFi-GAN dilations 1/3/5). At
T_out=8192 the W-after-lowering is 16386, just over the limit, and the
two affected ops fall back individually. At T_out ≥ 10240 the ANE
compiler refuses to plan the entire graph and the whole codec falls to
CPU under the catch-all `ANECCompile() FAILED`.

Topology, Snake replacement, weight_norm folding, kernel-7 pre_conv,
post-act, post-conv, and tanh are **NOT** the trigger. Activation tensor
size is.

### Fix path (Phase C v2 — done; see section below)

## Phase C+ — Audio parity (done)

`per_module/audio_parity.py` and `snake_parity.py`. Random codec tokens,
seed=42, T=256:

| Metric | Value |
|---|---|
| samples | 262 144 |
| max_abs error | 3.86e-1 |
| mean_abs error | 7.06e-3 |
| RMS ref | 5.73e-2 |
| RMS err | 1.51e-2 |
| **SNR** | **11.56 dB** |

Insufficient. The clamp at α·x = ±π/2 plus a 5th-order Taylor diverges
from sin² across the codec's full operating range (codec α can train up
to ~5; codec activations up to ~3 → α·x ≳ 10, well beyond the clamp).

Replacement plan (deferred until chunked nanocodec lands ANE):
1. **LUT-backed sin via Conv1d** (preferred) — encode sin(α·x) as a 1-D
   lookup table evaluated by a depthwise conv against a sentinel basis.
   Numerically exact to LUT resolution. Adds ~96 small convs.
2. Padé approximant of sin² in [-π/2, π/2].
3. Range reduction with `sin²(y + π) = sin²(y)` then Taylor5 over a
   smaller domain (ANE compatibility of `mod` is unverified).

## Phase D — decoder_step ANE pin (done)

Hypothesis going in: `decoder_step.mlmodelc` was pinned to `.cpuAndGPU`
in `MagpieModelStore.swift` with a comment claiming
`MILCompilerForANE error: ANECCompile() FAILED` from rank-4 split-K/V
scatter. Phase A had already shown rank-4 KV write lands 100 % ANE in
isolation — contradicting the comment. Phase D verifies the production
model.

### Static profile (M2, macOS 26.5)

`coreml-cli ~/.cache/fluidaudio/Models/magpie-tts/decoder_step.mlmodelc`:

| Compute Unit | ANE % | Median predict (ms) |
|---|---|---|
| `.cpuOnly` | 0 % | 19.6 |
| `.cpuAndGPU` (then-current pin) | 0 % | 22.5 |
| `.all` | 97.3 % | 17.3 |
| `.cpuAndNeuralEngine` | **97.3 %** | **15.2** |

Cold compile 58 ms — no `ANECCompile() FAILED`. Eight CPU-fallback ops:
4 int32 dtype rejections (`cast`, `add`, `select`), 2 schedulable casts,
1 `greater_equal` (unresolved input), 1 `gather` (unsupported index
type). Total estimated CPU runtime ~0.14 ms.

The stale comment no longer holds — the model compiles cleanly on ANE.

### End-to-end bench (M2, "Hello world. … Apple Silicon with the Swift port.", seed=42, John, en)

| Pin | decoder_step ms/step | synth wall | RTFx | Codes | Audio | EOS |
|---|---|---|---|---|---|---|
| `.cpuAndGPU` | 36.1 ms | 30.98 s | 0.62× | 651 | 19.23 s | **`false`** (no EOS, hit step cap) |
| `.cpuAndNeuralEngine` | **23.9 ms** | **9.25 s** | **1.21×** | 234 | 11.20 s | **`true`** (clean) |

Wall-clock is **~2× faster on ANE** *and* the AR loop terminates on EOS
instead of running out the maxStep budget with tail garbage. Reproducible
deterministically across 5 runs (codes=234 every run on ANE).

### Fix

`Sources/FluidAudio/TTS/Magpie/Assets/MagpieModelStore.swift`: replaced
the `gpuConfig` (`.cpuAndGPU`) with `aneConfig` (`.cpuAndNeuralEngine`)
for `decoder_step`, kept `.cpuOnly` honored when the manager is
explicitly requested as CPU-only. Comment rewritten with the actual
numbers.

## Phase C v2 — chunked nanocodec (done; M2)

`convert_nanocodec.py` already accepts `--max-frames N`. Built T_in ∈
{8, 16, 24, 32} and ran `per_module/chunked_parity.py` to measure
audio-level SNR for the stitched output vs the single-call reference.

### Latency (cpu_and_neural_engine, 10-iter median)

| T_in | Output samples | ANE % | Median predict (ms) | Notes |
|------|----------------|-------|---------------------|-------|
| 8 | 8192 | 88.4 % | 14.87 | 10 dilated convs CPU |
| 16 | 16384 | 71.5 % | 32.37 | + 2 ops over W=16384 |
| 24 | 24576 | 43.4 % | 38.41 | dilated convs + post stages CPU |
| 32 | 32768 | 0 % (CPU) | 215.86 | `ANECCompile() FAILED` whole-graph |

### Audio parity (random tokens, seed=42, T=256)

`[A] = PyTorch sin² vs CoreML T=256 single-call` (Taylor5 floor)
`[B] = PyTorch sin² vs stitched`
`[C] = CoreML T=256 single-call vs stitched` (pure chunking artifact)

| T_in | stride | overlap | [A] dB | [B] dB | [C] dB |
|------|--------|---------|--------|--------|--------|
| 8 | 8 | 0 | 11.56 | **−1.38** ✗ | **−1.44** ✗ |
| 16 | 8 | 8 | 11.56 | 5.40 | 5.67 |
| 24 | 8 | 16 | 11.56 | 8.74 | **11.46** ✓ |
| 32 | 8 | 24 | 11.56 | 8.79 | 11.70 |
| 32 | 16 | 16 | 11.56 | 8.76 | 11.62 |

Receptive field at the codec input level is **16 frames**; beyond that
[C] is at the Taylor5 noise floor and adding more left-context only buys
+0.2 dB. Larger stride is fine as long as overlap stays ≥ 16.

### Operating point

**T_in = 24, stride = 8, overlap = 16 frames** (16384 leading samples
discarded per call, 8192 fresh samples kept).

| Workload | Per-call | 240-frame wall | RTFx codec only |
|----------|----------|----------------|-----------------|
| Phase 0 (T=256, all-CPU) | n/a | ~8.76 s | ~1.27× |
| Phase C v2 (T=24, 30× ANE) | 38.41 ms | **~1.15 s** | **~9.6×** |

~7.6× speedup on the codec stage. Chunking error [C]=11.46 dB now ≈
Taylor5 floor [A]=11.56 dB, so Snake quality is the remaining
bottleneck (Phase C+).

### Files

`per_module/chunked_parity.py` — parity sweep harness (parameterized by
`--t-in` and `--stride`).
`build/nanocodec_decoder_t{8,16,24,32}.mlpackage` — converted artifacts.
`compiled/build/nanocodec_decoder_t{8,16,24,32}.mlmodelc` — ready for
ANE warmup.
`per_module/results/chunked_*.npy` — saved waveforms for inspection.

## Phase C v2 step 5 — Swift chunked-inference wrapper (done)

`Sources/FluidAudio/TTS/Magpie/Pipeline/Synthesize/MagpieNanocodec.swift`
auto-detects T_in from the model description and slides a 24-frame
window with stride 8 over the codec sequence, discarding the leading
16384 samples of each call. `MagpieModelStore.swift` selects between
`nanocodec_decoder_t24_v2.mlmodelc` (fp32, default) and
`nanocodec_decoder_t24.mlmodelc` (fp16, opt-in) via
`MagpieNanocodecPrecision`, falling back to the monolithic
`nanocodec_decoder.mlmodelc` if neither t24 build is present. Asset
manifest in `MagpieResourceDownloader` updated.

## Phase C v2 step 6 — fp32 weights (audio fidelity, done)

The fp16-converted T=24 build was audibly noisy on voiced segments —
silence-RMS metrics hid it (quiet-window noise floor only ~3.5 dB above
PyTorch reference) but A/B listening against PyTorch sin² made the
speech-correlated quantization noise obvious.

Diagnosis: `compute_precision=ct.precision.FLOAT16` in `convert_nanocodec.py`
quantizes all 96 Snake stages and 5 upsample stages to fp16 weights.
PyTorch Taylor5Clipped (Snake monkey-patched, otherwise fp32) sounds
pitch-perfect; CoreML Taylor5 with fp16 weights sounds noisy. Snake
approximation is innocent — fp16 weight quantization is the cause.

Fix: `convert_nanocodec.py` parameterized with `--precision {fp32,fp16}`,
default `fp32`. ANE is fp16-only, so fp32 forces CPU.

| Build | Compute | Quiet-RMS noise floor |
|---|---|---|
| PyTorch sin² (gold) | CPU fp32 | −77.4 dBFS |
| T=24 chunked, fp32 weights | CPU | **−73.6 dBFS** ← current |
| T=24 chunked, fp16 weights | CPU | −73.9 dBFS (audibly noisy) |
| T=24 chunked, fp16 weights | ANE | −66.8 dBFS (audibly noisy) |

| Build | Nanocodec wall (M2, ~11 s utterance) | RTFx end-to-end |
|---|---|---|
| T=256 mono, fp16 weights, CPU | ~8.76 s | ~0.62× |
| T=24 chunked, fp16 weights, ANE | ~2.28 s | ~1.21× (noisy) |
| T=24 chunked, fp16 weights, CPU | ~2.22 s | ~2.25× (noisy) |
| T=24 chunked, fp32 weights, CPU | 8.5–9.7 s | **~1.34× median** ← current |

fp32 trades ~4× codec throughput for fidelity; pipeline stays real-time.

`MagpieModelStore.swift` pins nanocodec to `.cpuOnly` regardless of the
caller-provided `computeUnits` (matches fp32-weight requirement).

## Phase C v2 step 7 — edge-replication padding (start-of-utterance pop, done)

The chunked path produced a sharp click in the first ~30 ms of every
utterance. Numerical: chunked wav peak in first 10 ms = 0.6359 vs mono's
0.0009 (~700× louder).

Cause: zero-padding the first call's left context with code 0. Code 0
is a real codebook entry, not silence — the codec was never trained to
see it as a stationary 16-frame prefix, so the dilated convs fire a
transient at t=0.

Fix: `MagpieNanocodec.swift` clamps out-of-range source indices to
`[0, T-1]` and replicates `row[0]` (first call's left context) and
`row[T-1]` (last call's right context). The dilated convs see a
stationary signal that matches the AR loop's near-silent first frame.

| Variant | Peak in first 10 ms |
|---|---|
| Mono CPU (no chunking) | 0.0009 |
| Chunked, zero-pad | 0.6359 |
| Chunked, edge-replicate | **0.0003** ← current |

User-confirmed audibly clean.

## Phase F — nanocodec noise isolation (done; full fp32 required)

Phase C v2 step 6 already established that fp16 weight quantization, not
the Snake approximation, is the audible noise source. Phase F is the
exhaustive sweep that asked: **is there a smaller fp32 island inside
nanocodec that retains audibility while keeping the rest fp16?** Answer:
no. Full fp32 is required.

### Phase F.1 — mono fp32 vs fp16 audibility A/B (done)

Built `nanocodec_decoder_mono_fp32.mlpackage` (T=256 single-call, fp32
weights) via `convert_nanocodec.py --max-frames 256 --precision fp32`
and ran `/tmp/mono_fp16_vs_fp32.py`: 4-way A/B with PyTorch sin² gold,
PyTorch Taylor5 (in-place patched, fp32), CoreML T=256 fp16, CoreML
T=256 fp32.

Random-token AR sample, T=72 codes (~3.3 s):

| Comparison | SNR (dB) | Interpretation |
|---|---|---|
| PyTorch sin²        vs PyTorch Taylor5  | 15.39 | Snake approximation alone |
| PyTorch sin²        vs CoreML fp32      | 15.39 | Same — CoreML conversion bit-exact at fp32 |
| PyTorch Taylor5     vs CoreML fp32      | **117.59** | CoreML fp32 = bit-exact PyTorch Taylor5 |
| PyTorch Taylor5     vs CoreML fp16      | **26.80** | Entire fp16 quantization budget |

Audibility (user-confirmed): PyTorch Taylor5 and CoreML fp32 both clean;
CoreML fp16 noisy. Confirms fp16 weight quantization is the sole audible
defect. Snake approximation is acoustically transparent despite 15 dB
SNR floor (structural phase offset, not noise).

### Phase F.2 — mixed-precision sweep by op_type (done; no island works)

`/tmp/mixed_precision_sweep.py` builds 3 variants via
`coremltools.converters.mil.mil.passes.defs.quantization.FP16ComputePrecision(op_selector=…)`:

- `v_convs_fp32` — `op.op_type in ("conv", "conv_transpose")` kept fp32, rest fp16
- `v_acts_fp32` — convs fp16, all activations (mul/add/clip/tanh) fp32
- `v_snake_fp32` — clip + Snake-scoped mul/add fp32, rest fp16

Decoded against the same AR-emitted codes as F.1:

| Variant            | SNR vs sin² | SNR vs fp32 | Pred(s) | Size(MB) |
|---|---|---|---|---|
| v_full_fp32        | 18.48 | **211.22** | 1.81 | 121.0 |
| v_full_fp16        | 16.65 | 27.43 | 0.72 | 60.9 |
| v_convs_fp32       | 18.46 | 48.05 | 1.54 | **121.1** |
| v_acts_fp32        | 16.72 | 27.76 | 2.31 | 60.9 |
| v_snake_fp32       | 16.65 | 27.43 | 1.73 | 60.9 |

User audibility: `v_full_fp32` clean; **all four other variants** —
including `v_convs_fp32` at 48 dB SNR — audibly noisy. The 48 dB →
27 dB gap is the same kind of noise, just less of it; even 48 dB SNR
isn't clean enough perceptually. Convs hold most of the noise budget
but activations contribute too. No op-type island works.

### Phase F.2b — per-location sweep (blocked by coremltools)

`/tmp/mixed_precision_sweep_b.py` attempted scope-string filtering on
`op.scopes["TORCHSCRIPT_MODULE_NAME"]` to keep specific HiFi-GAN
locations fp32:

- `v_last2_fp32` — `up_sample_conv_layers.{3,4}` + `res_layers.{3,4}` + post head fp32
- `v_output_head_fp32` — only `post_conv` + `out_activation` fp32
- `v_embeddings_fp32` — only `pre_conv` + FSQ dequantization fp32

Result — all three came out **bit-identical to v_full_fp16** (60.9 MB,
SNR vs fp32 = 26.80 dB). Logging the matched scopes returned an empty
counter: `op.scopes` is **not populated** when `FP16ComputePrecision`'s
`op_selector` runs in this coremltools version (8.x). The selector
fires on bare ops with op_type but no Torch scope metadata, so any
scope-substring test trivially returns "convert to fp16".

Per-location filtering at the op_selector level is unavailable. The
only path to per-stage precision control is post-conversion MIL graph
rewriting, which is out of scope for this project.

### Conclusion

| Approach | Result |
|---|---|
| Whole-graph fp32 | clean (current production) |
| op_type filter (convs only fp32) | audibly noisy at 48 dB SNR |
| op_type filter (activations / Snake only fp32) | identical to fp16 |
| scope filter (per-stage / per-location) | not supported by coremltools op_selector |

**Production default stays on `nanocodec_decoder_t24_v2.mlpackage`**
(full fp32 weights, CPU-only, ~1.3× RTFx; renamed from the intermediate
`_fp32` artifact). The fp16 build remains shipped as
`nanocodec_decoder_t24.mlpackage` for the opt-in
`MagpieNanocodecPrecision.fp16` selector. The only remaining ANE-recovery
path on a clean-sounding nanocodec would be model-level retraining or
QAT — out of scope. Phase F closed.

### Files

`/tmp/mono_fp16_vs_fp32.py` — F.1 4-way A/B harness.
`/tmp/mixed_precision_sweep.py` — F.2 op_type sweep (3 variants).
`/tmp/mixed_precision_sweep_b.py` — F.2b scope-filter attempt (3 variants).
`build/nanocodec_decoder_mono_fp32.mlpackage` — T=256 mono fp32 reference.
`build/nanocodec_decoder_mono_v_{convs,acts,snake}_fp32.mlpackage` — F.2 artifacts.
`build/nanocodec_decoder_mono_v_{last2,output_head,embeddings}_fp32.mlpackage` — F.2b artifacts.

## Pending

### Phase C+ — better Snake replacement (deferred)
1. Implement LUT-via-conv Snake.
2. Verify SNR > 30 dB against chunked sin² reference.
3. Re-convert nanocodec with the better Snake; verify ANE residency
   unchanged.

(Deferred — current Taylor5Clipped + fp32 + edge-pad sounds clean to
the ear despite ~17 dB SNR vs PyTorch sin². SNR is dominated by
structural Snake-approximation phase offset, not audible noise.)

### Phase D fusion — fused AR step
Merge `decoder_step + final_proj + local_transformer + 8 heads` into one
mlmodelc to eliminate per-call dispatch overhead. Update Swift port.

### Phase E — HuggingFace upload
Upload both t24 builds to the FluidAudio HF assets repo. User-managed.

- `nanocodec_decoder_t24_v2.mlmodelc` (fp32, default, audibly clean)
- `nanocodec_decoder_t24.mlmodelc` (fp16, opt-in, fast/ANE, audibly noisy)

Both share the same I/O contract; the runtime selector lives in
`MagpieModelStore` (`MagpieNanocodecPrecision`).

## Files

```
per_module/
├── __init__.py
├── modules.py                # diagnostic nn.Module wrappers (Snake variants, KV cache, weight_norm)
├── analyze.py                # Phase A driver: per-module conversion + ANE coverage
├── snake_parity.py           # Snake polynomial accuracy vs sin² reference
├── audio_parity.py           # codec output SNR vs PyTorch reference
├── nano_subgraph_probe.py    # Phase C+ progressive HiFi-GAN subgraph probe
└── results/
    ├── PHASE_A.md            # detailed Phase A + C+ findings (the report)
    ├── STATUS.md             # this file (cross-phase index)
    ├── ledger.json           # Phase A summary table
    ├── raw/                  # Phase A per-spec coreml-cli output
    ├── subgraph_ledger.json  # Phase C+ summary table
    └── subgraph_raw/         # Phase C+ per-spec coreml-cli output
```

## Reproducing

```bash
cd mobius/models/tts/magpie/coreml
uv sync

# Phase A (per-module ANE diagnostics)
uv run python per_module/analyze.py
cat per_module/results/ledger.json

# Phase C+ (subgraph threshold probe)
uv run python per_module/nano_subgraph_probe.py
cat per_module/results/subgraph_ledger.json
```
