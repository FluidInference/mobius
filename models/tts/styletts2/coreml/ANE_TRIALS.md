# StyleTTS2-ANE — 7-graph re-cut trial log

Companion to `TRIALS.md` (which covers the legacy 4-graph CoreML port and
the strategic decision to deprecate it — see Phase 4 / Trial 35). This file
covers the per-stage conversion trial-and-error for the ANE re-cut.

The re-cut mirrors laishere's Kokoro-ANE conversion shape:
**Albert → PostAlbert → Alignment → DiffusionStep → Prosody → Noise → Vocoder**.
Result is 7 `.mlpackage` / `.mlmodelc` bundles in `coreml/build/ane/`.
StyleTTS2's HiFi-GAN is iSTFT-free, so Kokoro's separate Tail collapses
into the Vocoder. The ADPM2 sampler stays in Swift (`StyleTTS2Sampler` is
model-agnostic and is reused unchanged).

| # | Graph | Compute | Precision | Shape regime |
|---|-------|---------|-----------|--------------|
| 1 | PLBert (Albert) | `cpuAndNeuralEngine` | fp16 + int8pal | RangeDim(2..MAX_T_TOK) |
| 2 | PostBert (TextEncoder + DurationEncoder + duration head) | `cpuAndNeuralEngine` | fp16 + int8pal | RangeDim(2..MAX_T_TOK) |
| 3 | Alignment (cumsum + broadcast) | `cpuAndNeuralEngine` | fp16 + int8pal | RangeDim T_tok / fixed T_a |
| 4 | DiffusionStep (single ADPM2 denoise) | `cpuAndNeuralEngine` | fp16 + int8pal | **fully static** |
| 5 | Prosody (F0Ntrain) | `cpuAndNeuralEngine` | fp16 + int8pal | **fully static** at MAX_T_A |
| 6 | Noise (SineGen alone) | `.all` | **fp32** + int8pal | fixed at MAX_T_A * 2 |
| 7 | Vocoder (HiFi-GAN body) | `cpuAndNeuralEngine` | fp16 + int8pal | fixed at MAX_T_A |

---

## Phase A: per-stage trial-and-error

### Trial A1 — PLBert: smooth one
**Approach:** Wrap upstream PLBERT (Albert) verbatim with
`RangeDim(2..MAX_T_TOK)` on the token axis. fp16 + int8pal kmeans.
**Result:** PASS first try. ANE-resident, parity 0.999+.
**Commentary:** Albert is well-trodden ground for ANE — same stack
laishere ships in Kokoro. No surprises.

### Trial A2 — PostBert: BiLSTM not ANE-native
**Symptom:** First trace had a vanilla `nn.LSTM(bidirectional=True)`
inside DurationEncoder. CoreML accepts the trace, but ANE rejects the
LSTM op and falls back to CPU+GPU per-step, demolishing the latency.
**Root cause:** ANE's compiler doesn't have a fused bidirectional
LSTM kernel; the canonical workaround is to unroll the recurrence
into per-timestep matmuls.
**Fix:** `BiLstmUnrolled` helper in `_styletts2_ane_lib.py` — lifts
the trick from Kokoro's `convert-coreml.py:272-301`
(`CoreMLDurationEncoder`). Drops the LSTM, replays the same
forward-pass arithmetic with explicit timestep loops the tracer
unrolls into a chain of matmul + sigmoid + tanh. Same numerics, ANE
accepts every op.
**Result:** PASS. ANE-resident, parity 0.999+.

### Trial A3 — Alignment: standalone graph (Kokoro pattern)
**Approach:** The legacy 4-graph did duration expansion in Swift —
manual cumsum + matrix broadcast inside `StyleTTS2Synthesizer`. The
ANE re-cut mirrors Kokoro: a small CoreML graph that takes
`(pred_dur, d, t_en)` and returns `(en, asr)` already aligned.
Lets the host code stay short and lets CoreML own the broadcast op.
**Result:** PASS. RangeDim on T_tok; fixed MAX_T_A on the output time
axis (host slices to actual T_a).

### Trial A4 — DiffusionStep: kill EnumeratedShapes + RangeDim attention_mask
**Symptom:** Legacy graph used `EnumeratedShapes` on
`embedding`/`attention_mask` for the diffusion UNet. Triggered
the runtime error
```
E5RT: tensor_buffer has known strides while the model has FlexibleShapeInfo
```
Forced legacy `f0n_energy` to CPU. Same mode would hit DiffusionStep
on ANE.
**Root cause:** The CoreML runtime's flexible-shape strider
disagrees with ANE's stride assumptions when ranges are used on
multi-axis inputs.
**Fix:** **Fully static shapes** for DiffusionStep — no enum, no
RangeDim. Inputs nail down to `[1,1,256] / [1] / [1,512,768] / [1,256]`.
The 5-step ADPM2 sampler (11 invocations per utterance) lives in
Swift; the host already knows the static shapes by construction.
The diffusion attention einsum-with-leading-ellipsis rewrite is
inherited from the legacy lib (already in
`_styletts2_lib.py:235-256`, `AttentionBase`).
**Result:** PASS. ANE-resident.

### Trial A5 — Prosody: same E5RT bug, same fix
**Symptom:** Legacy `f0n_energy` graph used `EnumeratedShapes` over
the time axis for variable mel length. Hit the same E5RT
FlexibleShapeInfo error and got pinned to CPU.
**Fix:** **Single fixed shape** at `MAX_T_A=2000`. The host pads `en`
on the time axis before calling, then slices `F0`/`N` back to the
real `T_a` after the call. Wastes some compute on padding cells but
keeps the graph ANE-resident.
**Result:** PASS. ANE-resident, parity 0.999+. Total runtime is still
much smaller than the legacy CPU-pinned graph.
**Commentary:** This is the prototypical "trade flexibility for ANE
residency" pattern. Worth it whenever the actual range is small (≤2×
average case).

### Trial A6 — Noise: fp16 phase saturation forces fp32
**Symptom:** Initial trace put SineGen (`_f02sine`) in the same fp16
graph as the rest of the vocoder. Long sequences (T_a > ~800)
produced audible high-frequency noise — the phase accumulator
saturates fp16's 11-bit mantissa, pulling sine output away from the
reference.
**Root cause:** Fundamental fp16 precision limit on cumulative-sum
phase accumulators. Kokoro hit the exact same wall; their solution
was a dedicated fp32 SineGen graph.
**Fix:** Split SineGen into its own `Noise` graph, fp32 weights/
activations, ComputeUnit `.all`. The `_f02sine` constant-fold patch
(`install_sinegen_v2_constfold_fix`) carries over verbatim from the
legacy lib — same Pythagorean-trig rewrite.
**Result:** PASS. The only fp32 graph in the pipeline. Parity
restored. ANE will run parts on fp16 emulation but that's fine — the
numerics that matter are fp32.
**Commentary:** Phase precision in fp16 is a recurring lesson —
anything with a `cumsum` over a long axis needs fp32. This will keep
biting future generative-audio ports.

### Trial A7 — Vocoder: Snake activation → cos-identity rewrite
**Symptom:** AdaINResBlock1 uses Snake activation (`x + sin²(αx)/α`).
ANE compiler doesn't have a native `sin²` op; trace lowers it to
`sin → mul`, which then fails to schedule on ANE. Falls back to CPU,
defeats the whole point of putting the vocoder on ANE.
**Fix:** Replace `sin²(αx)` with the trig identity `(1 - cos(2αx))/2`
— purely cos+mul+add, all of which ANE has. Apply via
`install_cos_snake_patch` in `_styletts2_ane_lib.py` (lifted verbatim
from Kokoro's `convert-coreml.py:40-52`). Numerically identical;
ANE-friendly.
**Result:** PASS. ANE-resident vocoder.
**Commentary:** This is the second cross-port from Kokoro and it
just works. The HiFi-GAN architecture is so similar between the two
projects that the ANE-friendliness fixes transfer 1:1.

---

## Phase B: end-to-end validation

### Trial B1 — `99_e2e_validate.py`: log-mel cosine ≥ 0.99 vs PT ref
**Approach:** Wire all 7 ANE graphs in Python, run against PyTorch
reference. Targets: log-mel cosine ≥ 0.99, F0 corr ≥ 0.95, RMS within
±2 dBFS.
**Result:** PASS on Vinay/Gavin/Nima; **FAIL on Yinghao** — see
`TRIALS.md` Trial 29.
**Voice A/B summary:**

| Voice | log-mel cos | F0 corr | PT RMS dBFS | CoreML RMS dBFS | ASR PT | ASR CoreML | Verdict |
|-------|-------------|---------|-------------|-----------------|--------|-------------|---------|
| Vinay | 0.99+ | 0.99+ | clean | clean | matches | matches | **default** |
| Gavin | 0.9627 | 0.9938 | -32.61 | -32.91 | "Hello word…" | "And no word…" | usable |
| Nima | 0.99+ | 0.99+ | clean | clean | matches | matches | usable |
| Yinghao | 0.9556 | 0.9935 | -30.20 | -30.51 | "A new word…" | **"Yeah."** | int8 collapse |

The log-mel and F0 metrics look fine for Yinghao but the ASR
transcript collapse is the disposing signal — the synth is
phonetically broken even though the spectral envelope matches. Lesson:
log-mel cosine is necessary but not sufficient for shipping a voice;
always run ASR-level validation.

### Trial B2 — Palettization (kmeans, nbits=8)
**Approach:** Apply `int8` palettization (kmeans, weight_threshold=
200_000) to all 7 graphs, mirroring Kokoro-ANE.
**Result:**
- Disk size: ~330 MB (vs ~2 GB unpalettized) — 6× shrink, matches
  Kokoro.
- Quality: no measurable regression on Vinay/Gavin/Nima.
- Yinghao regression: confirmed **caused by palettization** —
  unpalettized fp16 vocoder synthesizes Yinghao cleanly. Fix would
  require per-voice fp16 fallback in the vocoder loader, which the
  Swift backend can do but isn't wired today.
**Conclusion:** Ship int8pal. Document Yinghao as int8-incompatible;
recommend Vinay/Gavin/Nima for production.

---

## Open issues / future work

1. **Yinghao int8 collapse** — non-blocking but documented. Possible
   fixes: per-voice fp16 vocoder weights (storage cost), or
   per-voice palettization centroids tuned on the offending style
   axes (R&D).
2. **Streaming API** — out of scope for this PR; the 7-graph layout
   doesn't preclude streaming but the ADPM2 sampler is non-streaming
   by construction.
3. **Custom voices via user-supplied `ref_s.bin`** — works today
   through the legacy `StyleTTS2VoiceStyle.load(from:)` path; the
   named-voice catalog (FluidAudio#583) is opt-in.
4. **iOS validation** — macOS-only so far. ANE behavior should be
   identical but we haven't verified. Probably the next thing to do
   after merging.

---

## Cross-references

- **Strategic context (why ANE, why drop legacy):** `TRIALS.md`
  Phase 4 / Trial 35.
- **Voice catalog wiring (Swift):** FluidAudio PR #583.
- **ANE conversion scripts:** `scripts/ane/01_export_plbert.py` …
  `07_export_vocoder.py` + `_styletts2_ane_lib.py`.
- **Validator:** `scripts/ane/99_e2e_validate.py`.
- **Build artifacts (gitignored):** `coreml/build/ane/styletts2_ane_*.{mlpackage,mlmodelc}`.
- **Kokoro-ANE reference:** laishere's `convert-coreml.py` —
  `_cos_resblock1_forward` (lines 40-52) and `CoreMLDurationEncoder`
  (lines 272-301) are the two cross-ported helpers.
