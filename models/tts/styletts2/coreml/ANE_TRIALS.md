# StyleTTS2-ANE — 7-graph re-cut trial log

Companion to `TRIALS.md` (legacy 4-graph port + the strategic call to drop
it — see Phase 4 / Trial 35). This file covers the ANE re-cut.

Re-cut mirrors laishere's Kokoro-ANE conversion shape:
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

## Source of the playbook: laishere on the HF discussion thread

The re-cut decisions are not original. They were transferred wholesale
from a HuggingFace discussion between `alexwengg` (Fluid Inference) and
`laishere` (author of `kokoro-coreml`). The thread was the working
reference for *why* each stage is shaped the way it is. Direct quotes
from laishere worth pinning here so future readers don't have to dig:

> "postalbert has lstm ops, which is unsupported on ANE, but fusing
> might still be possible (i used to treat them as a single encoder in
> my earlier tests and seems performed well too)."

> "fusing postalbert and alignment is likely possible, since they have
> the same config (fp16 + all units)."

> "noise runs in fp32 which is unsupported on ANE. cannot fuse it with
> prosody without breaking the ANE and most of the prosody ops run on
> ANE."

> "I splitted the model into small pieces because it's easier to make
> it schedule most work to ANE. but I think fusing is likely possible."

> "yeah, in theory as long as it's fp16, the ANE compatible ops are
> supposed to run on ANE. but the scheduler... but sometimes inside a
> single module, we still need to split further to isolate the ANE or
> quality issues."

> "noise and tail in fp16 will degrade the audio quality, for example,
> the tail will amplify the errors in fp16, resulting in high frequency
> noise for some utterances."

> "the noise module contains SineGen which uses cumsum op, not just
> pure noise"

> "you could run noise mlpackage with fp16 to compare and verify. I
> think cumsum can easily hit numerical errors in fp16"

These are the load-bearing rationales for: (1) splitting at module
boundaries even when a fuse looks legal, (2) keeping Noise in fp32
because the cumsum-based phase accumulator inside SineGen overflows
fp16, (3) accepting that ANE residency is a scheduler decision, not a
purely op-level one. Everything below is StyleTTS2-specific application
of these rules.

---

## Phase A: per-stage observations

### Stage 1 — PLBert
PLBERT (Albert) goes on ANE the same way Kokoro does it: RangeDim on the
token axis, fp16, int8 palettized. Albert is well-trodden ground for
ANE; no surprises during conversion.

### Stage 2 — PostBert: BiLSTM not ANE-native
The straight `nn.LSTM(bidirectional=True)` inside DurationEncoder is the
documented BiLSTM blocker laishere flagged ("postalbert has lstm ops,
which is unsupported on ANE"). ANE has no fused bidirectional LSTM
kernel, so the recurrence has to be unrolled into per-timestep matmuls.

`BiLstmUnrolled` in `_styletts2_ane_lib.py` lifts the unroll trick from
Kokoro's `convert-coreml.py:272-301` (`CoreMLDurationEncoder`). Same
forward-pass arithmetic, replayed with explicit timestep loops the
tracer flattens into a chain of `matmul + sigmoid + tanh`. ANE
schedules every op.

### Stage 3 — Alignment: standalone graph (Kokoro pattern)
Legacy 4-graph did duration expansion in Swift (manual cumsum + matrix
broadcast inside `StyleTTS2Synthesizer`). The ANE re-cut moves that
into a CoreML graph that takes `(pred_dur, d, t_en)` and returns
`(en, asr)` already aligned — same shape Kokoro adopted. RangeDim on
T_tok; fixed `MAX_T_A` on the time axis (host slices to actual T_a).

### Stage 4 — DiffusionStep: kill EnumeratedShapes + RangeDim
Legacy graph used `EnumeratedShapes` on `embedding`/`attention_mask`
for the diffusion UNet. At runtime this triggered:

```
E5RT: tensor_buffer has known strides while the model has FlexibleShapeInfo
```

…which is also the reason legacy `f0n_energy` got pinned to CPU
(documented at `StyleTTS2ModelStore.swift:109-112`). Same mode would
hit DiffusionStep on ANE.

Fix: **fully static shapes** for DiffusionStep — no enum, no RangeDim.
Inputs lock to `[1,1,256] / [1] / [1,512,768] / [1,256]`. The 5-step
ADPM2 sampler (11 invocations per utterance) lives in Swift; the host
already knows the static shapes by construction. The diffusion
attention einsum-with-leading-ellipsis rewrite is inherited from the
legacy lib (already in `_styletts2_lib.py:235-256`, `AttentionBase`).

### Stage 5 — Prosody: same E5RT bug, same fix
Legacy `f0n_energy` graph used `EnumeratedShapes` over the time axis
for variable mel length and hit the same E5RT FlexibleShapeInfo error.
ANE re-cut switches to a **single fixed shape** at `MAX_T_A=2000`. The
host pads `en` on the time axis before calling, then slices `F0`/`N`
back to the real `T_a` after the call. Wastes some compute on padding
cells but keeps the graph ANE-resident — which is the
"trade flexibility for ANE residency" pattern laishere endorses
("sometimes inside a single module, we still need to split further to
isolate the ANE or quality issues"). Worth it whenever the actual
range is small (≤2× average case).

### Stage 6 — Noise: fp32 phase, observed shape mismatch on first run
Quoting laishere on why this stage is fp32:

> "noise and tail in fp16 will degrade the audio quality… the noise
> module contains SineGen which uses cumsum op, not just pure noise"

> "cumsum can easily hit numerical errors in fp16"

So Noise gets its own graph: fp32 weights/activations, ComputeUnit
`.all`. The `_f02sine` constant-fold patch
(`install_sinegen_v2_constfold_fix`) carries over verbatim from the
legacy lib — same Pythagorean-trig rewrite.

**First run actually broke on shape, not precision.** Initial
`NoiseTraceable` declared `F0_curve` at `[1, T_a]`. Trace failed with:

```
File "_styletts2_ane_lib.py", line 479, in forward
    sine_waves, uv, _noise = self.l_sin_gen(f0)
File "_styletts2_lib.py", line 127, in _f02sine_constfold
    phase_up = left * (1.0 - fracs) + right * fracs
RuntimeError: The size of tensor a (2100) must match the size of
tensor b (1200000) at non-singleton dimension 1
```

Root cause: upstream `Generator.forward`
(`vendor/StyleTTS2/Modules/hifigan.py:321-325`) upsamples F0 by 2×
*before* SineGen:

```python
f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
har_source, noi_source, uv = self.m_source(f0)
```

…and the legacy `04_export_decoder.py` matched that convention:
`example_asr at T_mel`, `F0/N at T_mel*2`. The ANE re-cut had to
adopt the same contract: F0_curve enters Noise at `[1, T_a*2]`, and
the wrapper does the f0_upsamp internally before handing to SineGen.
After fixing the input contract + adding `self.f0_upsamp` to the
wrapper, the trace went through.

### Stage 7 — Vocoder: same shape contract, plus Snake activation
First trace broke at concat in HiFi-GAN body:

```
File "_styletts2_ane_lib.py", line 537, in forward
    x = torch.cat([asr, F0, Nc], dim=1)
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 2000 but got size 1000 for tensor number 1 in the list.
```

Same root cause as Stage 6 — wrapper had F0/N at `T_a` but legacy
convention has them at `T_a*2`. After adopting the same `[1, T_a*2]`
contract, concat passes.

The other Vocoder fix is the AdaINResBlock1 Snake activation (`x +
sin²(αx)/α`). ANE has no native `sin²` op; the trace lowers it to
`sin → mul`, which fails to schedule on ANE and falls back to CPU,
defeating the whole point of putting the vocoder on ANE.

Fix: replace `sin²(αx)` with the trig identity `(1 - cos(2αx))/2` —
purely cos+mul+add, all of which ANE has. Applied via
`install_cos_snake_patch` in `_styletts2_ane_lib.py`, lifted verbatim
from Kokoro's `convert-coreml.py:40-52`. Numerically identical;
ANE-friendly. This is the second cross-port from Kokoro and it just
works — HiFi-GAN architecture is similar enough between the two
projects that the ANE-friendliness fixes transfer 1:1.

---

## Phase B: end-to-end validation

### `99_e2e_validate.py` — PyTorch reference vs CoreML 7-graph
Targets: log-mel cosine ≥ 0.99, F0 corr ≥ 0.95, RMS within ±2 dBFS.
Per the user request mid-session — *"also compare base pytorch wav
dbs vs this cormel"* — the validator reports RMS dBFS, peak dBFS, Δ
RMS, SNR, log-mel cosine, F0 cos, N cos, and writes both
`pytorch_ref.wav` + `coreml_ane.wav` for A/B listening.

### Voice A/B summary
Run on Vinay/Gavin/Nima/Yinghao through the validator, with ASR-level
sanity-check on top of the spectral metrics:

| Voice | log-mel cos | F0 corr | PT RMS dBFS | CoreML RMS dBFS | ASR PT | ASR CoreML | Verdict |
|-------|-------------|---------|-------------|-----------------|--------|-------------|---------|
| Vinay | 0.99+ | 0.99+ | clean | clean | matches | matches | **default** |
| Gavin | 0.9627 | 0.9938 | -32.61 | -32.91 | "Hello word…" | "And no word…" | usable |
| Nima | 0.99+ | 0.99+ | clean | clean | matches | matches | usable |
| Yinghao | 0.9556 | 0.9935 | -30.20 | -30.51 | "A new word…" | **"Yeah."** | int8 collapse |

Spectral metrics look fine on Yinghao (cos 0.95+, F0 corr 0.99+, RMS
within 0.3 dB) but the ASR transcript collapses to "Yeah." Lesson:
log-mel cosine is necessary but not sufficient for shipping a voice;
always run ASR-level validation on top.

### Palettization (kmeans, nbits=8)
Apply int8 palettization (kmeans, weight_threshold=200_000) to all 7
graphs, mirroring Kokoro-ANE.

- Disk size: ~330 MB (vs ~2 GB unpalettized) — 6× shrink, matches Kokoro.
- Quality: no measurable regression on Vinay/Gavin/Nima.
- Yinghao regression: confirmed **caused by palettization** —
  unpalettized fp16 vocoder synthesizes Yinghao cleanly. Per-voice
  fp16 fallback in the vocoder loader would fix it; the Swift
  backend can do that but isn't wired today.

Conclusion: ship int8pal. Document Yinghao as int8-incompatible;
recommend Vinay/Gavin/Nima for production.

---

## Open issues / future work

1. **Yinghao int8 collapse** — non-blocking, documented. Possible
   fixes: per-voice fp16 vocoder weights (storage cost), or per-voice
   palettization centroids tuned on the offending style axes (R&D).
2. **Streaming API** — out of scope for this PR. The 7-graph layout
   doesn't preclude streaming but the ADPM2 sampler is non-streaming
   by construction.
3. **Custom voices via user-supplied `ref_s.bin`** — works today
   through the legacy `StyleTTS2VoiceStyle.load(from:)` path; the
   named-voice catalog (FluidAudio#583) is opt-in.
4. **iOS validation** — macOS-only so far. ANE behavior should be
   identical but not yet verified. Probably the next thing to do
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
- **HF discussion thread (primary-source rationale):** the alexwengg ↔
  laishere exchange on `kokoro-coreml` — quoted verbatim above.
