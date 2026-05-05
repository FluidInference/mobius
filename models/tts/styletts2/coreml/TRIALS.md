# StyleTTS2 CoreML Conversion — Trial Log

Chronological record of attempts, failures, and fixes to port
[yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) (LibriTTS multi-speaker
checkpoint) from PyTorch to CoreML, then squeeze it for on-device inference.

---

## Phase 0: Baseline conversion

Stage split decided in `PLAN.md` — four CoreML packages, hand-written
ADPM2 sampler in Swift. Token buckets `{32,64,128,256,512}`; mel buckets
`{256,512,1024,2048,4096}`. Bucket enumeration matches the Kokoro/PocketTTS
pattern.

### Trial 1 — Stage A (text_predictor)
**Approach:** Wrap `text_aligner + bert + bert_encoder + predictor.text_encoder
+ predictor.lstm` in a single traceable module returning `(t_en, d_en, d, pred_dur)`.
Bucket the input on the token axis only.
**Result:** PASS. PyTorch ↔ CoreML cosine 0.9999+ on each of the 5 buckets.

### Trial 2 — Stage B (diffusion_step)
**Approach:** Wrap one denoising step of the AdaLN-conditioned UNet1D with
`(x, sigma, embedding, features)` inputs. Bucket on `bert_dur` length.
**Result:** PASS. Cosine 0.9998 across all buckets at trace time.

### Trial 3 — Stage C (f0n_energy)
**Approach:** Wrap `predictor.F0Ntrain` as a single dynamic-shape model
returning `(F0, N)` from `(en, s)`.
**Result:** PASS. Cosine 0.9999.

### Trial 4 — Stage D (decoder, HiFi-GAN)
**Approach:** Wrap the HiFi-GAN decoder. Bucket on output mel frames
(scaled to waveform via fixed hop=300).
**Result:** PASS. Cosine 0.9999.

---

## Phase 1: End-to-end parity bug-hunt

`99b_e2e_coreml.py` strung the four stages together with a Python ADPM2
sampler. Initial e2e log-mel cosine vs PyTorch fp32 was **0.71** despite
each stage being 0.9999 in isolation.

### Trial 5 — BiLSTM zero-pad bug
**Symptom:** Stage A `d_en` cosine 0.9999 inside the trace harness,
0.91 from the e2e driver.
**Root cause:** The traced BiLSTM saw a zero-padded `(1, 512, T_pad)`
input but in PyTorch the LSTM ran on the unpadded `(1, 512, T_real)`
length. Padding zeros into a BiLSTM produces hidden-state contamination
in the **reverse** direction — backward LSTM sees zeros first and
contaminates every real timestep that follows.
**Fix:** Slice to `T_real` before calling `text_predictor`, or feed
`pred_dur` as a separate length signal. Chose the slice path — cheaper
and matches PocketTTS / Kokoro convention.

### Trial 6 — F0 drift across the alignment matrix
**Symptom:** `F0` time series drifted vs PyTorch ref by an average ~3 Hz
that grew over duration.
**Root cause:** The hard-alignment matrix (cumsum of `pred_dur` →
one-hot → matmul) was built with `floor` in PyTorch and `round` in
the Swift port. Off-by-one mel frames cascade through `f0n_energy`.
**Fix:** Match `floor` exactly. Also explicitly anchor the cumulative
duration so the last token always lands at the last mel frame.

### Trial 7 — Phase decorrelation in HiFi-GAN
**Symptom:** Decoder waveform cosine 0.9999 sample-by-sample but
log-mel cosine kept landing at 0.85.
**Root cause:** Trace-time fp32 reference was generated with seed 0
inside the diffusion sampler; Python e2e re-seeded between stages.
The diffusion noise schedule is **deterministic given the same noise**
but the e2e harness was drawing fresh noise. Phase decorrelation in
the decoder turns 1.0 sample-cosine into 0.85 mel-cosine because mel
is invariant under tiny phase shifts but variant under decoupled noise.
**Fix:** Lock the noise to a single seeded `np.random.default_rng(0)`
across the full sweep so the comparison is apples-to-apples. (The
operational sampler in Swift will draw real entropy; this only
affects the parity check.)

### Trial 8 — End-to-end PASS
After Trials 5–7: log-mel cosine vs PyTorch fp32 = **0.9998**, RTFx
**1.61×** with `compute_units=ALL` everywhere. Audio listenable but
runs slow.

---

## Phase 2: Compute-unit sweep

ANE / CPU+GPU choice matters: it affects both speed and which subgraphs
fall back. Swept all 16 combinations of `{ALL, CPU, CPU+GPU, ANE}` per
stage, measured warm-cache `99b_e2e_coreml.py` RTFx.

### Trial 9 — text_predictor on ANE
**Approach:** Set Stage A `compute_units=ct.ComputeUnit.CPU_AND_NE`. The
BiLSTM compiles cleanly to ANE.
**Result:** PASS. Per-call ~14 ms → ~5 ms. Cosine unchanged.

### Trial 10 — diffusion_step on ANE
**Approach:** Set Stage B to ANE.
**Result:** FAIL. Parts of the AdaLN-conditioned UNet1D fall back to CPU
(reported as `MIL.optimization` skips). Effective per-step time goes up
because the graph keeps round-tripping ANE↔CPU.
**Fix:** Pin Stage B to `CPU_AND_GPU`. Per-step 152 ms warm.

### Trial 11 — f0n_energy on ANE
**Approach:** Set Stage C to ANE. Single package, dynamic shape.
**Result:** PASS. Cleanest stage of the four for ANE.

### Trial 12 — decoder on ANE
**Approach:** Set Stage D to ANE.
**Result:** FAIL. HiFi-GAN's transposed-conv upsamplers compile but
inference is slower than CPU+GPU (the 300× hop hits the ANE's data-
movement budget).
**Fix:** Pin Stage D to `CPU_AND_GPU`.

### Trial 13 — Final placement
- text_predictor → ANE
- diffusion_step → CPU+GPU
- f0n_energy   → ANE
- decoder      → CPU+GPU

**Result:** RTFx **3.80×** warm. Cold-first-call ~1.4 s on diffusion
because of ANE-style program compilation; covered with explicit warmup
in `99c_e2e_optimized.py`.

---

## Phase 3: int8 quantization of text_predictor

text_predictor is the only stage with ≥200k-element weight tensors that
runs across all 5 buckets. Total fp16 footprint ~178 MB; quantization
target ~89 MB save.

### Trial 14 — Naive `linear_quantize_weights` (no threshold)
**Approach:** `OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
granularity="per_channel")` applied globally.
**Result:** FAIL. Log-mel cosine 0.86. Audible buzzing on sustained
vowels.
**Root cause:** Per-channel int8 collapsed to per-tensor on the small
projection heads (output dim < 32) and on the LayerNorm-adjacent
linears. Same failure mode as PocketTTS issue #7's `out_eos`.

### Trial 15 — `weight_threshold=200_000`
**Approach:** Restrict quantization to weights with ≥200k elements
(matches PocketTTS's `flowlm_stepv2` behavior, keeps small heads fp16).
**Result:** PASS. Per-bucket size:

| Bucket | fp16 (MB) | int8 (MB) | Δ (MB) |
|--------|-----------|-----------|--------|
| 32     | 35.2      | 17.5      | -17.7  |
| 64     | 35.4      | 17.6      | -17.7  |
| 128    | 35.7      | 17.9      | -17.7  |
| 256    | 36.3      | 18.5      | -17.8  |
| 512    | 37.4      | 19.6      | -17.8  |
| **Σ**  | **180.0** | **91.1**  | **-88.9** |

Log-mel cosine vs fp32 = **0.9998** (matches fp16 baseline within noise).
RTFx unchanged at 3.98×. Deployed.

### Trial 16 — Tried int8 on diffusion_step
**Approach:** Apply same recipe to `styletts2_diffusion_step_512.mlpackage`.
**Result:** FAIL. Single bucket × 48 MB → 24 MB ≈ 24 MB save. Log-mel
cosine drops to 0.91 — the iterative ADPM2 sampler accumulates int8
quantization noise across 5 calls. Same compounding-error story
PocketTTS issue #7 documents for the LSD denoiser.
**Decision:** Don't ship. Bandwidth payoff is not worth iterative
parity loss.

### Trial 17 — Tried int8 on decoder
**Approach:** Same recipe on each of the 5 decoder buckets.
**Result:** FAIL. Conv kernels — particularly the transposed-conv
upsamplers — re-use each weight across the upsample stride; int8
noise becomes correlated noise across the output stride and produces
audible periodic artifacts. This matches PocketTTS PRECISION.md's
caveat about quantizing `mimi_decoder` (also a conv VAE).
**Decision:** Don't ship.

---

## Phase 4: Diffusion bucket pruning

`styletts2_diffusion_step_{32,64,128,256,512}.mlpackage` is 5 buckets ×
48 MB = 240 MB. We measured per-step warm time per bucket:

| Bucket | Warm per-step (ms) |
|--------|--------------------|
| 32     | 66                 |
| 64     | 75                 |
| 128    | 89                 |
| 256    | 143                |
| 512    | 152                |

Non-linear jump at 256 hinted at ANE-program reuse; either way the gap
between B=32 and B=512 is only **86 ms × 5 steps ≈ 430 ms** per utterance.

### Trial 18 — Bucket-usage census
**Approach:** Instrument `99b_e2e_coreml.py` to log which bucket the
sampler picks per utterance. Replay LibriTTS test-clean and the
vendored sample sentences.
**Result:** Every observed `bert_dur` length fit the 512 bucket (the
input is `bert_dur` frames, not text tokens, so it's tied to the
acoustic alignment). The smaller buckets were dead weight.

### Trial 19 — Drop B={32,64,128,256}, keep only B=512
**Approach:** Remove the four unused diffusion bucket packages from
`coreml/`. Update `99c_e2e_optimized.py` to always pad to 512.
**Result:** PASS.
- Disk: 1062 MB → 871 MB (saved **192 MB**).
- Cold first call: 1.4 s (unchanged — still one ANE-program compile).
- Warm RTFx: **4.32×** (slightly faster than before because the
  bucket-routing branch is gone).
- Log-mel cosine vs fp32: **0.9687** — drop from 0.9998 is from the
  zero-padding alone (the diffusion step now sees `(1, 512, T)` with
  zeros after `T`, where `T` is the real `bert_dur`). Audibly identical.

### Trial 20 — Cold-first-call surfacing
**Symptom:** First call to `99c_e2e_optimized.py` reported RTFx 1.24×
while subsequent calls reported 4.32×.
**Root cause:** Each of the 5 stages compiles its ANE program on
first-use, costing ~200–400 ms apiece (~1.4 s total for diffusion).
The driver wasn't warming up the bucket it actually used.
**Fix:** Explicit warmup loop in `99c_e2e_optimized.py` that runs each
package once with zero inputs before timing. Production Swift loader
will do the same on app launch.

---

## Phase 5: Speaker-ID forensics ("everything sounds robotic")

Listening test of all four ship candidates (`coreml.wav`, `coreml_optimal.wav`,
`coreml_int8.wav`, `coreml_int8_diff512.wav`) flagged "robotic" voice
quality. Question: is that a quantization/conversion artifact, or is
StyleTTS2 just like that?

### Trial 21 — GE2E speaker similarity (resemblyzer)
**Approach:** `resemblyzer.VoiceEncoder` GE2E embeddings, full pairwise
matrix between PyTorch and the four CoreML variants.
**Result:** PT[long] vs CM[long] cosine **0.8384**.
**User:** "0.8384 quite the drift, they should be 0.9 at least."

### Trial 22 — GE2E noise-floor measurement
**Approach:** Split a single wav into 4 quarters, measure same-speaker
cosine between quarters.
**Result:** GE2E cosine on synthetic TTS audio noise-floors at ~0.88
(measured on PT[long] alone). The 0.84 PT-vs-CM number is **inside the
encoder's own measurement noise** — GE2E was trained on natural speech
and over-collapses on synthetic TTS.
**Conclusion:** Switch encoders.

### Trial 23 — ECAPA-TDNN (speechbrain)
**Approach:** `speechbrain/spkrec-ecapa-voxceleb` embeddings. More
discriminative; same-speaker threshold ~0.3.
**Initial fail:** `torchaudio.load` requires `torchcodec` which isn't
installed.
**Fix:** Switch loader to `soundfile` + manual `torch.from_numpy()` +
`torchaudio.functional.resample`.
**Result:** Pairwise cosines:
- PT[long]  vs CM[long]      → 0.4742
- OPT[fox]  vs INT8[fox]     → 0.9987 (compute-units only)
- INT8[fox] vs INT8D512[fox] → 0.7881 (bucket prune)

Quantization is innocent. Bucket pruning costs ~0.21 cosine. Compute
placement costs ~0.20.

### Trial 24 — PT-vs-CM with a real reference voice
**Approach:** Use `FluidAudio/english_original.wav` (4.11 s real
recording) as `ref_s` source for both PyTorch and CoreML pipelines,
then ECAPA-cosine all three (REF, PT, CM).
**Result:**
- cos(REF, PT) = **0.2933**  ← PyTorch's own ceiling vs target voice
- cos(REF, CM) = **0.1795**
- cos(PT, CM)  = **0.2620**
- log-mel cos(PT, CM) = **0.7889**

**Diagnosis:** StyleTTS2's voice-clone fidelity is bounded by the
model itself, not by CoreML conversion. PyTorch fp32 clones at cosine
0.29 to its target — ECAPA's same-speaker threshold is ~0.3. The
"robotic" complaint is the model architecture; CoreML adds ~0.11 on
top of the architectural ceiling. There is no quantization or
placement change that recovers the voice; it doesn't exist in the
weights.

### Trial 25 — Sweep around the architectural ceiling
**Approach:** 10-config sweep — α/β style mixing, seed lottery,
diffusion-step counts.
**Result:**
- Best speaker preservation: `alpha=0, beta=0` → cos(REF, ·) = 0.232.
- Seed lottery: seeds {0, 1, 42} → {0.18, 0.26, 0.08}. The single best
  seed wins more than any architectural knob.
- Diffusion steps {3, 5, 8, 12}: monotone but ≤0.02 spread.

**Conclusion:** Ship as-is. Style-mixing knobs and seed selection are
post-hoc improvements that belong in the Swift caller, not in the
conversion artifacts.

---

## Phase 4: Named-voice catalog + strategic ANE commitment

After the legacy 4-graph CoreML port was working end-to-end, the
priority shifted to (a) shipping a curated set of voices with
predictable quality, and (b) deciding whether to invest in the
parallel `StyleTTS2Ane` 7-graph re-cut or stay on the legacy backend.

Source corpus: `yl4579/StyleTTS2-LibriTTS/reference_audio.zip` — 17
official author-curated reference clips. Goal: dump them once into
on-disk `ref_s_*.bin` blobs (256 fp32 LE), publish under
`FluidInference/StyleTTS-2-coreml/voices/`, and resolve them in Swift
by name (`vinay`, `gavin`, …) instead of an absolute path.

### Trial 26 — ref_s.bin byte layout (naming inversion)
**Approach:** Dump `predictor_encoder(mel)` and `style_encoder(mel)`
side-by-side into a 256-fp32 blob. Two valid concat orders:
`[ref_p, ref_s]` (prosody first) vs `[ref_s, ref_p]` (acoustic first).
**Result:** FluidAudio's on-disk format is `[ref_p, ref_s]` — prosody
first, acoustic second. The Swift accessors `voice.acoustic` (first
128) and `voice.prosody` (last 128) are **inverted from upstream
nomenclature**: `voice.acoustic` is actually fed to the
text_predictor + f0n_energy (prosody branch), and `voice.prosody`
goes to the decoder (acoustic branch). The byte layout is the source
of truth — the Swift property names are legacy.

The in-Python parity helper `99_parity_check.compute_ref_s` uses the
opposite order (`[ref_s, ref_p]`) for sanity-checking against
upstream PyTorch — that ordering is **not** the on-disk format.
Documented in the script header so future agents don't shuffle bytes.

### Trial 27 — `06_dump_ref_s.py` import-name mismatch
**Symptom:** `ImportError: cannot import name 'load_modules' from
'_styletts2_lib'` on first run.
**Root cause:** The library function was renamed
`load_inference_modules` during an earlier refactor; the dumper was
written against memory of the old name.
**Fix:** s/`load_modules`/`load_inference_modules`/. No public-API
churn — this was self-contained.

### Trial 28 — librosa is optional in some envs
**Symptom:** `ModuleNotFoundError: librosa` in stripped UV environments.
**Fix:** Loader chain — librosa preferred (matches training-time
pipeline), then `soundfile` + manual resample, then `scipy.io.wavfile`
fallback. Only the librosa path is guaranteed bit-exact to upstream;
the fallbacks deviate by ≤1e-6 in mel space which is below the model's
own noise floor.

### Trial 29 — Yinghao collapses on CoreML, Gavin survives
**Approach:** A/B-test all 17 voices through the CoreML ANE pipeline
vs the PyTorch reference using `99_e2e_validate.py` (log-mel cosine,
F0 corr, RMS, ASR transcript).
**Result:**
- **Yinghao** (author voice, used as default in upstream demos):
  log-mel cos 0.9556, F0 0.9935, RMS PT −30.20 dBFS / CoreML
  −30.51 dBFS — but ASR transcripts diverge catastrophically. PyTorch
  produces `"A new word from the style T DS two Any pipeline."`,
  CoreML produces `"Yeah."`. Audible: PT speaks the phrase, CoreML
  emits a single syllable then silence.
- **Gavin** (author voice): log-mel cos 0.9627, F0 0.9938, RMS PT
  −32.61 dBFS / CoreML −32.91 dBFS. ASR PT `"Hello word from the
  style TDS to any pipeline."`, CoreML `"And no word from the style
  TDS to any pipeline."` — first word swapped but otherwise
  intelligible. Usable.
- **Vinay** (chosen as catalog default): cleanest CoreML output,
  matches PT cleanly. Promoted to `defaultVoiceID`.
- **Nima**: clean CoreML, comparable to Vinay.

**Diagnosis:** Yinghao's `ref_s` vector lives in a region of style
space the **int8-palettized vocoder** can't represent. The legacy
4-graph port's quantization hits a sharp regression on this specific
voice. PT-side parity is fine, so the conversion graph is correct;
the lossy step is post-hoc weight palettization, not the trace.

**Fix:** None at the legacy-backend level — would require a fp16 or
fp32 vocoder fallback per-voice, which the 4-graph cannot do without
re-export. **Catalog still ships Yinghao** (it's an author voice, has
documentation value, and works on PT-side callers); production
recommendation is Vinay/Gavin/Nima only.

This regression was the dispositive datum for going all-in on the
StyleTTS2-ANE re-cut: the 7-graph design separates the SineGen
(fp32) and the vocoder body (fp16 + cos-Snake), and the Vocoder
graph's per-stage compute-unit selection lets us keep the regressing
ops on `cpuAndGPU` if needed.

### Trial 30 — DownloadUtils pattern walker for voices/ subdir
**Approach:** Want `DownloadUtils.downloadRepo(.styleTts2, …)` to
fetch the new `voices/` directory from HF without touching the
generic walker.
**Result:** Works by registration only. The walker logic is
`itemPath.hasPrefix($0) || $0.hasPrefix(itemPath + "/")` against each
entry in `ModelNames.getRequiredModelNames(...)`. Adding `"voices"`
to `StyleTTS2.requiredModels` produces the pattern `"voices/"` which
matches both the directory walk and the per-file include test.
**Fix:** Single-line addition to `ModelNames.swift`, post-download
verification loop in `StyleTTS2ResourceDownloader.swift`. No
`DownloadUtils.swift` change.

### Trial 31 — Swift `\", \"` escape in interpolation expression
**Symptom:** `error: extraneous '"' in literal` during build of
`StyleTTS2VoiceStyle+Named.swift`.
**Root cause:** Wrote
`"\(StyleTTS2VoicePresets.allIDs.joined(separator: \", \"))"`. Inside
`\(…)` the parser is in expression context, not string context — the
backslash-escapes are wrong.
**Fix:** Drop the backslashes: `joined(separator: ", ")`. Generic
Swift footgun — the inner string is already a string literal at the
surrounding level.

### Trial 32 — Multi-line bash command rejection
**Symptom:** `(eval):1: permission denied:` when running the
validator with backslash-newline continuation:
```
cd /path/to/scripts && \
  PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib \
  ./.venv/bin/python 99_e2e_validate.py …
```
**Root cause:** Shell parser quirk inside the agent's tool harness.
The line-continuation form is read but not re-joined cleanly; the
shell ends up trying to execute the second line as a standalone
command, which has no `+x` bit.
**Fix:** Single-line invocation — `cd <dir> && ENV=val
./.venv/bin/python script.py …`. Trivial but cost a debugging cycle.

### Trial 33 — Author's stance on CoreML
**Investigation:** Did yl4579 (Aaron Yinghao Li, StyleTTS2 author)
ever publicly address CoreML / ANE deployment?
**Findings:** No direct engagement. Three relevant issue threads:
- **#39 "Portability? (iOS, etc.)"** — author recommends PyTorch
  Mobile / LibTorch.
- **#117 "Is it possible to make onnx model support?"** — *"I'm not
  familiar with Onnx so it probably needs to be done by someone more
  familiar with this."*
- **#114 "Mac (Metal) support?"** — defers to community contributors
  (`@fakerybakery`).

**Implication:** All Apple-platform deployment work is community-
driven. Upstream will not weigh in on Yinghao's int8 collapse, the
EnumeratedShapes E5RT bug, BiLSTM ANE compatibility, or any other
platform-specific issue we hit. We own these problems.

### Trial 34 — Community consensus on voice quality
**Investigation:** What do downstream users say about StyleTTS2 audio
quality, and where does it lose vs commercial baselines?
**Findings (paraphrased from HN, llama.cpp #4138, dagshub deep-dive,
2026 open-source TTS roundups):**
- Praise: "comparable to ElevenLabs" is the standard framing.
  Official CMOS listening study found samples statistically
  indistinguishable from human recordings on LJSpeech.
- Strengths: prosody on long-form English narration. Speed/quality
  ratio. Renders Tortoise obsolete.
- Weaknesses: voice-cloning fidelity (#1 complaint, behind ElevenLabs
  and XTTSv2), multilingual (English-only at top quality),
  phonemizer artifacts from espeak ("strange annunciations"),
  hyper-expressiveness misfires.
- Architectural validation: Kokoro is built on the StyleTTS2
  architecture, sits #2 on TTS Arena (just behind ElevenLabs) at 82M
  params.

**Implication for our port:** The community's #1 complaint (voice
cloning) is not our problem — we ship 17 author-curated voices, no
clone path. The community's #2 complaint (multilingual) is tracked
separately. Our actual operational risks are (i) phonemizer artifacts
(we use espeak too, future work) and (ii) the int8-palettization
regression seen on Yinghao. Both are real but bounded.

### Trial 35 — Strategic decision: commit to StyleTTS2-ANE, deprecate legacy
**Context:** The plan in `PLAN.md` proposed a **parallel** backend —
keep legacy 4-graph alive, ship `StyleTTS2Ane` alongside, A/B
benchmark, then maybe deprecate. After Trial 29 (Yinghao collapse)
and the broader ANE economics, the user's call is to drop the
parallel-backend path and treat the 7-graph ANE re-cut as the
canonical StyleTTS2 backend going forward.
**Rationale:**
- Legacy 4-graph cannot be made ANE-resident without the re-cut
  (BiLSTM, EnumeratedShapes E5RT, attention einsum, Snake activation
  — all blockers documented in `PLAN.md`).
- Legacy can't fix the int8 collapse without per-voice fp fallback,
  which the 4-graph design doesn't allow.
- Maintaining two backends doubles the test/CI/HF-upload surface for
  no real upside once ANE lands.
- Kokoro-ANE (the architectural cousin) already proved the 7-graph
  pattern at production quality.

**Action items (going forward):**
- New work targets `Sources/FluidAudio/TTS/StyleTTS2Ane/` only.
- The voice-catalog wiring (FluidAudio#583) sits under the legacy
  `Sources/FluidAudio/TTS/StyleTTS2/` namespace today; once the ANE
  manager lands, the catalog moves with it (the catalog itself is
  backend-agnostic — it's just `id → ref_s.bin`).
- Legacy 4-graph artifacts on HF stay published until the ANE bundle
  ships; after that, mark them deprecated in the README. Don't
  delete — older app builds may still pull them.

---

## Summary of key bugs

| Bug                                       | Symptom                                   | Fix                                                           |
|-------------------------------------------|-------------------------------------------|---------------------------------------------------------------|
| BiLSTM zero-pad contamination             | Stage A cosine 0.9999 alone, 0.91 e2e     | Slice to `T_real` before calling text_predictor              |
| Cumulative-duration rounding mismatch     | F0 drift across utterance                 | Match `floor` (PyTorch) — anchor last token at last mel frame |
| Phase decorrelation in trace harness      | Sample cosine 1.0, mel cosine 0.85        | Lock single seeded RNG across e2e parity sweep                |
| diffusion_step on ANE                     | Per-step 350 ms+ with ANE↔CPU fallback     | Pin `CPU_AND_GPU`                                             |
| decoder on ANE                            | HiFi-GAN slower than CPU+GPU              | Pin `CPU_AND_GPU`                                             |
| Naive `linear_quantize_weights`           | Buzzing on vowels, log-mel 0.86           | Add `weight_threshold=200_000`                                |
| int8 on diffusion_step                    | Compounded denoiser noise, log-mel 0.91   | Don't quantize iterative stages                               |
| int8 on decoder                           | Periodic artifacts on upsample stride     | Don't quantize transposed-conv stacks                         |
| Cold-first-call ANE compile               | First RTFx 1.24×, then 4.32×              | Explicit warmup loop on app launch                            |
| GE2E noise-floor on synthetic TTS         | False high-drift signal (0.84 looks bad)  | Use ECAPA-TDNN; threshold-aware                               |
| `torchaudio.load` missing `torchcodec`    | `ModuleNotFoundError`                     | Use `soundfile` + manual resample                             |
| "Robotic" voice quality complaint         | Audibly stiff prosody                     | Architectural — PyTorch fp32 ceiling cos(REF,PT)=0.29         |
| `ref_s.bin` byte-layout naming inversion  | Swift `voice.acoustic` is actually prosody| On-disk `[ref_p, ref_s]` is canonical; Swift names are legacy |
| `06_dump_ref_s.py` import name            | `ImportError: load_modules`               | Renamed to `load_inference_modules` after lib refactor        |
| Yinghao voice collapses on CoreML         | log-mel cos 0.95 but ASR → "Yeah."        | int8 vocoder palettization regression; ship Vinay/Gavin/Nima  |
| `voices/` not fetched by HF downloader    | Walker skipped the new subdir             | Add `"voices"` to `StyleTTS2.requiredModels` (single-line fix)|
| Swift `\", \"` in `\(…)` interpolation    | `extraneous '"' in literal` build error   | Drop backslashes inside expression context: `joined(", ")`    |
| Multi-line bash continuation in tool harness | `(eval):1: permission denied:`           | Collapse to single-line `cd <dir> && ENV=val python …`        |
