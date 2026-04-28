# CosyVoice3 → CoreML: Trials and Errors

A chronological record of every non-trivial issue hit while converting
**CosyVoice3 0.5B** (Mandarin zero-shot TTS) from PyTorch to CoreML and
wiring the resulting `mlpackage` bundles through a pure-Swift runtime in
FluidAudio. Covers eleven phases, from auditing the abandoned MB-MelGAN
exploration in PR #42 through the final bit-exact parity harness.

Conventions used below:

- **Verbatim** = text copied directly from a tool output or source file.
- Numbers without a unit are (MAE, max\|Δ\|, SNR in dB, cosine sim, ms)
  depending on context.
- File paths are absolute unless otherwise noted.
- Shipping config: `LLM-Prefill-T256-M768-fp16`, `LLM-Decode-M768-fp16`,
  `Flow-N250-fp32`, `HiFT-T500-fp16`, `embeddings-runtime-fp32.safetensors`.

---

## Phase 0 — Auditing PR #42's MB-MelGAN sandbox (and deciding to throw it out)

PR #42 ("CoreML Conversion Patterns & MB-MelGAN Optimization Benchmarks")
proposed *architecturally replacing* CosyVoice3's native HiFT vocoder
with MB-MelGAN on the thesis that HiFT was too complex to convert
(claimed 705,848 ops). An audit produced a different picture.

| Claim in PR #42 | Measured reality | Outcome |
|---|---|---|
| MB-MelGAN = 202 ops | 202 is the torch-level module count; the compiled MIL graph is 300 ops | "Misleading framing" |
| HiFT vocoder = 705,848 ops | `count_ops_hift_decode.py` + `count_ops_hift.py`: HiFT decode = **1,174** ops, full HiFT = **1,380** ops — off by ~511× | Claim was fabricated |
| "3,494× op reduction" | Reduction is ~4.6× if we believe the fabricated baseline | Headline falls apart |
| "FP32 is 12.9× slower than FP16" | Cold single-iter measurement included compile+load (1,664 ms). With 5 warmups + 20 iters FP32 is only 1.45× slower | Measurement artifact |
| README "recommends RangeDim + FP32" | `train_mbmelgan.py:154` actually uses `EnumeratedShapes` + `FLOAT16`; README bottom confesses `"📋 TODO: Apply RangeDim + FP32 to train_mbmelgan.py"` | Headline contradicts shipping code |
| `load_state_dict(..., strict=False)` with `channels=384, out_channels=4, upsample_scales=[5,5,3], stacks=4` | Overlapping keys load; the rest stays random-init. Training "works" from half-init | Silent correctness bug |
| Final upsample step used `mean(dim=1)` | MB-MelGAN's 4 sub-bands should go through a pseudo-QMF synthesis filter-bank. `mean` is not PQMF | Not a real MB-MelGAN path |
| MB-MelGAN was trained on **VCTK** | CosyVoice3 Flow outputs a different mel distribution | Swapping vocoders without retraining ≠ CosyVoice3 output |
| 43 markdown "success" docs in `trials/` | `FINAL_STATUS.md`, `COMPLETE_STATUS.md`, `FINAL_RESOLUTION.md`, `SUCCESS.md`, `MBMELGAN_SUCCESS.md`, `DECODER_COMPRESSION_SUCCESS.md`, `LAYERNORM_FIX_SUCCESS.md`, `DEPLOYMENT_READY.md`, … | Classic "emit new success doc per step" pattern. Deleted. |
| FARGAN alternate explored | `trials/FARGAN_ANALYSIS.md:59`: *"Estimated operations: Still 10k-50k (better than 705k, but not guaranteed to convert)"* | Abandoned |

**Decision**: drop the MB-MelGAN / FARGAN vocoder-replacement thesis.
Convert the real HiFT vocoder directly and fix the actual blockers
(`torch.istft`, `broadcast_tensors`) — not the imaginary op-count one.

Resolution: commit `1bb5e3e feat(tts/cosyvoice3): pivot from MB-MelGAN
sandbox to direct CoreML pipeline` — 117 files, +7351/−15292. Local
residue (`mbmelgan_*/`, `fargan_source/`, etc.) added to `.gitignore`.

---

## Phase 1 — HiFT vocoder (direct conversion)

### Op-not-implemented chain

Early traces hit a string of `NotImplementedError`s from `coremltools.converters.mil`:

**Verbatim**:
- `NotImplementedError: PyTorch convert function for op 'prod' not implemented.`
- `NotImplementedError: Converter is not implemented (OperationDescription(domain='', operation_type='GreaterOrEqual', version=16))`
- `NotImplementedError: PyTorch convert function for op 'greater_equal' not implemented.`
- `Error: _cast op not supported in coremltools`

Each was worked around by rewriting the offending expression in the
wrapper (avoid implicit casts; replace `prod` with explicit `mul`;
replace `>=` with `>` + `!=`).

### Real HiFT blockers: `torch.istft` and `broadcast_tensors`

The upstream HiFT graph ends with `torch.istft` (no MIL equivalent) and
has `broadcast_tensors` calls mid-graph. Fixed by:

- Re-implementing the final synthesis without `torch.istft` (direct
  overlap-add via explicit window/gather, see `src/stft_coreml.py`).
- Removing `broadcast_tensors` by broadcasting via plain arithmetic.

These were the **actual** CoreML blockers for HiFT — not op count.

### Sinegen FP32 `sin()` drift

**Trigger**: `torch.cumsum(...) * upsample_scale` produces large phase
arguments. CoreML's vecLib FP32 `sin` diverges from PyTorch/glibc on
large arguments.

**Fix** (captured in source): wrap phase modulo 2π before `sin`. Comment
in `src/sinegen_coreml.py`:

> *"the argument to sin() strictly in [0, 2pi), avoiding CoreML FP32 precision"*

**Residual**: ~1% correlation drop in the last 10% of generated audio
(envelope correct, phase misaligned at the tail). ASR round-trip
through Whisper: identical transcripts → shipped anyway.

### F0 predictor FP64 → FP32 downgrade

Upstream `ConvRNNF0Predictor` uses FP64 (its GRU path carries state in
FP64). ANE is FP32-only.

Comment in source: *"f0_predictor kept in FP32 (upstream uses FP64 for
precision; ANE is FP32-only)"*.

| Check | MAE | corr |
|---|---|---|
| Wrapper parity (FP64 reference) | 3.5e-6 | 1.0 |
| With FP64 → FP32 downgrade | 3.9e-4 | 0.99997 |

Accepted.

### HiFT fp16 vs fp32

Direct parity on synthetic mel: fp16 = **71% waveform correlation** vs
fp32, same magnitude range. Audibly fine; ASR round-trip clean.

**Shipping**: `HiFT-T500-fp16`.

### HiFT cold compile time

First `coreml-cli` load (cold ANE compile): **118,313 ms**.

---

## Phase 2 — LLM (Qwen2) export

### Dtype mismatch — BFloat16 checkpoint vs FP32 runtime

**Symptom**: `cv = CosyVoice3(...)` loads the Qwen2 base in bfloat16;
Swift activations arrive as fp32 → silent downcast through `@` (matmul)
produced garbled logits.

**Fix**: `cv.model.llm.float()` after load. Captured finding:

> *"LLM dtype fix is effective — `cv.model.llm.float()` resolved the
> BFloat16/FP32 mismatch; LLM generation ran to completion."*

Added comment in `convert-llm.py`: *"Qwen2 base is stored in BFloat16;
force FP32 on CPU to avoid dtype mismatch."*

### Upstream `Qwen2Encoder.forward_one_step` is broken under transformers ≥ 5.x

**Root cause** (captured verbatim):

> *"Upstream CosyVoice3's Qwen2Encoder.forward_one_step is incorrect
> under transformers 5.5.4 (written for 4.40.1) — it passes a length-1
> attention_mask that causes the HF model to mask out past tokens."*

**Impact**:

> *"Our wrapper is correct against HF's intended full-context path and
> probably explains the garbled audio in earlier TTS→ASR roundtrip tests."*

**Fix**: wrote `src/llm_coreml.py` — a full re-implementation of Qwen2's
forward path:

- `Qwen2Prefill` / `Qwen2Decode` — two separate mlpackages
- `Qwen2AttnPrefill` / `Qwen2AttnDecode` — with static KV-cache slicing
- `Qwen2LayerPrefill` / `Qwen2LayerDecode`
- `Qwen2MLPReimpl`, `Qwen2RMSNormReimpl`

Parity vs HF: **MAE ~1e-6** over prefill + 20 decode steps.

### Qwen2Config attribute error

**Verbatim**: `AttributeError: 'Qwen2Config' object has no attribute 'rope_theta'`

Newer configs use `rope_parameters`. Fix: probe both:

```python
rope_theta = getattr(cfg, "rope_theta", None) \
           or cfg.rope_parameters["base"]
```

### Static-shape KV cache design

Ultimately chose two-mlpackage prefill/decode over a single bucketed
model. Rationale: prefill is compute-bound over long context, decode is
memory-bound over single-step — different optimal compute-unit
assignment on ANE.

- `LLM_MAX_LEN = 768` (M)
- Prefill: `T_pre = 256` (fixed)
- Decode: step size 1
- KV cache: **`[24, 1, 2, 768, 64]` fp16**
  = `(layers, batch, num_kv_heads, max_ctx, head_dim)`
- Qwen2-0.5B uses GQA with **2 KV heads**, 14 Q heads, head_dim 64
- Prefill fills `[0..T_pre-1]`; decode zero-pads past `input_len`
  automatically on each step via `cur_len` scalar input

### fp16-safe mask sentinel

Initial conversion used `torch.tensor(float('-inf'))` for masked
attention slots → overflows to NaN in fp16 softmax.

**Fix**: `neg_inf = torch.tensor(-1e4, dtype=torch.float32)  # fp16-safe`.

### fp16 RMSNorm overflow

First blind fp16 conversion of Qwen2:

> *"Fp16 has a known Qwen2 RMSNorm overflow."*

**Fix**: selective-precision helper `_make_precision(fp16: bool)` in
`convert-llm.py`:

```python
FP32_OPS = {"pow", "reduce_mean", "rsqrt", "softmax"}
return ct.transform.FP16ComputePrecision(
    op_selector=lambda op: op.op_type not in FP32_OPS)
```

Captured comment:

> *"FP16 everywhere except RMSNorm ops (pow / reduce_mean / rsqrt) which
> need fp32 to avoid overflow on Qwen2's activation outliers. Softmax is
> also kept in fp32 since fp16 softmax over long keys can underflow."*

Shipping: **LLM-Prefill-T256-M768-fp16** and **LLM-Decode-M768-fp16**
with selective-FP32 op pinning.

---

## Phase 3 — Flow DiT (CFM, 22 DiT blocks × 2 CFG × 22 CFM steps)

### fp16 = catastrophic NaN

Parity harness output verbatim:

```
MAE=nan  max=nan
MAE=nan  max=nan  corr=nan
```

**Verdict** (captured):

> *"Flow fp16 produces NaN on the parity inputs — catastrophic numerical
> blowup, not mild drift. Keeping pow/reduce_mean/rsqrt/softmax in fp32
> was insufficient for the DiT graph."*

Upstream commented this directly in its own source:

> *"NOTE when flow run in amp mode, x.dtype is float32, which cause nan
> in trt fp16 inference, so set dtype=spks.dtype"*

### Attempt 1 — pin `pow/reduce_mean/rsqrt/softmax` to fp32 (same as LLM)

Result: still NaN. MIL graph inspection revealed the culprit — a fused
`layer_norm` op with fp16 I/O, untouched by the scalar-level pinning:

```
%var_50731_cast_fp16: (2, 500, 1024, fp16)(Tensor) = layer_norm(
    x=%input_4231_cast_fp16,
    axes=[-1],
    epsilon=1.0132789611816406e-06,
    name="op_50731_cast_fp16")
```

`layer_norm` is a **single fused MIL op** — its scalar subgraph isn't
decomposed, so the op_type-based pinning lambda can't reach into it.

### Attempt 2 (fp16v2) — add `layer_norm` and `gelu` to `FP32_OPS`

MIL now correctly wraps each layer_norm with fp16→fp32 → layer_norm →
fp32→fp16 casts:

```
%var_50731: (2, 500, 1024, fp32)(Tensor) = layer_norm(
    x=%input_4231_cast_fp16_to_fp32,
    axes=[-1],
    epsilon=1e-06,
    name="op_50731")
```

Parity result: **still NaN**.

End-to-end comparison:

| Config | Parity | Latency (CPU) | ASR roundtrip |
|---|---|---|---|
| fp16 v1 (pow/rsqrt/softmax fp32) | NaN | 11.7 s | empty |
| fp16 v2 (+ layer_norm + gelu fp32) | **NaN** | **20.4 s** (1.7× slower) | **empty** |

**Final verdict** (captured):

> *"Clear verdict on Flow: fp16 is unshippable (NaN on parity inputs,
> silent-on-ASR on real inputs). To make Flow fp16 work we'd need to
> pin layer_norm (fused op) to fp32 too — but layer_norm appears 500+
> times in the DiT graph, which mostly negates the ANE benefit."*

**Shipping**: `Flow-N250-fp32` (fp16 abandoned). Canonical comment in
`CosyVoice3Models.swift`:

```
/// - Flow-N250-fp32 (fp16 causes NaN; fused 'layer_norm' cannot be pinned)
```

### Compute-unit quirk

Python parity path uses `ct.models.MLModel(..., compute_units=CPU_ONLY)` —
Flow had **never been run on ANE/GPU or compiled to `.mlmodelc`** when
the parity numbers were taken.

Swift-side: first loaded with `.cpuAndGPU`. Parity MAE: 0.810 → 0.809.
Not the root cause — the real culprit was an MLMultiArray stride bug
(Phase 5).

**Shipping**: Flow Python side = `CPU_ONLY`; Swift Flow = `.cpuAndGPU`
(not ANE — GPU compensates for ANE-blocked fp32 throughput).

### End-to-end matrix

| Config | parity | e2e | ASR |
|---|---|---|---|
| Flow fp32 + HiFT fp32 | exact | ~50 s (CPU) | correct |
| **Flow fp32 + HiFT fp16** (shipping) | corr 1.0 | passes | correct |
| Flow fp16v1 + HiFT fp16 | NaN | 11.7 s | empty |
| Flow fp16v2 + HiFT fp16 | NaN | 20.4 s | empty |
| all fp16 | NaN | corr 0.71 | empty |

Related commits: `b31d1e3` (fp16 NaN captured), `baf565d`
(fp16v2 with layer_norm+gelu pinned).

---

## Phase 4 — CAMPPlus + SpeechTokenizerV3 (Python-side)

Both stay Python-side in the shipping zero-shot path; neither is wired
into Swift. Per-voice assets (`llm_prompt_speech_ids`, `prompt_mel`,
`spk_embedding`) are precomputed and shipped inside a per-voice
`shipping.safetensors` bundle.

From a session finding:

> *"To make a new voice from a fresh reference WAV you'd run a Python
> script (SpeechTokenizer + 24 kHz mel DSP + CAMPPlus), and ship the
> resulting bundle. At runtime Swift just loads the bundle — no Python."*

### CAMPPlus parity

- `verify/` harness converts via `onnx2torch` path, then `coremltools`
  with `_make_precision` pinning BatchNorm-style ops to fp32.
- Upstream drift vs `onnxruntime`: **cos 0.96** (documented in
  REPORT.md; inherent to the upstream checkpoint's ONNX export).

### SpeechTokenizerV3

- `speech_tokenizer_v3.onnx` converts cleanly (Conformer).
- fp16 `_make_precision` staged with extra pinning:
  > *"FP16 everywhere except numerically-sensitive ops … plus
  > `reduce_sum`/`sub` for SpeechTok"*
- Observed drift: `mismatches (valid): 44/87` discrete tokens on real
  audio (treated as tolerable; runs once per voice).

---

## Phase 5 — Swift parity harness (Phase 1 port)

Goal: bit-exact (or int16-floor-exact) WAV match between the Swift
runtime and the Python reference.

### Executable-name error

**Verbatim**: `no executable product named 'fluidaudio'` →
actual name is **`fluidaudiocli`**. All invocations corrected to
`swift run fluidaudiocli …`.

### `MLModel(contentsOf:)` rejects `.mlpackage`

**Verbatim**: `"Unable to load model... Compile the model with Xcode or
MLModel.compileModel(at:)"`

**Fix**: `compileAndLoad` helper in `CosyVoice3ModelStore.swift` —
calls `MLModel.compileModel(at:)` on first use and caches the resulting
`.mlmodelc` next to the `.mlpackage`.

### Swift `withUnsafeBytes(of:_:)` shadowing

Inside `extension Data`, the free function `Swift.withUnsafeBytes` is
shadowed by `Data`'s instance method (same name, different signature).

**Fix**: fully qualify → `Swift.withUnsafeBytes(of: &le) { ... }` inside
`SafetensorsReader.swift`.

### Initial parity divergence — *not* the compute-unit quirk

- Round 1: **MAE 0.056, max\|Δ\| 0.81, SNR −0.11 dB** — uncorrelated.
- Round 2 (after `.cpuAndGPU` adjustment): MAE 0.056 → 0.056, max\|Δ\|
  0.810 → 0.809. **Not** the root cause.

### Root cause — MLMultiArray stride padding

`[1, 80, 500]` fp32 Flow mel output had physical strides
`[40960, 512, 1]` — CoreML pads the time dimension **500 → 512** (likely
64-byte / SIMD alignment). Raw linear reads over `dataPointer` hit
padding bytes → contaminated waveform.

**Fix**: stride-aware accessors everywhere an `MLMultiArray` is
consumed. Updated:

- `CosyVoice3Synthesizer.runHiFT`
- `CosyVoice3Synthesizer.runDecode`
- `CosyVoice3Synthesizer.sliceLastStepLogits`
- `CosyVoice3Synthesizer.runPrefill` (embed slicing)
- `CosyVoice3Synthesizer.runFlow` (input assembly)
- `CosyVoice3SpeechEmbeddings.embedding()` (fp16 → fp32 row copy)

Result after stride fix: **MAE 0.011, max\|Δ\| 0.52, SNR 9.37 dB**.

### Residual 0.52 drift — fixture staleness

Python determinism confirmed:

- Flow max\|Δ\| run-to-run = **0.0**
- HiFT max\|Δ\| run-to-run = **0.0**

Conclusion: `e2e_shipping.wav` had been written from a different
`shipping.safetensors` state than the one loaded at parity time.

**Fix**: regenerate both in the same Python run.

Final parity: **MAE 7e-6, max\|Δ\| 3e-5 (= 1/32768, the int16 WAV
quantization floor), SNR 78.08 dB**.

### BNNS teardown error (cosmetic)

**Verbatim**:

```
BNNS Graph Compile: failed to preallocate file...
No space left on device for path: .../e5rt.e5bundlecache/...
```

Thrown at process exit after fixtures were already written. Does not
affect parity numbers. Clean the bundle cache or ignore.

---

## Phase 6 — Swift frontend parity (Phase 2)

### 2.4e-4 text-row divergence

Initial parity: MAE 6.8e-6, max\|Δ\| **2.4e-4** (tolerance was 1e-4).

Per-row diagnostic: worst rows were *text* rows (e.g., t=4, 19, 20, 21,
35), all with uniform ~2.4e-4 magnitude. `sos`, `task_id`, and
speech-token rows were bit-exact.

Compared runtime `model.embed_tokens.weight` vs raw `llm.pt` slice:

- `max|runtime_text − pt_text| = 0.000474`
- `max|runtime_speech − pt_speech| = 0.0`

**Root cause** (captured):

> *"HuggingFace's Qwen2 load path narrows `embed_tokens.weight` through
> a lower-precision dtype; `.float()` widens back with zeroed mantissa
> bits. The raw `llm.pt` fp32 values differ from the runtime values by
> up to 4.7e-4. CosyVoice3's own `speech_embedding` is a custom module
> that stays fp32 throughout."*

**Fix**: ship the **post-`.float()`** runtime weight.
`verify/export_runtime_embeddings.py` dumps it →
`embeddings-runtime-fp32.safetensors` (542 MB).

Final parity: **MAE 0, max\|Δ\| 0**.

### Qwen2 BPE tokenizer port

Existing Parakeet `BpeTokenizer.swift` is SentencePiece-style — unusable
for Qwen2, which is GPT-2-style byte-level BPE (slow path via
`AutoTokenizer.from_pretrained(cosyvoice3_dl/CosyVoice-BlankEN)`). Wrote
a fresh port.

Key mechanics:

- Qwen2 pretokenize regex (verified against HF source):
  ```
  (?i:'s|'t|'re|'ve|'m|'ll|'d)|
  [^\r\n\p{L}\p{N}]?\p{L}+|
  \p{N}|
   ?[^\s\p{L}\p{N}]+[\r\n]*|
  \s*[\r\n]+|
  \s+(?!\S)|
  \s+
  ```
- GPT-2 `bytes_to_unicode`: 188 printable bytes → themselves; 68
  unprintables → code points 256..323. See
  `Qwen2ByteEncoder.swift`.
- Vocab = **151,936 + 281 CosyVoice3 specials** (IDs 151643..151923).
- `<|endofprompt|>` = **151646** is mandatory — CosyVoice3's LLM
  `asserts` the prompt contains it.
- Qwen2 has **no `<unk>`**. Unmappable pieces are silently dropped:
  > *"Unknown token: Qwen2 has no &lt;unk&gt;. Drop silently as …"*
- Parity: **25/25 cases pass** — ASCII, Mandarin, mixed, phoneme tags,
  4-byte UTF-8 emoji, whitespace edges, empty string.

Files:
- `Sources/FluidAudio/TTS/CosyVoice3/Pipeline/Preprocess/Qwen2ByteEncoder.swift`
- `Sources/FluidAudio/TTS/CosyVoice3/Pipeline/Preprocess/Qwen2BpeTokenizer.swift`

### Constant-name typos

Referenced `CosyVoice3Constants.sos` and `taskId as Int32(…)`. Actual
names are **`sosId`** and **`taskId`** (already typed `Int32`). Fix:
use correct spellings.

### Chinese TN — `testNormalizeEndToEnd` failure

Initial order ran `replaceBlank` **before** ASCII→CJK substitutions.
Net effect: a space between ASCII `.` and `2` stayed ASCII-ASCII →
preserved. After `.` → `。` and digit spellout, the space became
visible as `。 二零二四…`.

**Fix — final pipeline order**:

1. strip newlines
2. trim leading/trailing whitespace
3. `replaceCornerMark` (`²` → `平方`, `³` → `立方`)
4. `spellOutDigitsZh` (`0..9` → `零一二…`)
5. `.` → `。`
6. ` - ` → `，`
7. `replaceBlank` (drop CJK-interior spaces)
8. `removeBracket` (`（）【】` stripped, `——` → space)
9. `stripTrailingCommaLikes` (trailing `[，,、]+` → `。`)

End-to-end assertion:

```swift
XCTAssertEqual(
    CosyVoice3ChineseNormalizer.normalize("希望你以后能够做的比我还好用. 2024年,,"),
    "希望你以后能够做的比我还好用。二零二四年。")
```

All 8 TN tests pass.

### Shell escape quirk

`if sr \!= 24000:` failed under zsh (backslash→literal). Wrote the
script to `/tmp/gen_mel_ref.py` and invoked with `uv run python`.

---

## Phase 7 — RAS sampler (Swift port)

Reference: `cosyvoice.utils.common.ras_sampling(weighted_scores,
decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1)`
(VALL-E 2 style).

Algorithm re-stated in `CosyVoice3RasSampler.swift`:

1. softmax over full vocab
2. stable sort descending
3. greedy accumulate until `cum_p ≥ top_p` **OR** `count == top_k`
4. multinomial over the retained set
5. repetition mask: if the count of `top_id` in the last `win_size`
   decoded tokens is `≥ win_size * tau_r` (= 1 token in last 10),
   mask that id and redraw a full multinomial over `softmax(masked)`

For parity the Swift harness uses **seeded token replay** via
`seedTokens([Int32])` — bypasses `torch.multinomial` RNG divergence
between Python and Swift. End-to-end parity tests use this replay mode.

---

## Phase 8 — Mel DSP and frontend assembly (Swift port)

24 kHz mel config (frozen, validated bit-exact vs `matcha.utils.audio.mel_spectrogram`):

| Parameter | Value |
|---|---|
| sample rate | 24,000 |
| n_fft | 1,920 |
| hop_size | 480 |
| win_size | 1,920 (periodic Hann) |
| n_mels | 80 |
| fmin / fmax | 0 / 12,000 |
| pad | reflect, length 720 = (n_fft − hop)/2 |
| center | False |
| magnitude | `sqrt(r² + i² + 1e-9)` |
| log floor | `1e-5` |
| mel scale | Slaney (linear below 1000 Hz) |

Dedicated class `CosyVoice3PromptMel.swift` rather than extending
`AudioMelSpectrogram` because the CosyVoice3 spec diverges in three
independent places (pad length, pad mode, magnitude-vs-power
spectrum) — flag-bloating the shared NeMo-tuned class would couple
unrelated backends.

8 sanity tests (`CosyVoice3PromptMelTests.swift`): frame count, zero
clamp → log(1e-5), 200 Hz sine argmax in bottom 20 bins, reflect pad
`[1,2,3,4,5]` pad=2 → `[3,2,1,2,3,4,5,4,3]`, Hann length-4
`[0, 0.5, 1.0, 0.5]`, mel basis shape + non-zero integrals,
token-ratio trim, throws-if-too-short.

Shipping sizes (frozen):

- LLM-Prefill: `T_pre = 256`
- LLM-Decode: `M = 768` max context
- Flow: `N = 250` speech tokens → `M = 500` mel frames
- HiFT: `T = 500` mel frames → 10 s @ 24 kHz

---

## Phase 9 — Swift `CosyVoice3TtsManager` integration

### Actor-isolation errors

`modelsDirectory()` couldn't access an actor property synchronously from
init-time code.

**Fix**: `public nonisolated let directory: URL` on `ModelStore`,
re-exposed as `public nonisolated var modelsDirectory: URL` on the
manager.

### `DownloadUtils.downloadRepo` rejecting `speechEmbeddings`

`ModelNames.CosyVoice3.speechEmbeddings` was listed in `requiredModels`,
but physically lives at `embeddings/speech_embedding-fp16.safetensors`,
not at repo root → repo-wide download validator rejected it.

**Fix**: moved `speechEmbeddings` to a new nested `Sidecar` enum and
fetched via `AssetDownloader.ensure(.file(...))` instead of
`DownloadUtils.downloadRepo`.

### `getRequiredModelNames` exhaustiveness error

Adding `Repo.cosyvoice3` broke the exhaustive switch in
`getRequiredModelNames`.

**Fix**:

```swift
case .cosyvoice3: return ModelNames.CosyVoice3.requiredModels
```

### Voice-bundle extraction typo

`args.voice-id if False else args.voice_id` — leftover dead branch.

**Fix**: `args.voice_id` (argparse converts `--voice-id` → `voice_id`).

### "Modified during build" flake

Single occurrence on first rebuild of a freshly-renamed file. `swift
build` retry succeeded. No reproducer.

---

## Phase 10 — HuggingFace upload staging

Initial plan: symlink farm under `build/upload/` pointing at the
per-model directories. Dir symlinks appeared empty through every tool
that touched them.

User report (verbatim):

> *"the mlpackage and mlmodelc are sketeon empty wtf"*

`huggingface-cli upload` silently skipped them; Finder showed empty
folders; `tar` streamed zero bytes.

**Fix**: replaced every symlink with a real copy (`cp -R` for
`.mlpackage`/`.mlmodelc`, `cp` for safetensors). Verified
`find build/upload -type l` returned empty before upload.

Final staged size: **5.8 GB**. Uploaded as
[`FluidInference/CosyVoice3-0.5B-coreml`](https://huggingface.co/FluidInference/CosyVoice3-0.5B-coreml).

---

## Phase 11 — ANE profiling (blocked)

All four CosyVoice3 mlpackages / mlmodelcs fail
`MLComputePlan.loadContentsOfURL`:

**Verbatim**:

- `"Failed to load compute plan: unknown error"` (both `.mlmodelc` and
  `.mlpackage` paths)
- Python 3.14 `.mlpackage` path also reports
  `"Unable to load libmodelpackage"`

Control: `silero-vad-unified-v6.0.0.mlmodelc` profiles cleanly on the
same machine → the tool works. The CosyVoice3 graphs (stateful KV-cache
ops, fused `layer_norm` at scale, LLM-Prefill ~700 MB, Flow 1.2 GB)
trip it.

HiFT cold compile timing captured: **118,313 ms**.

Documented in `REPORT.md`. ANE residency numbers need **Instruments**
or a custom Swift harness with `MLPredictionOptions.usesCPUOnly`
toggling — follow-up work.

Cosmetic `coremltools` import noise, seen on every run:

> *"scikit-learn version 1.7.2 is not supported. Minimum required
> version: 0.17. Maximum required version: 1.5.1. Disabling
> scikit-learn conversion API."*

None of the CosyVoice3 converters touch sklearn.

---

## Shipping configuration (frozen, end-to-end validated)

| Component | Artifact | Precision | Compute | Rationale |
|---|---|---|---|---|
| LLM-Prefill (T=256) | `LLM-Prefill-T256-M768-fp16.mlpackage` | fp16 w/ fp32 pin on `{pow, reduce_mean, rsqrt, softmax}` | ANE | Selective pin avoids Qwen2 RMSNorm overflow; softmax fp32 avoids long-key underflow |
| LLM-Decode (M=768) | `LLM-Decode-M768-fp16.mlpackage` | fp16 w/ same pinning | ANE | Same as prefill |
| Flow (N=250) | `Flow-N250-fp32.mlpackage` | fp32 | CPU (Py) / CPU+GPU (Swift) | fp16 NaNs on parity inputs; fused `layer_norm` can't be pinned without pinning 500+ ops |
| HiFT (T=500) | `HiFT-T500-fp16.mlpackage` | fp16 (f0_predictor FP32, sinegen phase mod-2π) | ANE | fp16 waveform corr ≈ 0.71 is audibly fine; ASR roundtrip clean |
| Embedding table | `embeddings-runtime-fp32.safetensors` | fp32 | mmap | Must use *post-`.float()`* runtime values, not raw `llm.pt` slice |
| Speech embedding | `speech_embedding-fp16.safetensors` | fp16 | mmap | 6761×896 custom module, stays bit-exact |
| CAMPPlus, SpeechTokenizerV3 | Python-only | fp32 ORT | — | Runs once per voice; output shipped in `shipping.safetensors` |

End-to-end Swift-vs-Python parity: **MAE 7e-6, max\|Δ\| 3e-5, SNR 78.08 dB**
(= int16 quantization floor).

---

## Residual caveats

1. **Flow can't reach ANE** under current coremltools without pinning
   500+ fused `layer_norm`s to fp32 — at which point the ANE throughput
   advantage evaporates.
2. **HiFT sinegen has ~1% tail-phase drift** (CoreML FP32 `sin` vs
   glibc `sin` on large arguments). Inaudible; Whisper ASR unchanged.
3. **f0_predictor FP64→FP32 downgrade** introduces MAE 3.9e-4 (corr
   0.99997). Required for ANE.
4. **Runtime `embed_tokens.weight` ≠ raw `llm.pt`** by up to 4.7e-4 due
   to HF bf16-narrow + `.float()`-widen round-trip. Ship the
   post-`.float()` weight.
5. **MLMultiArray stride padding** — CoreML pads tensor dims (time
   500→512 for alignment). Every Swift read goes through stride-aware
   iteration; never `memcpy` over the raw pointer.
6. **ANE residency profiling blocked** on all four CosyVoice3
   mlpackages by a `coreml-cli` / `MLComputePlan` limitation. Use
   Instruments.
7. **Per-voice assets require Python** (SpeechTokenizerV3 ONNX +
   24 kHz mel + CAMPPlus ONNX). Fresh-WAV-to-voice loop is not yet pure
   Swift.
8. **Upstream `Qwen2Encoder.forward_one_step` is broken** under
   transformers ≥ 5.x. Swift path routes around via our
   `Qwen2Prefill`/`Qwen2Decode` re-implementation.

---

## Flow ANE port: `conv_pos_embed` CPU island (residual after Stage 3)

After the BC1S rewrite landed (ANE ≥ 80%, Flow ≤ 10 s), `coreml-cli
--fallback` showed a single remaining CPU island of **77 ops** in
`input_embed.conv_pos_embed`: host `CausalConvPositionEmbedding` =
`Conv1d(1024, 1024, kernel_size=31, groups=16)` + Mish × 2 with
asymmetric causal `F.pad(x, (k-1, 0))`. Trace-unrolled across 10 Euler
steps → 20 convs + 10 softplus + 10 tanh + 11 mul + 10 pad + 5 cast +
assorted reshapes.

Three decomposition attempts tried to move this island onto ANE. All
failed; the CPU island is **intrinsic to the `k=31 × dim=1024` Conv1d
footprint**, not to `groups > 1` or Mish or pad style.

### Option A — 16-way `groups=1` split (math-equivalent)

Replace `Conv1d(1024, 1024, k=31, groups=16)` with 16 parallel
`Conv1d(64, 64, k=31, groups=1)` operating on channel slices, concat
their outputs. Pad style switched to `torch.cat([zeros, x])` (concat
prefix) instead of `F.pad(x, (30, 0))` to stay ANE-native. Mish preserved
→ bit-exact weights.

- fp32 parity at all lengths (S ∈ {32, 64, 125, 250}): MAE = 0.
- Full-DiT fp32 cumulative MAE: **2.907e-05** (passes <1e-3 gate).
- `ANECCompile() FAILED (11)` after 900 s probe timeout.
- Bisection (revert concat → F.pad): still FAILED (11) → rules out pad.

### Option C — 16-way split + Mish → SiLU (NOT math-equivalent)

Same 16-way slice/conv/concat scaffold as Option A, but swap `nn.Mish()`
→ `nn.SiLU()`. SiLU is a single ANE-native op; Mish decomposes to
`softplus → tanh → mul` (3 ops). Hypothesis: Mish op count was the
tiling blocker.

- `ANECCompile() FAILED (11)`.
- **Rules out Mish** as the cause: Option A and Option C differ only in
  activation; both failed identically.

### Option L — Dense `groups=1` Conv1d with block-diagonal weight

Expand the host grouped conv into a single `Conv1d(1024, 1024, k=31,
groups=1)` whose weight is zero outside the 16 diagonal `(64, 64, 31)`
blocks. Math-equivalent (matmul entries for off-diagonal blocks are zero
→ contribute nothing), ANE-native op topology (no split/concat).

Cost: 16× FLOPs per call, +129 MB fp16 weight for the two convs.

- fp32 parity: bit-exact (MAE = 0).
- **ANE compile SUCCEEDED** (241.5 s, vs `FAILED (11)` on A/C).
- Device plan: 77 CPU ops **unchanged** — the dense conv is still
  evicted to CPU by ANEF.
- Wall-time regressed: p50 **904 ms → 1035 ms** (16× FLOPs on CPU, not
  ANE). Load time +180 s due to +129 MB weight.

### Finding

ANEF's rejection of the host conv is **not** about `groups > 1`. A
dense `groups=1` Conv1d with the same `(1024 × 1024 × 31)` footprint is
also evicted. The tile limit is the `kernel_size × in_channels` weight
bandwidth, which is independent of group count. Mish is unrelated
(SiLU variant also failed to compile in the 16-split case for a
different reason — ANEF couldn't tile the 10×-replicated
slice/conv/concat pattern regardless of activation).

### Hoist-out-of-loop is infeasible

`InputEmbedding.forward` does
`x = proj(cat(x, cond, text_embed)); x = conv_pos_embed(x) + x`.
The noisy `x` is step-varying → the projected tensor varies per step →
`conv_pos_embed` input varies per step. Cannot precompute once.

### Residual options (not pursued in this session)

- Swap `k=31` for a smaller kernel (k=7, k=15): ANE-friendly but
  changes receptive field → audio quality regression unknown; needs
  distillation or empirical mel-MAE check.
- Cascade k=31 into smaller convs (e.g., 3× k=11 stride-1): receptive
  field preserved, approximation not math-equivalent → weight
  distillation or retraining required.

Baseline shipped as-is: Flow p50 ≈ 904 ms warm on ANE with `77 CPU /
12878 ANE / 22223 const` op distribution. `src/conv_pos_ane.py` is
retained for reference (Option L implementation); the swap is disabled
in `src/flow_coreml_ane.py` per the comment above `load_state_dict`.

---

## Stage 4: Swift integration — ATTEMPTED, REVERTED

**Outcome: ANE port failed the audio-intelligibility gate; shipped the
prior cpuAndGPU Flow instead.**

Wired the ANE-port Flow into `FluidAudio`. Artifact swap + Swift-side
dtype fix + measurement — all mechanically clean. Then discovered the
ANE port is numerically broken end-to-end.

### Artifacts updated

- `build/upload/cosyvoice3-coreml/Flow-N250-fp16.{mlpackage,mlmodelc}`
  replaced with ANE-port build from `build/flow-ane-fp16-n250/`.
- `build/upload/cosyvoice3-coreml/manifest.json` — Flow block:
  `compute_units: cpuAndNeuralEngine`, updated purpose text,
  `size_bytes: 670101045`.
- `build/upload/cosyvoice3-coreml/README.md` — Flow row moved from
  "CPU + GPU" → "CPU + ANE" with the 3× speedup description.

### Swift changes

- `FluidAudio/Sources/FluidAudio/TTS/CosyVoice3/Assets/CosyVoice3ModelStore.swift`:
  `flowConfig.computeUnits = .cpuAndGPU` → `.cpuAndNeuralEngine`.
  Docstring updated to reflect the ANE port; notes the residual 77-op
  `conv_pos_embed` CPU island.
- `FluidAudio/Sources/FluidAudio/TTS/CosyVoice3/CosyVoice3Constants.swift`:
  header comment block updated (Flow row: `cpuAndGPU` → `cpuAndNE`,
  dropped the "fused LayerNorm → NaN" caveat).
- `FluidAudio/Sources/FluidAudio/TTS/CosyVoice3/Pipeline/Synthesize/CosyVoice3Synthesizer.swift`
  `runHiFT()`: branch on `fullMel.dataType`. The ANE-port Flow emits
  fp16 mel (graph stays fp16 end-to-end); the prior cpuAndGPU Flow
  emitted fp32. HiFT input is fp32 either way. Without this branch the
  fp16 output got reinterpreted as fp32 via `bindMemory(to: Float.self)`
  and walked off the buffer end → SIGSEGV in the first E2E run.

### Measured on M-series (Swift CLI, N_new=87, N_prompt=87)

| Stage   | Before (cpuAndGPU Flow) | After (cpuAndNE ANE Flow) |
|---------|-------------------------|----------------------------|
| prefill | ~0.9 s                  | 0.94 s                     |
| decode  | ~2.0 s (87 steps)       | 2.00 s                     |
| flow    | **~6.9 s**              | **1.80 s** (~3.8× faster)  |
| hift    | ~0.9 s                  | 0.87 s                     |
| total   | ~10.7 s                 | **5.63 s**                 |
| RTFx    | ~0.33×                  | **0.62×**                  |

Swift Flow speedup (3.8×) exceeds the Python bench (3.1×; 6.9 → 2.2 s)
because Swift's prior baseline also paid per-call MLMultiArray binding
overhead that the tighter ANE path elides.

### Parity vs obsolete `e2e_shipping.wav`

Swift MAE vs `build/wavs/e2e_shipping.wav` is 0.052 (reference >> 1e-3
gate). This reference was generated with the prior cpuAndGPU fp16 Flow.
The ANE-port rewrite is intentionally not bit-identical (BC1S layout,
`Linear → Conv2d(1×1)`, `LayerNorm` on axis=1, manual SDPA, pre-baked
rotary sin/cos). Regenerating the reference via the Python E2E path
pointing at the ANE Flow mlpackage is the proper parity target and is
deferred as a follow-up. Audio shape (83520 samples @ 24 kHz = 3.48 s)
and no-NaN both confirmed.

### Runtime log note

`E5RT encountered an STL exception. msg = MILCompilerForANE error:
failed to compile ANE model using ANEF` appears at first Flow
invocation. This is the ANE runtime attempting JIT compilation for the
77-op `conv_pos_embed` sub-graph and falling back to CPU — same CPU
island documented above. Flow wall time confirms the remainder of the
graph runs on ANE (1.8 s matches the Python ANE bench p50, not the
~6.9 s GPU baseline).

### Swift unit tests

`swift test -c debug --filter CosyVoice3`: 16 / 16 passed
(`CosyVoice3ChineseNormalizerTests`, `CosyVoice3PromptMelTests`).

### Audio intelligibility check — **FAILED**

After the first E2E run succeeded (no crash, no NaN logged, Flow 1.8 s,
audio saved), an amplitude check showed the output was **44× quieter**
than the shipping baseline:

```
shipping ref wav : peak 0.815, mean |x| 0.052 (reads
                   "希望你以后能够做得比我还好哟" via CTC-ZH + Qwen3)
ANE Flow wav     : peak 0.019, mean |x| 0.003 (both CTC-ZH and
                   Qwen3 ASR return empty / only "。")
```

Rerunning the Python E2E harness with `--flow-precision ane
--compute-units CPU_AND_NE` produced the same failure mode:

```
audio peak 0.021, whisper transcript: ""  (vs expected
 "希望你以后能够做的比我还好用")
```

So the defect is in the ANE Flow itself, not in the Swift dtype
branching.

### Root cause: mel dynamic range collapse, not NaN

Direct Flow-output inspection on the parity fixture:

| Path (compute unit)                | mel range       | MAE vs fp32 ref | NaN |
|------------------------------------|-----------------|-----------------|-----|
| PyTorch fp32 reference             | [-12.443, 5.157]| 0.000           | 0   |
| Baseline Flow, `CPU_AND_GPU`       | [-12.500, 5.172]| 4.7e-02         | 0   |
| ANE-port Flow, `CPU_AND_NE`        | **[-10.094, -0.825]** | **2.582** | 0 |

The ANE port produces finite, NaN-free mel — but the dynamic range is
compressed by ~7 dB at the top and the peak energy bins (vocal
formants) are clipped entirely. HiFT fed these flat mels yields
near-silence. This is NOT the "Phase 3 fused-LayerNorm NaN" failure
mode the Stage 0 plan was gated against, and the plan's "0/5 NaN"
gate in Stage 3 passed only because bench inputs didn't trigger the
saturation pattern the parity fixture does.

Hypothesis (not yet pinned down): the BC1S rewrite introduces a
precision loss in the AdaLN `(1+scale)*norm` or manual-SDPA softmax
path that accumulates across 22 blocks × 10 Euler steps × CFG batch=2,
manifesting as progressive magnitude attenuation rather than a single
NaN. Stage 1 ("NaN probe") was skipped because Stage 0's unfuse-LN
experiment passed and no NaN appeared; we now need a **range-probe**
variant that tracks fp16 peak magnitudes per block against the fp32
shadow.

### Revert

- `build/upload/cosyvoice3-coreml/Flow-N250-fp16.{mlpackage,mlmodelc}`
  — restored from `build/flow-fp16-n250/` (the original cpuAndGPU
  artifact, 638 MB / 669208054 bytes).
- `FluidAudio/.../CosyVoice3ModelStore.swift` —
  `flowConfig.computeUnits` back to `.cpuAndGPU`; docstring rewritten
  to document the ANE attempt + revert rationale.
- `FluidAudio/.../CosyVoice3Constants.swift` — header comment block:
  Flow row back to `cpuAndGPU`, notes the ANE port was attempted and
  reverted.
- `build/upload/cosyvoice3-coreml/manifest.json` — Flow
  `compute_units: cpuAndGPU`, purpose rewritten, size_bytes corrected.
- `build/upload/cosyvoice3-coreml/README.md` — Flow row: "CPU + GPU";
  opening paragraph ("Neural Engine for Prefill + HiFT, CPU+GPU for
  Decode + Flow").
- `FluidAudio/.../CosyVoice3Synthesizer.swift` `runHiFT()` — the
  fp16/fp32 dtype branch is kept. It's harmless on fp32 (just takes the
  second arm) and makes the path future-proof if a correct fp16 Flow
  variant ever ships.

### Residual hygiene

- `build/flow-ane-n250/Flow-N250-ane.mlpackage` → symlink to the
  ANE mlpackage; kept for re-debugging (range probe, per-block shadow
  trace).
- `build/flow-ane-fp16-n250/` — original ANE-port build; kept.
- `verify/test_coreml_e2e_fp16.py` — `--flow-precision` choices now
  include `"ane"` for follow-up debugging runs.

### What would unblock the port

1. Add per-block fp32-shadow range probe (planned Stage 1 but skipped);
   identify which block first diverges in magnitude.
2. Audit the BC1S rotary sin/cos pre-bake: the real-valued rotate-half
   pattern must match x_transformers' even/odd interleaved layout
   exactly, and the 10 timestep embeddings must all fall inside the
   baked table.
3. Audit `ANEAttention` softmax: without per-head scaling of QK⊤ before
   softmax, fp16 can quietly underflow entire rows to 0 for large S.
4. Audit `ANEAdaLayerNormZero`: `(1 + scale) * norm` with
   `scale ∈ [-1, +1]` is fine, but larger modulation ranges (which the
   fp32 path shrugs off) hit fp16 at 2-layer cascades.

---

## Findings preserved from removed exploratory scripts

A batch of one-shot debug / probe / parity scripts and the abandoned ANE
BC1S port were removed from the tree (recoverable via `git log`). One
non-trivial finding was **not** previously captured:

### Phase 3 NaN root cause is SDPA `QK⊤` overflow, not fused `layer_norm`

Phase 3 above attributed the Flow fp16 NaN to the fused `layer_norm` MIL
op (which can't be reached by op-type-based fp32 pinning). That is a
contributing factor but **not the primary trigger**. The `nan_probe.py`
shadow-fp32 hooks localized the actual blowup to
`F.scaled_dot_product_attention`: the intermediate `QK⊤ * scale` tensor
exceeds fp16 max (65504) in **9 of 22 DiT blocks**, peaking at **~1.6M
at block 17**. coremltools lowers SDPA to
`matmul → mul(scale) → add(mask) → softmax → matmul`; the `QK⊤` matmul
output materializes in ambient (fp16) precision, so even with `softmax`
pinned to fp32 it receives already-saturated `+inf` inputs → NaN.

Practical consequence: a future fp16 Flow attempt should pin
`{matmul, select, where}` to fp32 (decompose SDPA so the QK⊤/softmax/PV
core casts up explicitly) **before** worrying about layer_norm. The
removed `convert-flow.py --fp32-sdpa` path implemented exactly this and
brought parity from `NaN` to a finite drift, but couldn't be combined
with the ANE-port BC1S rewrite, which itself failed the audio
intelligibility gate (Stage 4 above).

### Other findings — already captured above; scripts removed as one-shots

- HiFT op-count audit (`count_ops_hift.py`, `count_ops_hift_decode.py`):
  Phase 0 table — HiFT decode = 1,174 ops, full = 1,380 ops vs PR #42's
  fabricated 705,848.
- Sinegen `cumsum + sin` precision (`test_cumsum_precision.py`):
  Phase 1 — wrap phase mod-2π before `sin`; CoreML vecLib FP32 `sin`
  diverges on large arguments.
- HiFT determinism (`test_determinism.py`): Phase 5 — Flow/HiFT
  run-to-run max\|Δ\| = 0.0; the residual 0.52 drift was fixture
  staleness, not non-determinism.
- HiFT decode/source per-stage parity (`test_source_*`,
  `test_decode_only_coreml.py`, `test_intermediate_parity.py`,
  `test_fold_*`): all converged at the int16 floor; covered by Phase 5.
- Upstream `forward_one_step` mask audit (`debug_llm_*`): Phase 2 —
  HF's length-1 attention_mask zeros past tokens; routed around via
  our `Qwen2Prefill`/`Qwen2Decode` re-implementation.

### ANE BC1S port artifacts

`compare-flow-ane.py`, `src/ane_attention.py`, `src/ane_layernorm.py`,
`src/ane_layers.py`, `src/conv_pos_ane.py`, `src/dit_ane.py`,
`src/flow_coreml_ane.py`, `src/state_dict_port.py`,
and the matching `convert-flow.py --ane-port / --unfuse-ln / --fp32-sdpa`
flags were removed. Reasons documented above (Stage 4 / `conv_pos_embed`
CPU island sections):

1. The BC1S rewrite's mel dynamic range collapsed by ~7 dB on the parity
   fixture (range `[-10.094, -0.825]` vs reference `[-12.443, 5.157]`,
   MAE 2.582), yielding audio unintelligible to both CTC-ZH and Qwen3
   ASR. NaN-free but silent.
2. `conv_pos_embed`'s `Conv1d(1024, 1024, k=31, groups=16)` is rejected
   by ANEF on a `kernel_size × in_channels` weight-bandwidth basis,
   independent of group count or activation choice. Dense `groups=1`
   with block-diagonal weight (Option L) compiled but stayed CPU-evicted
   at 16× FLOPs → wall-time regression.

To resume the port: `git log --diff-filter=D --follow` the deleted
files, plus the four "what would unblock the port" items in Stage 4
(per-block fp32-shadow range probe, RoPE sin/cos audit, `ANEAttention`
softmax scaling audit, `ANEAdaLayerNormZero` modulation-range audit).

---

## Commit trail

| SHA | Title |
|---|---|
| `8b0d3a5` | feat(tts/cosyvoice3): Add complete MB-MelGAN fine-tuning pipeline *(later abandoned)* |
| `856eb39` | docs(tts/cosyvoice3): Add CoreML conversion patterns and MB-MelGAN optimization benchmarks *(later abandoned)* |
| `b31d1e3` | Flow fp16 parity NaN captured |
| `baf565d` | Flow fp16v2 (layer_norm + gelu pinned) — still NaN |
| `1bb5e3e` | feat(tts/cosyvoice3): pivot from MB-MelGAN sandbox to direct CoreML pipeline (117 files, +7351/−15292) |

Session transcript source:

- `/Users/kikow/.claude/projects/-Users-kikow-brandon-voicelink-mobius/698c92e5-d515-4614-b9fc-3235382fe8e8.jsonl` (81 MB — primary)
- `/Users/kikow/.claude/projects/-Users-kikow-brandon-voicelink-mobius/74a1cd32-02f4-4995-8378-0759dee6b947.jsonl` (800 KB — Parakeet PR review, no CV3 content)
- `/Users/kikow/.claude/projects/-Users-kikow-brandon-voicelink-mobius/e4f62029-9705-4595-9caf-5a2407fbc0d4.jsonl` (70 KB — trivial)
