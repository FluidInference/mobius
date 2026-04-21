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
One diagnostic script retained: `verify/count_ops_mbmelgan.py`.

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
