# Qwen3-TTS CoreML Conversion: Trials and Errors

A chronological record of every issue encountered while converting Qwen3-TTS 0.6B from PyTorch to CoreML, reducing the model from ~5 GB to ~1 GB, and fixing the robotic audio output.

---

## Phase 1: The Starting Point — Monolithic Models (~5 GB)

The original approach converted the Qwen3-TTS 0.6B model as two large monolithic CoreML models:

| Model | Size | Description |
|-------|------|-------------|
| `qwen3_tts_lm_prefill_v9.mlmodelc` | ~2.8 GB | Full prefill (text → hidden states), float32 |
| `qwen3_tts_lm_decode_v10.mlmodelc` | ~1.8 GB | Autoregressive decode, float32 |
| **Total** | **~4.6 GB** | |

These models were float32, unquantized, and bundled the entire pipeline into two pieces. The audio output had issues: Chinese sounded choppy, English sounded robotic.

---

## Phase 2: Discovering the 6-Model Architecture

Studying a reference CoreML implementation of TTSKit revealed the pipeline could be split into **6 specialized models** with W8A16 quantization:

| Reference Model | Reference Size | Our Equivalent | Our Size |
|---|---|---|---|
| TextProjector (W8A16) | ~317 MB | `lm_prefill_v9` | ~2.8 GB |
| CodeDecoder (W8A16-stateful) | ~445 MB | `lm_decode_v10` | ~1.8 GB |
| CodeEmbedder (W16A16) | ~6 MB | (embedded in decode) | — |
| MultiCodeEmbedder (W16A16) | ~60 MB | (embedded in decode) | — |
| MultiCodeDecoder (W8A16) | ~105 MB | (embedded in decode) | — |
| SpeechDecoder (W8A16) | ~109 MB | (separate) | — |

The size difference was dramatic: the reference total was ~1 GB vs our ~4.6 GB.

### Decision: Reverse-engineer the 6-model approach

Read the MIL (Machine Intermediate Language) programs from the reference compiled `.mlmodelc` bundles to understand their exact architecture, then write our own conversion scripts from PyTorch.

---

## Phase 3: Writing 6 Conversion Scripts

### Scripts created:
1. Embedders conversion — CodeEmbedder + MultiCodeEmbedder
2. TextProjector conversion — TextProjector
3. CodeDecoder conversion — 28-layer transformer, stateful KV cache
4. MultiCodeDecoder conversion — 5-layer transformer
5. SpeechDecoder conversion — convolutional audio decoder
6. Master orchestration script — runs all conversions

### Errors during conversion:

**Error: MultiCodeEmbedder segfault during trace**
- The `[30720, 1024]` concatenated embedding table (15 codebooks × 2048 tokens × 1024 dim) caused a segfault during `torch.jit.trace`
- Fix: Process CodeEmbedder and MultiCodeEmbedder separately, release memory between them with `gc.collect()`

**Error: RoPE patching breaks batch forward**
- The `patch_rmsnorm_for_trace` function patches ALL `Qwen3TTSRMSNorm` instances to use `.float()`, but `q_norm` and `k_norm` have different dimensions (2048) than the layer norm (1024)
- Caused `RuntimeError: size of tensor a (1024) must match the size of tensor b (2048)`
- Fix: Only patch RMSNorm during trace, not during validation comparisons

**Error: `_cast` op not supported in coremltools**
- Some PyTorch operations generated `_cast` ops that coremltools couldn't handle
- Fix: Reworked the wrapper to avoid implicit type casts

---

## Phase 4: First Inference Attempt — Garbled Audio

After converting all 6 models, the first inference produced audio with signal (RMS 0.234, max 0.9) but Whisper returned an **empty transcript** — the audio was garbled noise.

### Diagnosis: SpeechDecoder verified correct
- Cosine similarity between PyTorch and CoreML SpeechDecoder: **0.99**
- The SpeechDecoder was not the problem

### Diagnosis: Prefill length mismatch
- CoreML had 18 prefill positions, PyTorch had 17 (later confirmed both were 18 — miscounted)
- Embedding cosine similarity at positions 0-6: **1.0** (perfect match)
- Divergence starting at position 7: cosine dropped to **0.54**
- Root cause: prefill sequence construction in `inference.py` didn't match PyTorch's `generate()` function

---

## Phase 5: KV Cache Timing Bug — The First Major Fix

### Discovery

The CodeDecoder produced dramatically different outputs from PyTorch:
- Hidden state cosine similarity: **0.36** (terrible)
- Logit cosine similarity: **0.47** (very low)
- Top-1 CB0 token: CoreML=1995, PyTorch=1221

### Investigation steps:

1. **Wrapper vs PyTorch batch forward**: The `CodeDecoderWrapper` (pure PyTorch, no CoreML) already had cosine **0.34** vs batch forward — the bug was in the wrapper logic, not CoreML conversion

2. **Layer-by-layer isolation**:
   - 1 layer: cosine = **1.0** (perfect)
   - 2 layers: cosine = **0.995** (slight drift)
   - 28 layers: cosine = **0.34** (catastrophic)

3. **Position 0 divergence**: Even with only 1 token at 1 position, the output diverged (cosine **0.59** at layer 0). This ruled out cache accumulation.

### Root cause

The `_transformer_layer` method computed attention using the **old cache** (which didn't contain the current position's K/V). The current K/V was written to the cache **after** attention, not before.

```python
# BUG: attention uses old cache (missing current position)
attn_output = attention(query, key_cache, value_cache)
key_cache[pos] = new_key    # written too late
value_cache[pos] = new_value

# FIX: write K/V to cache BEFORE attention
key_cache = key_cache * (1 - mask) + new_key * mask
value_cache = value_cache * (1 - mask) + new_value * mask
attn_output = attention(query, key_cache, value_cache)
```

### After fix:
- All 28 layers: cosine = **1.0**, max_diff < **0.001**
- First CB0 token: **1221** (matches PyTorch exactly)

---

## Phase 6: MultiCodeDecoder NaN — The fp16 Overflow

After fixing the CodeDecoder, CB1-CB15 tokens were all zeros. Debugging revealed the MultiCodeDecoder was producing **NaN** outputs.

### Investigation steps:

1. **MCD has the same KV cache timing bug** — applied the same fix as CodeDecoder

2. **NaN persists after KV fix** — with random inputs, the MCD worked fine. With actual CodeDecoder hidden states (max value ~97.4), NaN appeared

3. **Not a quantization issue** — even the non-quantized (fp16-only) model produced NaN

4. **Content-dependent, not magnitude-dependent**:
   - Scaling hidden states to 5% magnitude: still NaN
   - Clamping to ±20: works
   - Clamping to ±30: NaN
   - Shuffling value positions: works
   - Specific positions [512:768] triggered the NaN

5. **RMSNorm weight outliers**: The talker's `model.norm` had weights up to 19.75 in the [512:768] range. Combined with large pre-norm activations, post-norm values reached ~97.4, which caused fp16 overflow in the Q/K/V projection intermediate computations

6. **PyTorch fp16 did NOT overflow** — the NaN was CoreML-specific, likely due to different computation ordering or fused operations

### Fix: FLOAT32 compute precision for MCD

```python
# Before (NaN):
compute_precision=ct.precision.FLOAT16

# After (works):
compute_precision=ct.precision.FLOAT32
```

The MCD is only 5 layers, so FLOAT32 had negligible performance impact.

### After fix:
- CB1-CB15: diverse, non-zero values (14-21 unique per codebook)
- Whisper transcript: "Oh no, this is a task." — intelligible speech, but wrong content

---

## Phase 7: Prefill Construction and Decode Loop Bugs

With models numerically correct, the transcriptions were wrong:

| File | Whisper Transcript | Expected |
|------|-------------------|----------|
| `hello_world_coreml.wav` | "Heart add warmth" | "Hello world, this is a test." |
| `hello_g2p_coreml.wav` | "Healthy world." | "Hello world" |
| `hello_fixed.wav` | "Puntied warrant." | "Hello world" |

### Investigation: Text overlay during decode

The official PyTorch `talker.forward()` feeds **remaining text tokens** during the decode loop:

```python
if generation_step < trailing_text_hidden.shape[1]:
    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step]
else:
    inputs_embeds = inputs_embeds + tts_pad_embed
```

Our inference.py always added `tts_pad` — correct for non_streaming_mode (where all text is in the prefill), but we needed to verify the prefill construction was actually correct for non_streaming_mode.

### Investigation: Greedy comparison

Ran deterministic greedy decode in both PyTorch and CoreML. The PyTorch reference with greedy decoding ALSO produced wrong speech ("Oh, the whole fish is a pest."), confirming the issue was in the pipeline orchestration, not the models.

### User redirect

At this point the user said: *"i think we might be going off topic, we just want to shrink the model sizes to like 1 gb. and then we will see about the audio quality later"*

Model sizes were already at 1.3 GB total. The user confirmed: *"1.3g is good enough for now"*

---

## Phase 8: Getting Correct Transcriptions

After further fixes to the prefill construction and decode loop in `inference.py`:

### Fix: Correct embedding dual-stream construction
The prefill needed to match PyTorch's exact dual-embedding pattern:
- Text stream: `text_projection(text_embedding(token_id))`
- Codec stream: `codec_embedding(codec_token_id)`
- Combined: `text_embed + codec_embed` per position

### Fix: Proper sampling parameters
- CB0 (CodeDecoder): temperature=1.0, top_k=50
- CB1-CB15 (MultiCodeDecoder): temperature=0.9, top_k=50
- Repetition penalty: 1.05

### Result
- Whisper transcript: **"Hello world, this is a test."** — perfect match
- EOS detected properly at reasonable frame counts

---

## Phase 9: Robotic Prosody — "the prosodic is absent"

Audio was intelligible and correctly transcribed, but sounded flat and robotic.

### Hypothesis 1: Missing speaker embedding (partially correct)
- The base 0.6B model requires an **x-vector speaker embedding** for natural prosody
- The "customvoice" variant of the reference implementation uses voice cloning with speaker embeddings
- Created `extract_speaker_embedding.py` to extract 1024-dim x-vector from reference audio via ECAPA-TDNN
- Added `--speaker` and `--ref-audio` flags to `inference.py`

**Error: Standalone mel implementation mismatch**
- Our standalone mel spectrogram extraction gave cosine similarity of only **0.82** vs the official pipeline
- Fix: Use `model.model.extract_speaker_embedding()` directly instead of reimplementing mel computation
- Result: cosine = **1.0** (exact match)

### After adding speaker embedding
- Audio quality improved but **still robotic**
- F0 (pitch) analysis showed similar variance between PyTorch and CoreML, but CoreML had much lower voiced ratio (41% vs 61%)

### Hypothesis 2: Sampling parameters wrong (ruled out)
- Verified: both PyTorch and CoreML use identical sampling (temp=0.9, top_k=50 for subtalker)

### Hypothesis 3: Model conversion errors (ruled out)
Cross-decoder diagnostic (`diagnose_robotic.py`) generated 5 WAV files:

| File | Codes From | Decoder | Quality |
|------|-----------|---------|---------|
| `pytorch_codes_pytorch_decoder.wav` | PyTorch | PyTorch | Natural |
| `coreml_codes_pytorch_decoder.wav` | CoreML | PyTorch | **Natural** |
| `coreml_codes_coreml_decoder.wav` | CoreML | CoreML | **Robotic** |

**Key finding**: Same CoreML codes decoded by PyTorch = natural audio. This proved the codes were fine; the SpeechDecoder was the problem.

Numerical verification of all 3 models:
- CodeDecoder: correlation **1.000000** across 20 autoregressive steps
- MultiCodeDecoder: **15/15** greedy token match, correlation 0.997-0.999
- SpeechDecoder: **0.96-0.999** per-frame correlation

All models were numerically accurate, yet the SpeechDecoder produced robotic audio. Why?

---

## Phase 10: Root Cause — Frame-by-Frame SpeechDecoder

### Discovery

The PyTorch `speech_tokenizer.decode()` calls `chunked_decode`:

```python
def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
    # Feeds 25 frames of LEFT CONTEXT per chunk for smooth cross-frame transitions
```

Our CoreML SpeechDecoder was converted with input shape `[1, 16, 1]` — **single frame only**. The inference loop processed frames one at a time with **zero cross-frame context**.

The SpeechDecoder is a convolutional architecture with 70+ conv layers (including `pre_transformer`, `pre_conv`, `upsample` blocks with kernel sizes up to 7). These convolutions have cross-frame receptive fields that require context from neighboring frames to produce smooth audio transitions.

### Fix attempt 1: EnumeratedShapes (variable-length input)
- Tried converting with `ct.EnumeratedShapes` for T=[1, 10, 50, 125]
- **Failed**: `NotImplementedError: Dynamic padding for n-dimensional tensors is not supported`
- The `CausalConvNet` uses kernel_size-dependent padding that becomes dynamic when T varies

### Fix attempt 2: Traced conditional branch
- Traced with T=10, tried verifying with T=1
- **Failed**: `TracerWarning: Converting a tensor to a Python boolean`
- The `if seq_len > 1` branch in the pre_transformer attention was traced as a constant

### Fix attempt 3: Fixed T=125 batch mode (success)
- Trace with fixed T=125 (maximum frames for 10 seconds of audio)
- Always apply causal mask (removed `if seq_len > 1` conditional)
- Pad shorter sequences to T=125, trim output to actual frame count

**Error: First background conversion killed before saving**
- Task b6952ec completed W8A16 quantization but was killed before `ml_model.save()`
- Had to re-run as task bb8511d

### Conversion result:
- Input: `[1, 16, 125]` (batch of up to 125 frames)
- Output: `[1, 1, 240000]` (125 × 1920 samples per frame)
- Size: 109 MB (W8A16)
- Diff vs PyTorch: 0.649669 (normal for W8A16)

### Inference update:

```python
# OLD (frame-by-frame, robotic):
audio_chunks = []
for frame in all_frames:
    codes = np.array(frame, dtype=np.int32).reshape(1, 16, 1)
    out = speech_dec.predict({"audio_codes": codes})
    audio_chunks.append(out["audio"].flatten())
audio = np.concatenate(audio_chunks)

# NEW (batch, natural prosody):
SPEECH_DEC_T = 125
codes = np.array(all_frames, dtype=np.int32)  # [N, 16]
padded = np.zeros((SPEECH_DEC_T, 16), dtype=np.int32)
padded[:num_frames] = codes
out = speech_dec.predict({"audio_codes": padded.T.reshape(1, 16, SPEECH_DEC_T)})
audio = out["audio"].astype(np.float32).flatten()[:num_frames * 1920]
```

### Result:
- User confirmed: **"batch_speech_decoder_test.wav has prosodic now"**
- Whisper transcript: **"The quick brown fox jumps over the lazy dog."** — perfect match
- Natural prosody with proper intonation

---

## Phase 11: Model Size Reduction — 2,704 MB → 1,056 MB

After Phase 10, the models worked correctly but were 2.6x larger than the reference implementation. The biggest offenders were CodeDecoder (1,775 MB vs 445 MB) and TextProjector (635 MB vs 317 MB).

### Root cause investigation

Checked the `weight.bin` sizes inside each `.mlpackage`:

| Model | weight.bin | Expected (W8A16) | Actual bytes/param |
|-------|-----------|-------------------|-------------------|
| CodeDecoder | 1,774.5 MB | ~447 MB | 3.97 (FP32) |
| TextProjector | 634.9 MB | ~317 MB | 2.0 (FP16) |

**Root cause**: During the KV cache bug fix (Phase 5), both models were reconverted but the `--quantize-w8` flag was accidentally omitted. The CodeDecoder weights were stored as FP32 (4 bytes/param) and TextProjector as FP16 (2 bytes/param) instead of W8A16 palettized (1 byte/param).

### Fix

Reconverted both models with `--quantize-w8`:

```bash
python convert_argmax_code_decoder.py --model-path ./model_0.6b --output-dir ./argmax_models --quantize-w8 --skip-verify
python convert_argmax_text_projector.py --model-path ./model_0.6b --output-dir ./argmax_models --quantize-w8
```

TextProjector verification: max diff 0.008387 vs PyTorch (normal for W8A16 palettization).

### Results

| Model | Before | After | Reduction |
|-------|--------|-------|-----------|
| CodeDecoder | 1,775 MB | 445 MB | 4.0x |
| TextProjector | 635 MB | 318 MB | 2.0x |
| **Total** | **2,704 MB** | **1,056 MB** | **2.6x** |

### Audio quality verification

Ran inference with the newly quantized models — both English and Chinese produce correct Whisper transcriptions:

| Language | Input | Whisper Transcript | Match |
|----------|-------|--------------------|-------|
| English | "The quick brown fox jumps over the lazy dog." | "The quick brown fox jumps over the lazy dog." | Perfect |
| Chinese | "今天天气真好，我们一起去公园散步吧。" | "今天天氣真好,我們一起去公園散步吧" | Perfect (simplified↔traditional is Whisper) |
| Chinese | "人工智能正在改变我们的生活方式，未来将更加美好。" | "人工智能正在改变我们的生活方式,未来将更加美好。" | Perfect |
| Chinese | "我喜欢在周末和朋友们一起爬山，山顶的风景非常壮观。" | "我喜欢在周末和朋友们一起爬山山顶的风景非常壮观" | Perfect |

Model load time also improved: ~355s (down from ~1220s with the old 2.7 GB models).

---

## Final Model Sizes

| Model | Size | Quantization | Precision |
|-------|------|-------------|-----------|
| TextProjector | 318 MB | W8A16 | FLOAT16 |
| CodeEmbedder | 6 MB | W16A16 | FLOAT16 |
| MultiCodeEmbedder | 63 MB | W16A16 | FLOAT16 |
| CodeDecoder | 445 MB | W8A16 | FLOAT16 |
| MultiCodeDecoder | 110 MB | W8A16 | FLOAT32 |
| SpeechDecoder | 115 MB | W8A16 | FLOAT16 |
| **Total** | **~1,056 MB (~1.0 GB)** | | |

Matches reference implementation total (~1,042 MB).

---

## Summary of All Bugs Found and Fixed

| # | Bug | Symptom | Root Cause | Fix |
|---|-----|---------|------------|-----|
| 1 | KV cache timing (CodeDecoder) | Cosine 0.36 vs PyTorch, wrong CB0 tokens | Attention computed before writing current K/V to cache | Write K/V before attention |
| 2 | KV cache timing (MultiCodeDecoder) | CB1-CB15 all zeros, NaN logits | Same bug as CodeDecoder | Same fix |
| 3 | fp16 overflow (MultiCodeDecoder) | NaN in Q/K/V projections | CodeDecoder hidden states have outliers (max ~97), CoreML fp16 overflow | Use FLOAT32 compute precision |
| 4 | Prefill construction | Wrong transcriptions ("Heart add warmth") | Dual-embedding pattern didn't match PyTorch's generate() | Matched official prefill sequence exactly |
| 5 | Frame-by-frame SpeechDecoder | Robotic, flat prosody | No cross-frame context for convolutional decoder | Batch T=125 input, process all frames at once |
| 6 | Speaker embedding missing | Flat prosody | Base 0.6B model requires x-vector for natural speech | Added speaker embedding extraction and injection |
| 7 | `if seq_len > 1` trace branch | Causal mask skipped for T=1 | PyTorch tracing converts conditionals to constants | Always apply causal mask |
| 8 | EnumeratedShapes dynamic padding | Conversion failure | CausalConvNet has kernel-dependent padding | Use fixed T=125 instead of variable shapes |
| 9 | Standalone mel mismatch | Speaker embedding cosine only 0.82 | Our mel filterbank didn't match qwen_tts implementation | Use official extract_speaker_embedding() method |
| 10 | MultiCodeEmbedder segfault | Crash during torch.jit.trace | 30720×1024 embedding table memory pressure | Process models separately with gc.collect() |
| 11 | Missing W8A16 quantization | CodeDecoder 4x too large, TextProjector 2x too large | `--quantize-w8` flag omitted during reconversion after KV cache fix | Reconvert both with `--quantize-w8` |
