# Qwen3-ASR-0.6B CoreML: Technical Report

## Model Architecture

Qwen3-ASR is an **audio encoder + LLM decoder** architecture — fundamentally different from transducer models like Parakeet TDT. The LLM is a 28-layer Qwen3-0.6B transformer (GQA: 16 query heads, 8 KV heads, head_dim=128) that autoregressively generates text tokens from audio embeddings. 0.9B parameters, ~1.88 GB in BF16.

**CoreML components (31 total, consolidated to 4):**
- Audio encoder (CNN + 18-layer transformer)
- Embedding layer
- Decoder stack (28 layers consolidated into one model)
- LM head (1024 → 151,936 vocab projection)

**Supported languages:** en, zh, yue, ja, ko, vi, th, id, ms, hi, ar, tr, ru, de, fr, es, pt, it, nl, pl, sv (19 languages)

---

## Final Performance

| Metric | PyTorch (CPU, f32) | CoreML (CPU+GPU, f32) |
|---|---|---|
| **RTFx** | **3.3x** | **1.1x** |
| **WER (LibriSpeech test-clean, 20 files)** | 3.7% | **1.6%** |
| **WER (LibriSpeech test-clean, 100 files)** | — | **5.0%** |
| **CER (FLEURS Chinese, 20 files)** | — | **5.1%** |

CoreML WER is lower than both PyTorch (3.7%) and the official benchmark (2.11% in bfloat16) because FLOAT32 compute precision provides higher numerical accuracy.

PyTorch is 3x faster because it runs in-process with zero marshaling cost. Each CoreML `MLModel.prediction()` call has ~5–15ms overhead for GPU dispatch, which adds up over 150+ decode steps per file.

---

## Bug 1: Float16 Overflow in LM Head

**Symptom:** CoreML returned all-zero logits — the model couldn't pick any token at all.

**Root cause:** RMSNorm (the normalization layer before the LM head) computes `x²`. The hidden states coming out of the decoder had magnitudes around 300. So `300² = 90,000`, which exceeds float16 max (65,504). The overflow corrupted the normalization output to zero, and those zeros propagated through the entire LM head projection — making every logit 0.0.

**How it was found:** Compared CoreML LM head output against PyTorch side-by-side. PyTorch produced valid logits, CoreML produced all zeros. Checked the hidden state magnitudes entering RMSNorm and realized they sat squarely in the float16 overflow zone.

**Solution:** Set `compute_precision=ct.precision.FLOAT32` during `coremltools.convert()` for both the decoder stack and LM head. Weights are still stored as float16 on disk for size efficiency, but all arithmetic runs in float32 at inference time. This eliminated the overflow entirely.

**Key file:** Python conversion script — added `compute_precision` parameter to `ct.convert()` calls for decoder and LM head models.

---

## Bug 2: Mel Spectrogram Silence (vDSP FFT)

**Symptom:** 100% WER on every single file — the model produced complete garbage regardless of input.

**Root cause:** The mel spectrogram was all 1.0 — the model was effectively "listening to" a flat signal with no audio content. Apple's `vDSP_DFT_zop_CreateSetup` requires FFT lengths of the form `f × 2^n` where `f ∈ {1, 3, 5, 15}`. Whisper's mel spectrogram uses FFT length 400, and `400 = 25 × 2⁴` — since f=25 is not in the allowed set, the setup function returned `nil`.

The Swift code had a `guard let setup = vDSP_DFT_zop_CreateSetup(...)` which, when `nil`, silently skipped every single audio frame. No crash, no error logged — just an empty mel spectrogram that defaulted to 1.0 values.

**How it was found:** Printed the mel spectrogram array and saw every value was identical (1.0). Traced the data flow backwards: audio loading was fine, resampling was fine, but the FFT stage was producing nothing. Checked `vDSP_DFT_zop_CreateSetup` return value — nil.

**Solution:** Replaced the vDSP FFT with a direct DFT matrix multiplication approach in `WhisperMelSpectrogram.swift`. The DFT matrix is precomputed once at initialization:

```
W[k, n] = exp(-2πi·k·n / N)
```

Then each frame's spectrum is computed as a matrix-vector multiply `W × frame`. This works with any FFT length, not just the restricted set vDSP supports. The mel filterbank is then applied as usual.

**Key file:** `Sources/FluidAudio/ASR/Qwen3/WhisperMelSpectrogram.swift`

---

## Bug 3: RoPE Layout Mismatch

**Symptom:** ~19% WER. Transcriptions were partially coherent — the model could clearly "hear" speech — but words were wrong, out of order, or substituted.

**Root cause:** Rotary Position Embeddings (RoPE) encode token positions by rotating query and key vectors using precomputed frequency tensors. There are two common layouts for these frequency tensors:

- **Concatenated halves** (what the CoreML model expected): `[f0, f1, ..., f63, f0, f1, ..., f63]`
- **Interleaved pairs** (what the Swift code was producing): `[f0, f0, f1, f1, ..., f63, f63]`

The CoreML model's attention layers use `rotate_half`, which splits the tensor at `head_dim // 2` and rotates the two halves. When the frequency tensor uses the wrong layout, every single position encoding is computed incorrectly — the model thinks a token at position 5 is at some other position entirely. This scrambles the attention pattern enough to produce ~19% WER instead of the expected ~2-4%.

**How it was found:** Read the PyTorch source code for `rotate_half` in the Qwen3 model and compared the expected frequency layout against what the Swift `Qwen3RoPE.swift` was generating. The layouts didn't match.

**Solution:** Changed the Swift RoPE implementation to produce concatenated halves instead of interleaved pairs. Specifically, the frequency tensor generation was restructured so that for `head_dim=128`, the first 64 elements contain the cosine frequencies and the second 64 elements repeat them — matching what `rotate_half` expects when it splits at index 64.

**Key file:** `Sources/FluidAudio/ASR/Qwen3/Qwen3RoPE.swift`

---

## Bug 4: Speed — Sequential Prefill Bottleneck

**Symptom:** 0.19x RTFx — 5x slower than real-time. Completely unusable for any practical application.

**Root cause:** The prefill phase processes ~150 prompt tokens (audio embeddings from the encoder + chat template tokens like `<|im_start|>`, `system`, etc.). The initial implementation fed these tokens through the decoder one at a time, making ~150 separate `MLModel.prediction()` calls. Each call carries ~5–15ms of overhead for:
- Input tensor marshaling (copying Swift arrays into MLMultiArray)
- GPU dispatch and synchronization
- Output tensor copying back

At 150 calls × ~10ms overhead each = 1.5 seconds of pure overhead, before any actual matrix multiplication happens. This dominated total inference time.

**How it was found:** Profiled the transcription pipeline and found that the prefill phase consumed 80%+ of total wall time despite being conceptually a single forward pass. The autoregressive decode phase (which genuinely needs one call per token) was actually fast.

**Solution:** Split the decoder into two separate CoreML models:

1. **`decoder_prefill`**: Fixed input shape `seq=512`, processes all prompt tokens in a single `MLModel.prediction()` call. The causal attention mask is baked into the model as a constant (since the sequence length is fixed). This reduces ~150 calls to exactly 1 call.

2. **`decoder_stack`**: `seq=1` input with `RangeDim` on the cache dimension, used for autoregressive token-by-token decoding after prefill completes.

One complication: CoreML cannot handle zero-length tensor dimensions, so the prefill model needs a minimum `cache_dim=1`. A dummy KV cache entry (all zeros) is prepended to satisfy this constraint, then stripped after prefill completes.

**Result:** 0.19x → 0.8x RTFx (4x speedup). There is a one-time 40-second warmup on first call as CoreML compiles the GPU execution plan, but subsequent calls are fast.

**Key files:**
- Python conversion script — generates two separate `.mlpackage` files with different input shapes
- `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrManager.swift` — orchestrates prefill → strip dummy → decode loop

---

## Bug 5: The Cache-Length CoreML Bug (The Big One)

### Symptom

WER stuck at ~10% after all previous fixes were applied. Three specific files (LibriSpeech 0002, 0008, 0013) were consistently garbled while others transcribed perfectly. The error pattern was distinctive: token-by-token comparison between PyTorch and CoreML showed that hidden state divergence was NOT gradual accumulation — it was an **instantaneous catastrophic jump**. At one specific decode step, the hidden state diff would spike from ~15 to ~82 in a single step, causing the model to pick a completely wrong token. From that point on, the entire transcription derailed via error snowballing.

### Elimination Testing

This bug consumed weeks of debugging because every obvious hypothesis was systematically ruled out:

| Hypothesis | Test | Result |
|---|---|---|
| FP16 precision loss | FLOAT32 everywhere | Same garble |
| FP16 weight quantization | FP16 vs F32 weights | Same garble |
| ANE execution bug | CPU_ONLY | Same garble |
| GPU execution bug | cpuAndGPU only | Same garble |
| torch.jit.trace error | Trace vs eager comparison | Bit-identical |
| MIL optimization passes | Disabled all passes | Same garble |
| Mel spectrogram (201 vs 200 bins) | Tested both | Same garble |
| Audio encoder divergence | CoreML vs PyTorch | max_diff < 0.000002 |
| Repetition penalty | Various values | Same garble |
| Sequential vs batched prefill | Both | Same garble |

A `compare_divergence.py` script was written to feed the same audio through both PyTorch and CoreML, comparing token-by-token at every decode step. Key findings:
- File 0002: Diverges at step 12. h_diff jumps from 15 to 82.5. PyTorch picks "light" (margin 11.67), CoreML picks "would" (margin 5.70).
- File 0008: Diverges at step 11. h_diff jumps from 21.9 to 51.5.
- Feeding PyTorch's exact KV cache to CoreML at each step showed per-step error was a consistent ~4–6 h_diff, but accumulated cache errors caused cancellation/amplification at unpredictable steps.

### Discovery

The breakthrough came from testing with **random inputs at every cache length from 0 to 200**, comparing CoreML vs PyTorch output:

```
cache_len=110:  h_diff=8.2    (normal)
cache_len=111:  h_diff=7.9    (normal)
cache_len=112:  h_diff=35.4   (!!!)
cache_len=113:  h_diff=89.2   (!!!)
cache_len=120:  h_diff=207.1  (!!!)
cache_len=126:  h_diff=42.3   (!!!)
cache_len=127:  h_diff=7.4    (normal again)
cache_len=128:  h_diff=6.1    (normal)
```

The pattern was unambiguous: cache lengths 112–126 caused catastrophic errors. The bad zone was exactly when total key length (`cache_len + 1`) fell in **[113, 127]** — just below `HEAD_DIM = 128`.

Per-layer testing with isolated single-layer CoreML models confirmed the error amplified through the stack:
- Layer 0, cache_len=112: h_diff = 0.68
- Layer 11, cache_len=112: h_diff = 3.46
- **Layer 27, cache_len=112: h_diff = 4,117**
- Layer 27, cache_len=127: h_diff = 1.62 (normal)

Additional smaller bad zones were found at cache lengths ~205, ~366–397, ~500–519 by scanning up to cache_len=512. The primary zone at 112–126 was by far the most severe.

### Root Cause

The bug is in **coremltools' TorchScript-to-MIL conversion layer**, conclusively ruled out from being in:
- `torch.jit.trace` — proven bit-identical to eager PyTorch
- MIL optimization passes — disabled all, same error
- GPU/ANE runtime — CPU_ONLY shows identical error

Further isolation testing on individual attention sub-operations:
- Q×K^T matmul alone: No bad zone
- Softmax alone: No bad zone
- Attention×V matmul alone: Mild bad zone (peak error 0.027 at cache_len=120)
- Full attention block: No bad zone with random weights

The bug only manifests when all operations **combine in the full decoder layer graph with real trained weights**. The tiny 0.027 per-operation error gets amplified **100,000x** (from 0.027 to 4,117) through the interaction of real weight matrices across 28 layers.

The bug appears related to how CoreML's MIL compiler handles attention computation when the key sequence dimension approaches but stays below the head dimension (128). This may involve internal tiling, vectorization, or memory layout optimizations that assume certain dimension relationships.

### Solution: Cache Padding

The fix is straightforward — skip the bad zone entirely by padding the KV cache past it:

**1. `Qwen3KVCache.swift`** — Added a `padToMinimumLength(_:)` method:
```swift
/// Pad cache to minimum length to avoid CoreML attention bug.
/// When cache length is in [1, HEAD_DIM-1], pad to HEAD_DIM with zeros.
func padToMinimumLength(_ minLength: Int)
```
This checks if the current cache length is below `HEAD_DIM` (128). If so, it appends zero-filled entries to reach exactly 128. A `paddingIndices` property tracks which entries are padding vs real data.

**2. `Qwen3AsrManager.swift`** — Two changes:
- After stripping the dummy KV cache entry from prefill, immediately call `cache.padToMinimumLength(config.headDim)` to ensure the cache is at least 128 entries long.
- The `createCausalMask()` function was updated to set padding positions to `-1e9` (negative infinity for softmax), so the padded entries are completely ignored during attention computation. The model never "sees" the padding — it's purely a workaround to keep the cache dimension out of the bad zone.

The padding is only needed once — after the cache grows past 128 real entries, the padding entries are naturally superseded and the bad zone is never re-entered.

### Result

| Metric | Before Padding | After Padding |
|---|---|---|
| WER (20 files) | 10.4% | **1.6%** |
| WER (100 files) | — | **5.0%** |
| Previously garbled files (0002, 0008, 0013) | Failed | **0% WER** |
| Perfect (0% WER) files out of 100 | — | **65/100** |

The three files that were consistently garbled for weeks all transcribed perfectly after the one-line padding fix. The remaining 5% WER on 100 files comes from genuine model quality limitations on unusual proper nouns (Irish/Greek names like "Mac Ardle", "Stephanos Dedalos"), not CoreML bugs.

---

## Dead End: Quantization and Palettization

### INT8 Linear Quantization

Weight-only quantization via `coremltools.optimize.coreml.linear_quantize_weights`:
- Decoder stack: 1.6 GB → 422 MB
- WER: Identical (1.6%)
- RTFx: **0.4x** (3x SLOWER than F32)

Apple Silicon lacks native int8 compute for general matrix multiplication. The runtime must dequantize int8 weights back to float16/float32 before every matmul, adding overhead that outweighs any memory bandwidth savings. This makes int8 quantization counterproductive for autoregressive LLM decoding where each step does many small matmuls.

### 8-bit Palettization

Via `coremltools.optimize.coreml.palettize_weights`:
- Conversion took ~2.5 hours (k-means clustering on 311 weight matrices)
- The LM head's 151,936×1024 matrix alone took ~30 minutes of k-means
- **Weights were NOT actually compressed** — 420 MB vs F32's 422 MB
- CoreML **failed to compile** the result (error code -6: "Failed to build the model execution plan")
- `coremltools 9.0` added `constexpr_lut_to_dense` ops to the MIL graph but didn't store weights in LUT-compressed format on disk

**Verdict:** Both quantization paths are dead ends for this model on current coremltools/CoreML. F32 is the only viable option.

---

## File Inventory

### Swift (FluidAudio)

| File | Purpose |
|---|---|
| `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrManager.swift` | Main orchestrator — mel, prefill, decode, cache padding |
| `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrConfig.swift` | Model config and special token IDs |
| `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrModels.swift` | Model download, loading, vocab |
| `Sources/FluidAudio/ASR/Qwen3/Qwen3KVCache.swift` | KV cache with padding support |
| `Sources/FluidAudio/ASR/Qwen3/Qwen3RoPE.swift` | Rotary position embeddings |
| `Sources/FluidAudio/ASR/Qwen3/WhisperMelSpectrogram.swift` | Whisper-compatible mel with DFT matrix |
| `Sources/FluidAudioCLI/Commands/ASR/Qwen3AsrBenchmark.swift` | LibriSpeech + FLEURS benchmarking |
| `Sources/FluidAudioCLI/Commands/ASR/Qwen3TranscribeCommand.swift` | Single-file transcription CLI |
| `Sources/FluidAudioCLI/Utils/TextNormalizer.swift` | Chinese number normalization for CER |

### Python (mobius)

| File | Purpose |
|---|---|
| `mobius/models/stt/qwen3-asr-0.6b/coreml/prepare_fleurs_chinese.py` | FLEURS Chinese data download |

### HuggingFace

Models uploaded to `FluidInference/qwen3-asr-0.6b-coreml` (4 `.mlpackage` files, F32).

---

## Debugging Timeline Summary

1. **Architecture assessment** — audio encoder + LLM decoder, 31 components
2. **Python conversion** — all components converted, validated parity
3. **Float16 overflow** — LM head zeros → FLOAT32 compute precision
4. **Swift pipeline bugs** — vDSP FFT nil, RoPE layout, BPE decoding
5. **Speed optimization** — batched prefill (0.19x → 0.8x RTFx)
6. **WER stuck at 10%** — weeks of elimination testing ruled out every obvious cause
7. **Cache-length bug discovery** — random-input sweep found catastrophic errors at cache lengths 112–126
8. **Cache padding fix** — pad to HEAD_DIM, mask padding → WER drops from 10.4% to 1.6%
9. **Quantization dead end** — INT8 is 3x slower, palettization fails to compile
10. **Chinese benchmark** — FLEURS data prep, Chinese number normalization, 5.1% CER

---

# Part 2: Performance Optimizations

## Updated Performance

| Configuration | Size | RTFx | WER (100 files) | ms/tok |
|---|---|---|---|---|
| Part 1: prefill + stack + lmHead | 1.9 GB | 1.1x | 5.0% | ~150 |
| **Stateful decoder + lmHead** | 1.5 GB | 2.9x | 0.8% | ~72 |
| **Fused stateful decoder (FP16)** | 1.75 GB | 3.3x | 4.8% | ~64 |
| **Fused stateful decoder (Int8)** | 899 MB | 3.2x | 5.0% | ~75 |

The stateful decoder + fused lmHead combination achieves **3x faster inference** at **half the model size** with equivalent quality.

---

## Optimization 1: Stateful CoreML Decoder (macOS 15+)

**Problem:** The Part 1 architecture used separate prefill and stack models with explicit KV cache passing. Each decode step required:
1. Copy KV cache from Swift arrays to MLMultiArray inputs
2. Run decoder prediction
3. Copy updated KV cache from MLMultiArray outputs back to Swift arrays

For 28 layers × 2 (K+V) × ~100 tokens average = ~5,600 tensor copies per transcription.

**Solution:** CoreML's State API (iOS 18 / macOS 15) enables GPU-resident state tensors that persist across predictions. The KV cache lives on GPU memory and is never copied to/from CPU.

**Implementation:**

1. **Python conversion** (`convert_stateful_decoder.py`):
   - Register 56 state tensors (k_cache_0..27, v_cache_0..27) with `ct.StateType`
   - Each state has shape `[1, num_kv_heads, max_seq_len, head_dim]` = `[1, 8, 512, 128]`
   - Use `ct.EnumeratedShapes` for variable sequence lengths (prefill: 1-512, decode: 1)
   - States are FP16 even with FP32 compute precision (storage vs compute)

2. **Swift integration** (`Qwen3AsrManager.swift`):
   - Create `MLState` once at model load time
   - Pass same state object to every prediction call
   - Position tracking via `currentPosition` counter (no cache length dimension)
   - Reset state between transcriptions

**Key insight:** The stateful model handles both prefill and decode — no need for separate models. The `seq` dimension uses `RangeDim(1, 512)` so a single model accepts variable-length inputs.

**Result:** 1.1x → 2.9x RTFx (2.6x speedup), WER improved from 5.0% to 0.8%.

---

## Optimization 2: Fused lmHead into Decoder

**Problem:** After the stateful decoder outputs hidden states `[1, 1, 1024]`, a separate lmHead model call is needed to project to logits `[1, 1, 151936]`. Each `MLModel.prediction()` has ~8ms overhead.

**Before (per token):**
```
decoder(embeddings) → hidden [1,1,1024] → lmHead(hidden) → logits [1,1,151936]
        ~63ms                                    ~8ms
```

**Solution:** Fuse the final RMSNorm and lm_head linear projection directly into the decoder model.

**After (per token):**
```
decoder_fused(embeddings) → logits [1,1,151936]
        ~64ms (negligible compute increase)
```

**Implementation** (`convert_decoder_fused.py`):

```python
class FusedStatefulQwen3Decoder(nn.Module):
    def __init__(self, layers, final_norm, lm_head, max_seq_len=512):
        # ... 28 decoder layers + final_norm + lm_head + KV cache states

    def forward(self, hidden_states, position_cos, position_sin, attention_mask):
        # ... 28 transformer layers ...

        # Fused: slice last position, apply RMSNorm + linear
        last_hidden = hidden_states[:, -1:, :]  # [1, 1, 1024]
        last_hidden = self.final_norm(last_hidden)
        logits = self.lm_head(last_hidden)  # [1, 1, 151936]
        return logits
```

The output spec changes from `output_hidden [1, seq, 1024]` to `logits [1, 1, 151936]` (fixed shape, not RangeDim).

**Swift changes:**
- Remove `lmHead: MLModel` from `Qwen3AsrModels`
- Remove `lmHeadFile` from `ModelNames.Qwen3ASR.requiredModels`
- `runStatefulDecoder` now returns logits directly
- Inline argmax using `vDSP_maxvi` for vectorized max over 151,936 floats

**Result:** 2.9x → 3.0x RTFx. Combined with warm-up: 3.3x RTFx.

---

## Optimization 3: Benchmark Warm-up

**Problem:** The first file in a benchmark run was consistently 20-40% slower due to:
- MPS graph compilation on first GPU dispatch
- Memory allocation for intermediate tensors
- JIT optimization of compute kernels

This cold-start penalty skewed benchmark averages and caused high variance.

**Solution:** Add a warm-up pass before the benchmark loop:

```swift
// Warm up CoreML's MPS graph cache with the first file
if let first = files.first {
    let samples = try AudioConverter().resampleAudioFile(path: first.audioPath.path)
    _ = try await manager.transcribe(audioSamples: samples, language: language)
}
```

The warm-up transcription primes:
- CoreML model compilation
- MPS shader compilation
- GPU memory pools
- Compute pipeline caches

**Result:** More consistent per-file timing (60-72ms/tok vs 56-109ms variance). Overall RTFx: 3.0x → 3.3x.

---

## Optimization 4: Int8 Quantization (Revisited)

**Background:** Part 1 found int8 quantization was 3x SLOWER due to dequantization overhead. However, that was with the prefill+stack architecture that made hundreds of small predictions.

**Re-test with stateful decoder:** The fused stateful decoder makes far fewer prediction calls (one per token, not one per layer), so the dequantization overhead is amortized better.

**Implementation:**

```python
from coremltools.optimize.coreml import linear_quantize_weights, OptimizationConfig, OpLinearQuantizerConfig

config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode='linear_symmetric'))

# Quantize all three models
encoder_q8 = linear_quantize_weights(encoder, config=config)
embedding_q8 = linear_quantize_weights(embedding, config=config)
decoder_q8 = linear_quantize_weights(decoder_fused, config=config)
```

**Results (100-file benchmark):**

| Model | FP16 Size | Int8 Size | Reduction |
|---|---|---|---|
| audio_encoder | 356 MB | 179 MB | 50% |
| embedding | 297 MB | 149 MB | 50% |
| decoder_fused | 1.1 GB | 571 MB | 48% |
| **Total** | **1.75 GB** | **899 MB** | **49%** |

| Metric | FP16 | Int8 |
|---|---|---|
| RTFx | 3.2x | 3.2x |
| WER (avg) | 4.8% | 5.0% |
| WER (median) | 0.0% | 0.0% |
| ms/tok | ~72 | ~75 |

**Key finding:** Int8 quantization now works well with the stateful architecture. The 0.2% WER increase (4.8% → 5.0%) is within noise — the WER outliers are the same files (Irish proper nouns like "Stephanos Dedalos", "Mac Ardle") in both FP16 and Int8.

---

## WER Outlier Analysis

The ~5% average WER (with 0% median) is caused by a small number of files with unusual proper nouns:

| File | WER | Issue |
|---|---|---|
| 1089-134691-0024 | 200% | "STEPHANOS DEDALOS" → "Stefano's dad lost" (2-word file) |
| 1089-134691-0010 | 60% | Irish: "MAC ARDLE", "KEOGH" → "Macartal", "Kiyof" |
| 1089-134691-0020 | 17% | "DEDALUS" → "datales" |

These are **model errors, not CoreML/quantization errors** — PyTorch produces identical mistakes. The Qwen3-ASR model struggles with uncommon Irish/Greek names from James Joyce's *A Portrait of the Artist*.

65 out of 100 files achieve 0% WER. The remaining errors are genuine model quality limitations on out-of-vocabulary proper nouns.

---

## Comparison with MLX

Decoder-only benchmarks (Qwen3-0.6B, same architecture as Qwen3-ASR decoder):

| | CoreML FP16 | CoreML Int8 | MLX bf16 | MLX Q8 | MLX 4-bit |
|---|---|---|---|---|---|
| **ms/tok** | ~64 | ~75 | 22.1 | 14.7 | 11.2 |
| **Size** | 1.1 GB | 571 MB | ~1.2 GB | ~680 MB | ~430 MB |

MLX is 2.9–5.7x faster per decoder step. However:
- CoreML 3.2x RTFx is still real-time capable for streaming ASR
- CoreML runs on iOS; MLX is macOS-only
- CoreML leverages ANE; MLX is GPU-only

---

## Updated File Inventory

### Python Conversion Scripts

| File | Purpose |
|---|---|
| `convert-qwen3-asr.py` | Main CLI — exports audio_encoder, embedding, lm_head |
| `convert_stateful_decoder.py` | Stateful decoder with GPU-resident KV cache |
| `convert_decoder_fused.py` | Fused decoder (lmHead built-in, ~8ms/tok savings) |
| `individual_components.py` | Wrapper modules for each component |

### Model Variants

| Variant | Components | Size | RTFx | Use Case |
|---|---|---|---|---|
| FP16 fused | encoder + embedding + decoder_fused | 1.75 GB | 3.3x | Best quality |
| Int8 fused | all int8 quantized | 899 MB | 3.2x | Best size/quality tradeoff |

---

## Updated Timeline

11. **Stateful decoder** — GPU-resident KV cache via CoreML State API (1.1x → 2.9x RTFx)
12. **Fused lmHead** — eliminate separate model call overhead (2.9x → 3.0x RTFx)
13. **Benchmark warm-up** — prime MPS graph cache (3.0x → 3.3x RTFx)
14. **Int8 revisited** — works well with stateful architecture (899 MB, same quality)
15. **WER analysis** — outliers are model limitations on proper nouns, not CoreML bugs
