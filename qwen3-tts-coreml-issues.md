# Qwen3-TTS CoreML Implementation Issues & Fixes

This document captures the issues encountered during the CoreML port of Qwen3-TTS and their solutions.

---

> **Note:** Issues 1-3 below are from the original 5-model + numpy architecture and have been **superseded** by the 6-model Argmax-style pipeline rewrite. Kept for historical reference only.

## Issue 1: CB0 Token Repetition (Stuck LM) *(superseded)*

### Symptoms
- Chinese audio was silent or unintelligible
- English audio sometimes degraded
- CB0 tokens getting stuck at same values (e.g., `[1657, 1657, 1657, ...]`)
- Only 27 unique CB0 values out of 125 frames (should be ~98% unique)
- Audio RMS was 9x quieter than PyTorch reference

### Root Cause
The PyTorch implementation uses `repetition_penalty=1.3` by default, which penalizes recently generated tokens to prevent the LM from getting stuck in repetitive loops. The CoreML port was missing this.

### Fix
Added repetition penalty to the LM decode loop in `Qwen3TtsSynthesizer.swift`:

```swift
// Apply repetition penalty (matching PyTorch default of 1.3)
let repetitionPenalty: Float = 1.3
let recentTokens = allCodebooks.suffix(20).map { $0[0] }  // Last 20 CB0 tokens
for token in recentTokens {
    if token < logits.count && logits[token] > 0 {
        logits[token] /= repetitionPenalty
    } else if token < logits.count {
        logits[token] *= repetitionPenalty
    }
}
```

### Results After Fix
- English: 57/58 unique CB0 (98%), natural EOS at frame 58
- Chinese: 64/65 unique CB0 (98%), natural EOS at frame 65
- Both transcribe correctly with Whisper

## Issue 2: Temperature/TopK Tuning *(superseded)*

### Original Values (from PyTorch defaults)
- Temperature: 0.9
- TopK: 50

### Adjusted Values (better quality)
- Temperature: 0.7
- TopK: 30

Lower temperature produces more deterministic, cleaner audio with less noise artifacts.

## Issue 3: Audio Post-Processing *(superseded)*

### Symptoms
- Raw audio had sibilance (harsh "s" sounds)
- Some high-frequency artifacts

### Fix
Added `AudioPostProcessor.applyTtsPostProcessing()` with:
- De-essing: -4.0 dB reduction
- Smoothing: enabled

## Verification *(superseded)*

### Spectral Comparison (English)
- Mel spectrogram cosine similarity: 93.7%
- MFCC cosine similarity: 94.2%

### Whisper Transcription
- English: "Hello world, this is a test of the text to speech system." ✓
- Chinese: "您好,世界,这是一个文字转语音系统的测试" ✓

---

# 6-Model Architecture Issues (Argmax-style pipeline rewrite)

After rewriting from the 5-model + numpy architecture to the 6-model Argmax-style CoreML pipeline, the following new issues were encountered and fixed.

## Issue 4: MultiCodeDecoder NaN Outputs

### Symptoms
- CB1-CB15 all decoded to token 2047 (max vocab index)
- `all_logits` output from MultiCodeDecoder was entirely NaN
- `hidden_states` output also NaN
- Audio was speech-like but unintelligible (Whisper transcribed as "I don't know")

### Root Cause 1: Compute Unit Sensitivity
The MultiCodeDecoder model produces NaN outputs on all compute unit configurations except `.cpuOnly`:

| Compute Units | all_logits NaN count | Result |
|---------------|---------------------|--------|
| `.all`        | 30720/30720         | All NaN |
| `.cpuAndGPU`  | 30720/30720         | All NaN |
| `.cpuAndNeuralEngine` | 30720/30720 | All NaN |
| `.cpuOnly`    | 0/30720             | Works  |

This is likely a float16 overflow issue in the GPU/ANE execution paths for this particular model's weights.

### Root Cause 2: MLMultiArray Non-Contiguous Stride Layout
Even with `.cpuOnly`, MCD still produced intermittent NaN when KV caches were created fresh via `MLMultiArray(shape:dataType:)`.

CoreML compiled models use specific non-contiguous memory stride layouts internally. For example:
- Shape `[1, 5120, 1, 16]` created fresh → strides `[81920, 16, 16, 1]` (contiguous)
- Shape `[1, 5120, 1, 16]` from model output → strides `[163840, 32, 32, 1]` (non-contiguous, 2x padding)

The model reads raw memory using its compiled stride layout. When given contiguous strides, it reads wrong memory offsets → garbage → NaN.

### Fix
1. Set MCD compute units to `.cpuOnly` in `Qwen3TtsModelStore.swift`
2. Added `getModelStridedKVCaches()` warmup prediction that runs a dummy inference to obtain KV cache arrays with the model's native stride layout, then zeros them for reuse:

```swift
private static func getModelStridedKVCaches(
    model: MLModel, kvLen: Int
) async throws -> (MLMultiArray, MLMultiArray) {
    // Run dummy prediction with zero inputs
    // ... (creates minimal inputs, runs prediction)
    let outKey = out.featureValue(for: "new_key_cache")!.multiArrayValue!
    let outVal = out.featureValue(for: "new_value_cache")!.multiArrayValue!
    // Zero the caches while preserving their stride layout
    for i in 0..<outKey.count { outKey[i] = NSNumber(value: Float(0.0)) }
    for i in 0..<outVal.count { outVal[i] = NSNumber(value: Float(0.0)) }
    return (outKey, outVal)
}
```

### Why CodeDecoder Doesn't Need This
The CodeDecoder's KV caches are created fresh with `MLMultiArray(shape:dataType:)` and work fine. This is likely because:
- CodeDecoder runs on `.cpuAndGPU` where the runtime handles stride translation
- The MCD's 5-layer architecture with `.cpuOnly` has stricter requirements on memory layout

## Issue 5: toFloat16() Broke CodeDecoder by Destroying Stride Layout

### Symptoms
- After adding a `toFloat16()` that always copied to a new contiguous array, CodeDecoder `hidden_states` became NaN
- MCD received NaN inputs → NaN outputs cascaded through the pipeline

### Root Cause
`toFloat16()` was creating new contiguous float16 arrays even for inputs that were already float16. This destroyed the non-contiguous stride layout that the CodeDecoder output had:
- CodeDecoder outputs `hidden_states` with shape `[1, 1024, 1, 1]` and strides `[32768, 32, 32, 1]`
- Copying to contiguous `[1024, 1, 1, 1]` strides scrambled the data when fed back as input

### Fix
Changed `toFloat16()` to return as-is for float16 inputs:

```swift
private static func toFloat16(_ array: MLMultiArray) throws -> MLMultiArray {
    if array.dataType == .float16 { return array }
    // Only copy when actually converting from float32
    let result = try MLMultiArray(shape: array.shape, dataType: .float16)
    for i in 0..<array.count { result[i] = array[i] }
    return result
}
```

## Issue 6: SpeechDecoder Raw Pointer Access Ignoring Strides

### Symptoms
- Audio output was garbled even when CB0-CB15 tokens were correct
- SpeechDecoder codes tensor had wrong values

### Root Cause
The codes tensor `[1, 16, 125]` was filled using raw `dataPointer.bindMemory(to: Int32.self)` which reads/writes linearly, but MLMultiArray may have non-contiguous strides.

### Fix
Switched to subscript access which correctly handles strides:

```swift
// Before (broken):
let codesPtr = codes.dataPointer.bindMemory(to: Int32.self, capacity: numCb * fixedLen)
codesPtr[cb * fixedLen + t] = Int32(frame[cb])

// After (fixed):
codes[cb * fixedLen + t] = NSNumber(value: Int32(frame[cb]))
```

## Issue 7: Token ID Mismatch Between Python and Swift

### Symptoms
- English audio was speech-like but wrong content
- 25 prefill embeddings in Python vs 26 in Swift (off by one)

### Root Cause
The hardcoded English token IDs in the Swift CLI didn't match the Python Qwen3 tokenizer output:
- Python tokenizer: 14 tokens `[9707, 1879, 11, 419, 374, 264, 1273, 315, 279, 1467, 311, 8806, 1849, 13]`
- Swift hardcoded: 15 tokens `[9707, 1879, 11, 419, 374, 264, 1273, 315, 279, 1467, 4686, 1331, 39586, 1849, 13]`

The divergence starts at "text to speech" — different BPE tokenization of the same text.

### Fix
Updated Swift to use the exact Python tokenizer output. This gave 25 prefill embeddings (matching Python) and produced correct speech.

## Verification (6-Model Architecture)

### Whisper Transcription
- English: "Hello World, this is a test of the text to speech system." ✓
- Chinese: "你好,世界,這是一個文字轉語音系統的測試" ✓

### Performance (M2, debug build)

**Before KV cache optimization:**
| Metric | English | Chinese |
|--------|---------|---------|
| Model load | 2.8s | 1.8s |
| Prefill | 3.5s (25 pos) | 1.5s (21 pos) |
| Decode | 95.9s (99 frames) | 37.0s (55 frames) |
| Frames/s | 1.0 | 1.5 |
| RTFx | 0.04x | 0.10x |
| Audio duration | 4.2s | 4.4s |
| Peak memory | 1.46 GB | 1.42 GB |

**After KV cache optimization (Issue 8):**
| Metric | Value |
|--------|-------|
| Decode | 40.1s (67 frames) |
| Frames/s | 1.7 |
| RTFx | 0.10x |
| Audio duration | 4.23s |
| Peak memory | 1.48 GB |

**Speedup**: 1.7x frames/s, 2.5x RTFx improvement

## Issue 8: MCD KV Cache Warmup Running Every Frame

### Symptoms
- Decode performance was 1.0 frames/s, RTFx 0.04x
- Each decode step took ~1s on M2

### Root Cause
The `getModelStridedKVCaches()` warmup prediction was called inside `runMultiCodeDecoder()`, which is invoked **every frame** (up to 125 times). This added a full MCD prediction per frame just to get properly-strided KV cache templates.

Analysis showed 34 CoreML predictions per frame:
- 1 warmup (getModelStridedKVCaches)
- 16 MCD loop positions
- 1 CodeEmbedder + 15 MultiCodeEmbedder
- 1 CodeDecoder

### Fix
Moved the warmup to run once for the first frame, then reused the model's OUTPUT KV caches (which have proper non-contiguous strides) as templates for subsequent frames. Zero them in-place before each use.

```swift
// Before: warmup every frame (inside runMultiCodeDecoder)
var (mcdKey, mcdVal) = try await getModelStridedKVCaches(model: model, kvLen: kvLen)

// After: warmup once, reuse model outputs
if let keyTemplate = kvKeyTemplate, let valTemplate = kvValTemplate {
    mcdKey = keyTemplate  // From previous frame's final position output
    mcdVal = valTemplate
    // Zero in-place (preserves stride layout)
    for i in 0..<mcdKey.count { mcdKey[i] = NSNumber(value: Float(0.0)) }
} else {
    // First frame only: run warmup
    (mcdKey, mcdVal) = try await getModelStridedKVCaches(model: model, kvLen: kvLen)
}
```

### Results
- Decode time: 95.9s → 40.1s for similar frame counts
- Frames/s: 1.0 → 1.7 (1.7x speedup)
- RTFx: 0.04x → 0.10x (2.5x improvement)

Eliminated 124 warmup predictions (1 per frame except first).

## Issue 9: MCD Model Float32 Outputs Blocking ANE (Performance Bottleneck)

### Symptoms
- RTFx still at 0.10x after KV cache optimization (need 10x more to reach real-time)
- MCD decode step is the slowest operation per frame

### Root Cause
CoreML profiler (`coreml-cli --fallback`) revealed:

**MultiCodeDecoder:**
- **0% ANE usage** (0/602 ops on ANE, all 602 on CPU)
- Main blocker: "Invalid output tensor format: fp32"
- 546 ops cannot run on ANE due to float32 outputs
- Est. CPU cost: 3.6ms per prediction (×17 predictions per frame = 61ms baseline)

**CodeDecoder (for comparison):**
- **93% ANE usage** (2716/2921 ops on ANE)
- Only 205 ops on CPU
- Runs efficiently on `.cpuAndGPU` compute units

### Fix Required
Reconvert the MCD model in mobius with **float16 outputs** instead of float32. This is the conversion script's responsibility, not fixable in Swift.

The model was likely converted with `--output-dtypes float32` or without specifying float16 precision for outputs. CoreML's ANE requires float16 for most operations.

### Expected Impact
If MCD achieves 93% ANE usage like CodeDecoder:
- 3.6ms CPU → ~0.4ms ANE+CPU per prediction
- 61ms → 6.8ms per frame (9x faster)
- RTFx: 0.10x → ~0.9x (close to real-time on M2)

Further optimizations needed to exceed 1.0x RTFx:
- Batch the 15 MultiCodeEmbedder calls (if model supports batching)
- Use vDSP/SIMD for element-wise embedding summation
- Release build instead of debug

---

## Key Learnings

1. **Repetition penalty is critical** - Without it, autoregressive LMs can get stuck in loops, especially for languages with different token distributions (Chinese)

2. **CB0 drives CB1-15** - When CB0 gets stuck, the code predictor (CB1-15) also produces repetitive patterns, leading to silent/broken audio

3. **Debug with token diversity metrics** - Monitoring unique CB0/CB1 counts and consecutive repeats quickly reveals stuck patterns

4. **Temperature sampling is required** - Greedy decoding (argmax) never produces EOS because codec tokens always have higher logits than EOS token

5. **CoreML MLMultiArray strides are non-contiguous** - Model outputs use padded stride layouts (e.g., `[32768, 32, 32, 1]` for shape `[1, 1024, 1, 1]`). Never use `dataPointer` for raw access — always use subscript `array[i]`. Never copy to contiguous arrays when feeding back to the same model.

6. **Warmup predictions for KV cache stride matching** - The only reliable way to get properly-strided KV caches for a CoreML model is to run a dummy prediction and reuse the output arrays (zeroed).

7. **Compute unit sensitivity varies per model** - Some models (CodeDecoder) work fine on `.cpuAndGPU`, while others (MultiCodeDecoder) require `.cpuOnly` due to float16 overflow in GPU/ANE paths. Test each model individually.
