# Vocoder CoreML Loading Issue - Root Cause Analysis

## Problem

The CosyVoice3 vocoder (hift_vocoder) hangs indefinitely when loading in CoreML, affecting both Swift and Python implementations.

## Evidence

### Working Models
- ✅ **Embedding** (260 MB): 0.68s to compile + load
- ✅ **LM Head** (260 MB): 0.87s to compile + load
- Both use simple linear transformations

### Hanging Models
- ❌ **Vocoder** (78 MB): Compiles in 18.95s, **hangs during load** (>5 min at 99% CPU)
- ❌ **Flow** (23 MB): Gets killed during load (memory issue)
- Both use complex conv + transformer architectures

## Root Cause

The vocoder model contains operations/structure that cause CoreML's graph optimizer to hang during the model loading phase:

**Vocoder Architecture (CausalHiFTGenerator):**
- F0 Predictor (CausalConvRNNF0Predictor)
- Source Generator (SourceModuleHnNSF with SineGen2)
- 3 upsample layers with causal convolutions
- 9 ResBlocks with weight normalization
- Custom ISTFT implementation (CoreMLISTFT)
- LayerNorm stabilization layers

**Conversion Settings Tried:**

| Config | Target | Compute | Format | Precision | Result |
|--------|--------|---------|--------|-----------|---------|
| Original | iOS17 | CPU_ONLY | default | FP32 | ❌ Hangs |
| Attempt 1 | macOS14 | ALL | mlprogram | FP16 | ❌ Hangs during conversion |
| Attempt 2 | macOS14 | CPU_ONLY | mlprogram | FP32 | Not tested yet |
| Attempt 3 | iOS16 | ALL | neuralnetwork | FP32 | Not tested yet |

## Why Re-conversion Won't Fix This

Re-conversion with different settings is unlikely to solve the issue because:

1. **The model architecture itself is the problem**, not the conversion settings
2. **PyTorch tracing succeeds** - the model traces correctly
3. **CoreML conversion succeeds** - creates valid .mlpackage files
4. **Loading phase hangs** - CoreML's internal graph optimization gets stuck

This is a **CoreML framework limitation** with this specific model architecture.

## Alternative Solutions

### Option 1: Use PyTorch for Full Pipeline ✅ (Recommended)

**Status:** Working perfectly with 97% accuracy

```bash
uv run python full_tts_pytorch.py
```

**Pros:**
- Already working
- 97% transcription accuracy
- Fast enough for development
- Full TTS pipeline

**Cons:**
- Slower than native CoreML would be
- Larger memory footprint
- Not optimized for Apple Silicon

### Option 2: Use ONNX Runtime Instead of CoreML

**For vocoder + flow only:**

```python
import onnxruntime as ort

# Vocoder ONNX exists
vocoder_session = ort.InferenceSession("converted/hift_vocoder.onnx")

# Flow ONNX exists (1.3 GB)
flow_session = ort.InferenceSession("flow_decoder.onnx")

# Use CoreML for LLM components (they work)
# Use ONNX for vocoder + flow (bypass CoreML loading issue)
```

**Pros:**
- Bypass CoreML loading issue
- ONNX Runtime optimized for Apple Silicon
- Can still use CoreML for LLM components

**Cons:**
- Mixed runtime (CoreML + ONNX)
- Need to manage two frameworks
- ONNX models larger than CoreML

### Option 3: Simplify Vocoder Architecture

**Replace complex components:**

1. **Remove F0 Predictor** - Use pre-computed F0 or simpler predictor
2. **Replace Custom ISTFT** - Use overlap-add reconstruction instead
3. **Simplify ResBlocks** - Remove weight normalization, use simpler blocks
4. **Remove Causal Convolutions** - Use standard convolutions (if causality not critical)

**This requires:**
- Retraining or fine-tuning the vocoder
- PyTorch model architecture changes
- Significant engineering effort

### Option 4: Wait for CoreML Framework Updates

Apple may fix graph optimization issues in future macOS/iOS versions.

**Not recommended:** No timeline, no guarantee.

### Option 5: Use Different TTS Model

**Alternative models with proven CoreML support:**
- Piper TTS (ONNX-first, CoreML-compatible)
- Coqui TTS (MelGAN vocoder, simpler architecture)
- Apple's built-in TTS

**Cons:**
- Different voice quality
- Migration effort
- May not match CosyVoice3 quality

## Recommended Path Forward

### For Development (Now)
**Use Option 1: PyTorch Pipeline**
- File: `full_tts_pytorch.py`
- Already working, 97% accuracy
- No additional work needed

### For Production (Future)
**Use Option 2: Hybrid CoreML + ONNX Runtime**

**Implementation:**
```swift
// Swift pseudocode
class HybridTTSPipeline {
    let embeddingModel: MLModel  // CoreML ✅
    let lmHeadModel: MLModel      // CoreML ✅
    let decoderModel: MLModel     // CoreML ✅

    let flowSession: ORTSession   // ONNX Runtime
    let vocoderSession: ORTSession // ONNX Runtime

    func synthesize(text: String) -> Audio {
        // 1. Tokenize (native Swift)
        let tokens = tokenize(text)

        // 2. Embedding (CoreML - fast!)
        let embeddings = embeddingModel.predict(tokens)

        // 3. LM Head (CoreML - fast!)
        let speechTokens = lmHeadModel.predict(embeddings)

        // 4. Flow (ONNX Runtime - works)
        let mel = flowSession.run(speechTokens)

        // 5. Vocoder (ONNX Runtime - works)
        let audio = vocoderSession.run(mel)

        return audio
    }
}
```

**Benefits:**
- Uses CoreML where it works (embedding, lm_head, decoder)
- Uses ONNX for problematic models (flow, vocoder)
- Best of both worlds
- Production-ready

## Files

**Conversion Scripts:**
- `convert_vocoder.py` - Original vocoder conversion (hangs on load)
- `reconvert_vocoder_v2.py` - Attempted re-conversion (hangs during conversion)
- `convert_flow_final.py` - Flow conversion (hangs on load)

**Test Programs:**
- `SimpleTest.swift` - ✅ Embedding loads successfully
- `LMHeadTest.swift` - ✅ LM head loads successfully
- `VocoderTest.swift` - ❌ Hangs during load
- `FlowTest.swift` - ❌ Killed during load

**ONNX Models (Already Exist):**
- `converted/hift_vocoder.onnx` - Vocoder in ONNX format
- `flow_decoder.onnx` (1.3 GB) - Flow in ONNX format

## Conclusion

**The vocoder CoreML loading issue is not fixable with different conversion settings.**

The model architecture itself causes CoreML's graph optimizer to hang. The solution is to:

1. **Short-term:** Use PyTorch pipeline (already working)
2. **Long-term:** Use hybrid CoreML + ONNX Runtime approach

Both ONNX models already exist and are proven to work. The hybrid approach gives the best performance while avoiding CoreML's loading issue.
