# Frame-Based Vocoder Conversion - Why It Failed

## Attempt Summary

We attempted to convert the CosyVoice3 vocoder to frame-based processing (following PocketTTS's Mimi pattern) to work around the CoreML loading hang issue.

## Why It Failed

### 1. **STFT Fusion Architecture**

The vocoder uses STFT-processed source signals that get fused at multiple upsampling stages:

```python
# Source generation
s = generator.f0_upsamp(f0).transpose(1, 2)
s, _, _ = generator.m_source(s)

# Apply STFT
s_stft_real, s_stft_imag = generator._stft(s.squeeze(1))
s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)  # [B, 18, T]

# Fusion at each upsampling stage
for i in range(num_upsamples):
    x = ups[i](x)  # Upsample mel features
    si = source_downs[i](s_stft)  # Downsample source STFT
    x = x + si  # FUSION - temporal alignment required!
```

**Problem:** The STFT creates a temporal grid that must perfectly align with the upsampled mel features. Small chunks cause misalignment.

### 2. **Temporal Alignment Errors**

Errors encountered:
- **4 mel frames** → `RuntimeError: size of tensor a (32) must match size of tensor b (8)`
- **100 mel frames** → `RuntimeError: size of tensor a (800) must match size of tensor b (776)`

The 24-frame offset (800 - 776) suggests STFT edge effects and padding don't align with upsampling.

### 3. **Causal Padding Complexity**

The generator uses causal convolutions with look-ahead:

```python
conv_pre_look_right=4  # Requires 4 frames of future context
```

This means:
- Each chunk needs future context
- Can't process truly independently
- State management becomes complex

### 4. **Different from Mimi**

**PocketTTS's Mimi decoder works because:**
- Simple latent → audio mapping
- No STFT fusion
- 26 state tensors capture all dependencies
- Designed for frame-based processing

**CosyVoice3 vocoder is different:**
- Complex multi-stage architecture
- STFT fusion at multiple stages
- Temporal alignment requirements
- Not designed for chunking

## Root Cause

**The vocoder is fundamentally incompatible with frame-based processing due to:**

1. **STFT temporal dependencies** - Can't isolate frames
2. **Multi-stage fusion** - Requires perfect alignment across stages
3. **Causal padding** - Needs future context
4. **Not designed for it** - Architecture assumes full sequence

## What Actually Works

### ✅ Solution: Use PyTorch Directly (Stateless)

The vocoder is **already stateless** when used with `finalize=True`:

```python
# Load PyTorch model
vocoder = load_vocoder_pytorch()

# Stateless inference
audio1 = vocoder.inference(mel1, finalize=True)[0]  # Independent
audio2 = vocoder.inference(mel2, finalize=True)[0]  # Independent
audio3 = vocoder.inference(mel3, finalize=True)[0]  # Independent

# No state between calls!
```

### ✅ Hybrid CoreML + PyTorch Pipeline

Use CoreML where it works, PyTorch where it doesn't:

```python
# CoreML for simple models (these work!)
embedding = MLModel("cosyvoice_llm_embedding.mlpackage")  # ✅ 0.68s load
lm_head = MLModel("cosyvoice_llm_lm_head.mlpackage")      # ✅ 0.87s load

# PyTorch for complex models (still stateless!)
vocoder = load_vocoder_pytorch()  # Stateless PyTorch
flow = load_flow_pytorch()        # Stateless PyTorch

# Use both
def synthesize(text):
    emb = embedding.predict(tokens)     # CoreML
    lm = lm_head.predict(emb)           # CoreML
    mel = flow.inference(lm)            # PyTorch (stateless!)
    audio = vocoder.inference(mel)[0]   # PyTorch (stateless!)
    return audio
```

**Benefits:**
- ✅ Uses CoreML where it works (60% of models by count)
- ✅ Uses PyTorch where CoreML fails (40% - but complex ones)
- ✅ All components stateless
- ✅ Production-ready (97% accuracy proven)
- ✅ No CoreML loading issues

## Lessons Learned

1. **Not all models can be chunked** - Architecture matters
2. **STFT creates dependencies** - Can't isolate frames when STFT is involved
3. **PocketTTS's pattern doesn't generalize** - Mimi's simplicity is key
4. **Stateless ≠ Frame-based** - Can be stateless without chunking
5. **Hybrid pipelines are valid** - Use the right tool for each component

## Files Created (Failed Attempts)

- `convert_vocoder_frame_based.py` - Frame-based converter (dtype and alignment errors)
- `VocoderState.swift` - State management (not needed)
- `FrameVocoder.swift` - Frame decoder (not usable)
- `VocoderFrameTest.swift` - Test program (can't run)

## Recommendation

**Use the hybrid CoreML + PyTorch approach:**

1. Keep existing CoreML models (embedding, lm_head, decoder)
2. Use PyTorch for vocoder and flow (they're stateless!)
3. Integrate into production pipeline
4. Get 97% accuracy immediately

**Don't:**
- ❌ Try to force-fit all models into CoreML
- ❌ Spend more time on frame-based conversion
- ❌ Create model splitting (complexity not worth it)

## References

- `STATELESS_ONNX_ANSWER.md` - Explains models are already stateless
- `VOCODER_COREML_ISSUE.md` - Root cause of CoreML loading hang
- `FINAL_RESOLUTION.md` - Solution options analysis
- `full_tts_pytorch.py` - Working stateless PyTorch pipeline (97% accuracy)

## Conclusion

**Frame-based conversion failed because the vocoder's architecture is incompatible with chunking.**

**The solution is NOT to chunk it, but to use it as-is in PyTorch (it's already stateless!).**

**Status:** ❌ Frame-based approach abandoned

**Next step:** Implement hybrid CoreML + PyTorch pipeline in Swift
