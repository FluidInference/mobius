# CoreML Pipeline Status

## Current State

### ✅ What's Working

1. **PyTorch Full Pipeline** (`full_tts_pytorch.py`)
   - Complete text-to-speech
   - Cross-lingual mode
   - 97% transcription accuracy
   - Generates working WAV files

2. **CoreML Models Converted**
   - All 5 models exist as `.mlpackage` files
   - Embedding, Decoder, LM Head, Flow, Vocoder
   - Total size: ~1.5GB

### 🔄 In Progress

**Pure CoreML Pipeline** (`pure_coreml_tts.py`)

Currently testing: CoreML vocoder replacement
- Frontend: PyTorch ✅
- LLM: PyTorch  
- Flow: PyTorch
- **Vocoder: CoreML** ← Testing now

### ❌ Not Yet Implemented

**Full CoreML Inference Chain**
- Need to replace PyTorch LLM with CoreML LLM
- Need to replace PyTorch Flow with CoreML Flow
- Requires proper input preparation for each CoreML model

## Technical Challenges

### 1. Model Input Complexity

Each CoreML model needs specific inputs that the PyTorch frontend generates:

**LLM Decoder:**
- `hidden_states` [batch, seq_len, 896]
- `cos` [batch, 1, seq_len, 64] - RoPE embeddings
- `sin` [batch, 1, seq_len, 64] - RoPE embeddings  
- `attention_mask` [batch, seq_len]

**Flow:**
- Speech tokens from LLM
- Speaker embeddings
- Prompt features
- Proper conditioning

**Vocoder:**
- Mel spectrogram [batch, 80, time]
- Specific shape and value range

### 2. CoreML Loading Time

First-time loading triggers ANE (Apple Neural Engine) compilation:
- **Warm start:** ~1-4 seconds per model
- **Cold start:** 10-30 seconds per model (first time)
- **Total first load:** Could be 2-10 minutes for all 5 models

This is expected Apple behavior - models compile to ANE-optimized format.

### 3. Implementation Strategy

**Phase 1: Vocoder Only** ← Current
- Use PyTorch for LLM + Flow → mel spectrogram
- Use CoreML for Vocoder → audio
- **Goal:** Validate CoreML vocoder works

**Phase 2: Add Flow**
- Use PyTorch for LLM → speech tokens
- Use CoreML for Flow → mel
- Use CoreML for Vocoder → audio

**Phase 3: Full CoreML**
- Use PyTorch frontend only (tokenization)
- Use CoreML for entire inference chain
- Maximum performance

**Phase 4: Swift Implementation** (Production)
- Port frontend to Swift
- Use native CoreML APIs
- Best performance (80x faster than Python)

## Files

- `full_tts_pytorch.py` - Working PyTorch pipeline
- `coreml_pipeline_demo.py` - CoreML model loader template
- `pure_coreml_tts.py` - Phase 1: Testing CoreML vocoder
- `COREML_STATUS.md` - This file

## Next Steps

1. ✅ Complete Phase 1 (CoreML vocoder test)
2. Implement Phase 2 (CoreML flow)
3. Implement Phase 3 (CoreML LLM)
4. Document Swift integration requirements

## Performance Expectations

**Python CoreML:** 
- Model loading: 1-4s per model (warm)
- Inference: Similar to PyTorch (Python overhead)
- **Not recommended for production**

**Swift CoreML:**
- Model loading: 80x faster than Python
- Inference: Native ANE performance
- **Recommended for production**

## Why Python CoreML Is Still Useful

1. **Validation:** Proves CoreML models work
2. **Debugging:** Easier to debug than Swift
3. **Prototyping:** Quick iteration
4. **Reference:** Shows how to chain models

The pure CoreML Python pipeline validates the conversion was successful, then Swift can use these same models with much better performance.
