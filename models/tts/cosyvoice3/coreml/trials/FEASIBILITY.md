# CosyVoice3 CoreML Conversion Feasibility Assessment

Date: 2026-04-09
Status: **BLOCKED** - Significant challenges identified

---

## Executive Summary

CosyVoice3 CoreML conversion is **significantly more complex** than initially anticipated. Multiple blockers identified:

1. **Missing LLM implementation** - 508M param component not in ONNX format
2. **ONNX conversion tools incompatible** - onnx-coreml deprecated, doesn't work with modern coremltools
3. **Incomplete model artifacts** - ONNX files are only partial components, not full pipeline
4. **No reference implementation** - Need to reverse-engineer the complete inference pipeline
5. **Size constraints** - 868M total params may exceed ANE capabilities

## Detailed Analysis

### Model Component Status

| Component | Size | Format | CoreML Ready? | Blocker |
|-----------|------|--------|---------------|---------|
| LLM | 508M | PyTorch (.pt) | ❌ No | No ONNX export, architecture unknown |
| Flow DiT | 87M | ONNX (.onnx) | 🟡 Maybe | onnx-coreml broken, need alternative |
| Full Flow | 332M | PyTorch (.pt) | ❌ No | Contains DiT + wrapper, arch unknown |
| Vocoder | 21M | PyTorch (.pt) | 🟡 Maybe | HiFi-GAN variant, convertible but need arch |
| Speaker Embed | 7M | ONNX (.onnx) | 🟡 Maybe | Same ONNX issue |
| Tokenizer | ? | ONNX (.onnx) | 🟡 Maybe | Same ONNX issue |

### Technical Blockers

#### 1. ONNX Conversion Tooling

**Problem**: `onnx-coreml` package is deprecated and incompatible with coremltools 8.0+

```python
# This fails:
from onnx_coreml import convert
# Error: ModuleNotFoundError: No module named 'coremltools.converters.nnssa'
```

**Alternatives**:
- Convert ONNX → PyTorch → CoreML (requires onnx2pytorch or manual reconstruction)
- Use older coremltools version (not recommended, loses features)
- Manually reconstruct models in PyTorch from ONNX graph

#### 2. LLM Component Missing

**Problem**: The core LLM (508M params) is only available as `llm.pt` PyTorch checkpoint

**Unknowns**:
- What is the model architecture? (CosyVoice3LM)
- How to load the checkpoint?
- What are the input/output specifications?
- Can it be exported to ONNX?

**Investigation needed**:
- Find CosyVoice3 GitHub repo
- Locate model definition files
- Understand inference pipeline
- Test PyTorch inference before attempting conversion

#### 3. Incomplete Pipeline

**Problem**: ONNX files are isolated components, not a complete TTS system

The full pipeline requires:
```
Text → [Text Preprocessing?] → [LLM] → Tokens → [Flow Decoder] → Latent → [Vocoder] → Audio
                                 ↑                      ↑                         ↑
                             Missing              ONNX partial              PyTorch only
```

**Missing pieces**:
- Text normalization (CosyVoice-ttsfrd mentioned in docs)
- LLM wrapper and orchestration
- Token embedding layers
- Flow model wrapper (have decoder, need full pipeline)
- Reference audio → speaker embedding pipeline

#### 4. Size and Performance Concerns

**Total model size**: ~868M parameters (not 500M as advertised)

**Breakdown**:
- LLM: 508M (54% of total)
- Flow: 332M (38% of total)
- Vocoder: 21M (2% of total)
- Others: 7M (6% of total)

**ANE Compatibility Risks**:
- ✅ Vocoder (21M, Conv-based) - likely OK
- ⚠️ Flow DiT (87M decoder, 22 transformer blocks) - attention ops may fall back to CPU
- ❌ LLM (508M, autoregressive) - likely CPU/GPU only, too large for ANE

**Memory estimates** (FP32):
- Development: ~3.5 GB
- FP16: ~1.75 GB
- W8A16: ~1 GB (optimistic)

### Comparison to Qwen3-TTS

We have working conversion scripts for Qwen3-TTS. Why is CosyVoice3 harder?

| Aspect | Qwen3-TTS | CosyVoice3 |
|--------|-----------|------------|
| **Total params** | 1.7B | 868M |
| **ONNX availability** | ❌ None | 🟡 Partial (3/5 components) |
| **Reference CoreML** | ✅ TTSKit models | ❌ None |
| **Architecure** | Dual-track LM | LLM + Flow + Vocoder |
| **Conversion strategy** | 6-model split, W8A16 | Unknown |
| **Documentation** | Good (we wrote it) | Minimal |
| **Our status** | ✅ Working | ❌ Blocked |

**Key difference**: Qwen3-TTS had a **reference CoreML implementation** (TTSKit) that we reverse-engineered. CosyVoice3 has no such reference.

## Recommended Actions

### Option 1: Pause and Research (RECOMMENDED)

**Before continuing**, we need:

1. **Find official implementation**:
   - GitHub repo: `FunAudioLLM/CosyVoice`
   - Model architecture definitions
   - Inference example code

2. **Test PyTorch inference**:
   - Load all checkpoints
   - Run end-to-end generation
   - Understand component interactions

3. **Assess true complexity**:
   - Can LLM be exported to ONNX?
   - What's the minimum viable component set?
   - What's the realistic timeline?

**Estimated research time**: 4-8 hours

### Option 2: Convert Individual Components

**Incremental approach**:

1. ✅ **Vocoder first** (easiest):
   - Reconstruct HiFi-GAN architecture from `hift.pt` checkpoint
   - Similar to KittenTTS conversion we already have
   - ~21M params, should be ANE-compatible
   - **ETA**: 2-4 hours

2. 🟡 **Speaker embedding**:
   - Solve ONNX → CoreML conversion issue
   - Either: Use onnx2pytorch → CoreML, or manually reconstruct
   - **ETA**: 1-2 hours

3. ⚠️ **Flow decoder**:
   - Same ONNX issue as speaker embedding
   - 87M params, complex DiT architecture
   - **ETA**: 4-6 hours

4. ❌ **LLM** (hardest, may not be feasible):
   - Find architecture definition
   - Load checkpoint
   - Export to ONNX (if possible)
   - Convert to CoreML
   - **ETA**: Unknown, possibly days

**Total estimated time**: 2-4 days minimum (if LLM is feasible)

### Option 3: Use Alternative Model

**Consider simpler alternatives**:

1. **Qwen3-TTS** (we already have this working!)
   - 1.7B params but optimized 6-model architecture
   - ~1 GB total size with W8A16
   - 97ms latency
   - Already converted and tested

2. **Kokoro-82M**:
   - Already converted to CoreML
   - 82M params total
   - Proven ANE compatibility
   - Simpler architecture

3. **PocketTTS**:
   - Already in mobius
   - Streaming capable
   - Smaller model

**Question for user**: Why CosyVoice3 specifically? What features do you need that other models don't have?

## Conclusion

**CosyVoice3 CoreML conversion is feasible but HIGH EFFORT**:

- ✅ **Technically possible**: All components can theoretically be converted
- ⚠️ **Time-intensive**: Estimated 2-4 days minimum, possibly longer
- ❌ **No guarantees**: LLM (508M params) may not run efficiently on ANE
- ⚠️ **Incomplete information**: Need to reverse-engineer full pipeline

**Recommendation**:

1. **Pause conversion work**
2. **Research CosyVoice3 implementation** (find GitHub repo, test PyTorch inference)
3. **Clarify requirements** (why this model vs alternatives?)
4. **Re-assess feasibility** with complete information

**Alternative**: If you just need high-quality multilingual TTS, consider using Qwen3-TTS (already working) or waiting for a reference CoreML implementation of CosyVoice3 to appear.

---

## What We've Accomplished

Despite blockers, we made progress:

✅ Set up conversion environment
✅ Downloaded and analyzed all model files
✅ Documented architecture (868M params, 5 components)
✅ Identified ONNX models available
✅ Identified blockers (ONNX tools, missing LLM, incomplete pipeline)
✅ Created analysis scripts for future work
✅ Documented findings in TRIALS.md

**Files created**:
- `/mobius/models/tts/cosyvoice3/coreml/` - conversion directory
- `analyze_model.py` - model inspection tool
- `analyze_all_models.py` - comprehensive analysis
- `convert_onnx_models.py` - ONNX conversion attempt (blocked)
- `TRIALS.md` - detailed conversion log
- `FEASIBILITY.md` - this document

---

**Next step**: Await user decision on how to proceed.
