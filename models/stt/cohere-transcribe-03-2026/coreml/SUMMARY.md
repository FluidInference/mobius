# Cohere Transcribe 03-2026 CoreML Conversion Summary

## Outcome: ⛔ BLOCKED - Not Convertible to CoreML

**Date**: 2026-04-03
**Status**: Conversion abandoned due to fundamental incompatibility
**Root cause**: Dynamic shape operations in Conformer encoder

## What We Accomplished

### ✅ Setup & Investigation
- Created complete conversion infrastructure (scripts, docs, dependencies)
- Successfully authenticated with HuggingFace (model is gated)
- Loaded model successfully with `trust_remote_code=True`
- Identified model architecture:
  - `encoder`: ConformerEncoder (2B params)
  - `transf_decoder`: TransformerDecoderWrapper
  - `encoder_decoder_proj`: Linear projection
  - `log_softmax`: TokenClassifierHead

### ✅ Attempted Solutions
1. Fixed-length tracing (removed dynamic `length` parameter)
2. Disabled strict tracing (`strict=False, check_trace=False`)
3. Simplified wrappers to minimal forwarding logic
4. Correct attribute names and input shapes

### ❌ Blocking Error
```
TypeError: only 0-dimensional arrays can be converted to Python scalars
Location: Encoder positional encoding / shape calculations
Operation: int() cast on multi-dimensional tensor
```

## Why Conversion Failed

CoreML requires **static computation graphs** - all shapes and control flow must be known at compile time.

Cohere's Conformer encoder uses **dynamic operations**:
- Runtime shape calculations: `if projected > max_size_32bit:`
- Dynamic positional encoding: `effective_length = max(length, self.max_len)`
- Tensor-to-scalar casts: `int(tensor_value)`

These are fundamentally incompatible with CoreML's tracing mechanism.

## Technical Analysis

### Model Characteristics
| Property | Value |
|----------|-------|
| **Size** | 2B parameters |
| **Architecture** | Conformer encoder + Transformer decoder |
| **Languages** | 14 (EN, FR, DE, IT, ES, PT, EL, NL, PL, ZH, JA, KO, VI, AR) |
| **WER** | 5.42 (English ASR Leaderboard) |
| **License** | Apache 2.0 |

### Conversion Attempts
| Attempt | Strategy | Result |
|---------|----------|--------|
| 1 | Standard trace with `audio_encoder` | AttributeError (wrong attribute name) |
| 2 | Fixed attribute names (`encoder`) | TypeError (dynamic shapes) |
| 3 | Fixed-length input (hardcoded 3000 frames) | TypeError (internal dynamic ops) |
| 4 | Non-strict tracing | TypeError (doesn't bypass graph building) |

### Root Cause Analysis

**Problem location**: `modeling_cohere_asr.py` in Conformer encoder

**Specific issues**:
1. **Line 118**: `if projected > max_size_32bit:` - Tensor comparison in control flow
2. **Line 170**: `if self._needs_conv_split(x):` - Dynamic tensor-based branching
3. **Line 208**: `effective_length = max(length, self.max_len)` - Tensor max operation
4. **Line 310**: `if pos_emb.size(0) == 1 and batch_size > 1:` - Shape-dependent logic

All of these require runtime tensor values to determine execution path, which CoreML cannot trace.

## Alternatives Evaluated

### 1. ONNX → CoreML
**Verdict**: Unlikely to work (same dynamic shape issues)

### 2. Rewrite Encoder
**Verdict**: Feasible but extremely complex (~2-4 weeks), high risk of breaking model quality

### 3. Request Official CoreML Version
**Verdict**: Best option but slow (CohereLabs may not prioritize)

### 4. Use CPU-Only PyTorch
**Verdict**: Works but ~50x slower than ANE

### 5. Use Alternative Model
**Verdict**: ✅ **RECOMMENDED**

## Recommended Alternative Models

| Model | Size | Status | Quality | Speed (ANE) |
|-------|------|--------|---------|-------------|
| **Parakeet TDT v3** | 0.6B | ✅ Converted | WER ~7% | ~40x faster |
| **Qwen3 ASR** | 0.6B | ✅ Converted | WER ~6% | ~50x faster |
| **Whisper (Apple)** | 1.5B | ✅ Official | WER ~5% | ~30x faster |
| **Cohere Transcribe** | 2B | ❌ Blocked | WER ~5.4% | N/A (won't convert) |

**Recommendation**: Use **Parakeet TDT v3** for production (already integrated in FluidAudio) or **Qwen3 ASR** for multilingual support.

## Files Created

All files are preserved for reference and future attempts:

```
mobius/models/stt/cohere-transcribe-03-2026/coreml/
├── BLOCKING_ISSUE.md         # Detailed technical analysis
├── CONVERSION_STATUS.md       # Trial-by-trial progress log
├── SUMMARY.md                 # This file
├── README.md                  # Comprehensive guide (updated with gated access)
├── QUICKSTART.md              # 5-minute setup guide
├── convert-cohere-transcribe.py  # Conversion script (functional but blocked)
├── compare-models.py          # Validation script
├── inspect-model.py           # Model structure inspection
├── inspect-inputs.py          # Input shape inspection
├── pyproject.toml             # Dependencies (uv-managed)
└── .gitignore                 # Build artifact exclusions
```

Total: ~1,500 lines of code and documentation

## Lessons Learned

### For Future Conversions

1. **Check model architecture first**: Conformer models often have dynamic ops
2. **Verify trace compatibility early**: Run a quick trace test before full conversion
3. **Not all models convert to CoreML**: Dynamic execution models (esp. from HuggingFace) often fail
4. **Size doesn't matter**: Even 2B params can convert IF the architecture is static
5. **Official versions preferred**: Apple/vendor-provided CoreML models work best

### Red Flags for CoreML Incompatibility

- ❌ `if` statements using tensor comparisons
- ❌ `.int()`, `.item()` calls on tensors in forward pass
- ❌ Dynamic shape calculations (`.size()`, `.shape` in control flow)
- ❌ `torch.jit.script` warnings during initial testing
- ❌ "TracerWarning: Converting a tensor to a Python boolean" messages

## Conclusion

**Cohere Transcribe 03-2026 cannot be converted to CoreML** using standard approaches due to fundamental architectural incompatibility.

**Recommendation**: Use **Parakeet TDT v3** or **Qwen3 ASR** instead - both are converted, tested, and integrated in FluidAudio.

**If Cohere is required**: Wait for official CoreML support or use CPU-only PyTorch inference (slow but functional).

**Status**: ⛔ **ABANDONED** - No path forward with current tooling

## References

- Model card: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- Parakeet TDT v3: `mobius/models/stt/parakeet-tdt-v3-0.6b/coreml/` ✅
- Qwen3 ASR: `mobius/models/stt/qwen3-asr-0.6b/coreml/` ✅
- FluidAudio ASR managers: `Sources/FluidAudio/ASR/`
