# Swift Benchmark Status

## Summary

Attempted to run WER benchmark on the Swift FluidAudio implementation. Discovered and fixed several issues, but full benchmark requires model interface alignment.

## Issues Found & Fixed ✅

### 1. Mel Padding Shape Mismatch
**Error**: `MultiArray shape (1 x 128 x 3001) does not match the shape (1 x 128 x 3500)`

**Root cause**: Code was padding to 3001 frames, but encoder model expects 3500 frames

**Files fixed**:
- `CohereAsrManager.swift`: All `3001` → `3500`
- `CohereStatelessManager.swift`: All `3001` → `3500`

**Commit**: `c0fa58ffc`

### 2. Encoder Output Name Mismatch
**Error**: `encodingFailed("Failed to get encoder output")`

**Root cause**: Code was looking for output named `"encoder_outputs"`, but encoder actually outputs `"hidden_states"`

**Fix**: Changed `encoderOutput.featureValue(for: "encoder_outputs")` → `"hidden_states"`

**Verification**: Matches the encoder export script in mobius:
```python
# mobius/models/stt/cohere-transcribe-03-2026/coreml/exports/export-encoder.py:116
outputs=[ct.TensorType(name="hidden_states")]
```

**Commit**: `c0fa58ffc`

### 3. Closure Capture - Explicit Self
**Error**: Compilation error requiring explicit `self` in closure

**Fix**: `maxSeqLen` → `self.maxSeqLen` in CohereStatelessManager

**Commit**: `e42955d23` (initial), refined in `c0fa58ffc`

## Remaining Issue ⚠️

### Decoder Interface Mismatch

**Error**: `Feature position_ids is required but not specified`

**Root cause**: The auto-downloaded `cohere_decoder_stateful.mlpackage` has a different interface than what `CohereAsrManager` expects.

**Downloaded decoder expects**:
- `position_ids` (plural)
- Different cache interface (stateful CoreML API)
- Designed for macOS 15+ with CoreML state management

**CohereAsrManager provides**:
- `position_id` (singular)
- External cache management (`cache_k`, `cache_v`)
- Designed for macOS 14+

**Cache-external decoder** (what we built in mobius):
- `position_id` (singular)
- 16 separate cache inputs/outputs (Parakeet pattern)
- Works on macOS 14+
- **Not in the auto-downloaded models**

## Model Compatibility Matrix

| Model | Interface | macOS Version | Status |
|-------|-----------|---------------|--------|
| **Encoder** | ✅ Compatible | 14+ | Working |
| **Decoder (stateful, auto-download)** | ❌ Interface mismatch | 15+ | Incompatible with current code |
| **Decoder (cache-external, mobius)** | ✅ Compatible | 14+ | Not auto-downloaded |

## Next Steps to Run Full Benchmark

### Option 1: Use Cache-External Decoder (Recommended)

1. Copy cache-external decoder to FluidAudio models cache:
```bash
cp mobius/models/stt/cohere-transcribe-03-2026/coreml/build-test/cohere_decoder_cache_external.mlmodelc \
   ~/Library/Application\ Support/FluidAudio/Models/cohere-transcribe/q8/
```

2. Update `CohereAsrModels.swift` to use cache-external decoder:
```swift
// Change from:
named: ModelNames.CohereTranscribe.decoderStateful
// To:
named: "cohere_decoder_cache_external"
```

3. Update `CohereAsrManager.swift` to use cache-external interface:
   - Use `CohereDecoderState` and `CohereModelInference`
   - Remove old stateful cache code
   - Use correct EOS token (3, not 151643)

### Option 2: Update Code for Stateful Decoder

1. Fix decoder input parameters:
   - `position_id` → `position_ids`
   - Use CoreML state API instead of manual cache passing

2. Update to macOS 15+ deployment target

3. Test with stateful decoder

## Test Results from Python (Mobius)

**Baseline WER** (cache-external decoder, EOS token fixed):
- **11.95% WER** on 10 LibriSpeech test-clean samples
- 2/10 samples achieved perfect 0.00% WER
- Most errors are punctuation differences

This is the target to match in Swift once models are aligned.

## Files Modified

### FluidAudio Repository
```
Sources/FluidAudio/ASR/Cohere/
├── CohereAsrManager.swift          ✅ Fixed (mel padding, encoder output)
├── CohereStatelessManager.swift    ✅ Fixed (mel padding, EOS token, self capture)
├── CohereDecoderState.swift        ✅ Added (cache-external support)
├── CohereModelInference.swift      ✅ Added (cache-external support)
└── CohereAsrModels.swift           ⚠️ Needs update for cache-external
```

### Commits
- `e42955d23` - Add cache-external decoder with correct EOS token
- `c0fa58ffc` - Fix mel padding, encoder output, closure capture

## Recommendations

1. **Immediate**: Use cache-external decoder for benchmarks (matches Python results)
2. **Short-term**: Update `CohereAsrManager` to use `CohereDecoderState` + `CohereModelInference`
3. **Long-term**: Decide on single decoder strategy (cache-external vs stateful)

## Status

**Python/Mobius**: ✅ Working, 11.95% WER achieved
**Swift/FluidAudio**: ⚠️ Partially fixed, needs decoder alignment for full benchmark

---

**Last updated**: April 8, 2026
**Branch**: `feat/cohere-transcribe-int8-integration`
**Commits**: `e42955d23`, `c0fa58ffc`
