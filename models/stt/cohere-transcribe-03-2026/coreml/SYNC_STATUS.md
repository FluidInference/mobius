# Sync Status: Mobius ↔ FluidAudio

Both repositories have been updated with the EOS token fix and cache-external decoder implementation.

## Mobius Repository ✅

**Branch**: `docs/cohere-transcribe-coreml-decoder-fix`
**Remote**: `https://github.com/FluidInference/mobius.git`

### Commits Pushed

1. **`5d12a80`** - Fix EOS token detection in cache-external decoder
   - Fixed EOS_TOKEN from 151643 to 3
   - WER improved from 29.88% to 11.95% (60% improvement)
   - All test scripts updated
   - Documentation updated

2. **`e007570`** - Verify .mlmodelc compilation for Swift integration
   - Compiled to .mlmodelc format
   - Swift test verified model loads and runs
   - WER consistency test passed (11.29% on 3 samples)
   - Documentation: MLMODELC_VERIFIED.md

### Files Modified/Added

```
models/stt/cohere-transcribe-03-2026/coreml/
├── test-wer-hybrid.py                         ✅ EOS_TOKEN: 151643 → 3
├── test-debug-tokens.py                       ✅ EOS_TOKEN: 151643 → 3
├── test-wer-cache-external.py                 ✅ EOS_TOKEN: 151643 → 3
├── test-mlmodelc.swift                        ✅ New: Swift .mlmodelc test
├── test-wer-mlmodelc.py                       ✅ New: WER test for compiled
├── CACHE_EXTERNAL_DELIVERED.md                ✅ Updated with results
├── MLMODELC_VERIFIED.md                       ✅ New: Compilation guide
└── librispeech_test_samples/
    └── wer_results_cache_external.json        ✅ Updated: 11.95% WER
```

---

## FluidAudio Repository ✅

**Branch**: `feat/cohere-transcribe-int8-integration`
**Remote**: `https://github.com/FluidInference/FluidAudio.git`

### Commit Pushed

**`e42955d23`** - Add Cohere cache-external decoder support with correct EOS token

### Files Added

```
Sources/FluidAudio/ASR/Cohere/
├── CohereDecoderState.swift          ✅ KV cache state management
├── CohereModelInference.swift        ✅ Decoder execution helper
└── CohereStatelessManager.swift      ✅ Stateless decoder (EOS fixed: 3)
```

### Key Implementation Details

**CohereDecoderState.swift**:
- Manages 16 KV cache arrays (8 layers × K/V)
- Each cache: [1, 8, 108, 128]
- Updates from decoder output each step
- Parakeet pattern: cache passed in/out

**CohereModelInference.swift**:
- Executes decoder with cache-external pattern
- Growing attention mask: [1,1,1,1] → [1,1,1,108]
- Greedy sampling from logits
- Returns (logits, updated_state)

**CohereStatelessManager.swift**:
- Fixed `eosTokenId = 3` (was 151643)
- Stateless O(n²) decoder alternative
- Simpler than cache management
- Also works with correct EOS token

---

## Critical Fix Applied to Both Repos

### The Bug
```python
# WRONG (in all Python test scripts)
EOS_TOKEN = 151643  # Token doesn't exist! Vocab only has 16384 tokens
```

```swift
// WRONG (in CohereStatelessManager.swift)
private let eosTokenId = 151643  // Token doesn't exist!
```

### The Fix
```python
# CORRECT
EOS_TOKEN = 3  # <|endoftext|> - verified from model.generation_config.eos_token_id
```

```swift
// CORRECT
private let eosTokenId = 3  // <|endoftext|> - verified from model.generation_config.eos_token_id
```

### Impact
- **WER**: 29.88% → 11.95% (60% improvement)
- **Dots padding**: Eliminated
- **Text repetition**: Eliminated (samples 5 & 6 now perfect)
- **Natural stopping**: Decoder stops at EOS instead of max length

---

## Verification Results

### Python (Mobius)
✅ WER test: 11.95% on 10 LibriSpeech samples
✅ 2/10 samples achieved perfect 0.00% WER
✅ Most errors are just punctuation differences

### Swift (Ready for Integration)
✅ .mlmodelc compiles successfully
✅ Swift can load and run the compiled model
✅ All 21 inputs / 17 outputs validated
✅ Cache shapes correct: [1, 8, 108, 128]

---

## Next Steps

1. ✅ Fix EOS token in mobius Python scripts
2. ✅ Compile to .mlmodelc
3. ✅ Verify .mlmodelc works in Swift
4. ✅ Fix EOS token in FluidAudio Swift code
5. ✅ Push both repos
6. ⬜ Test cache-external decoder in FluidAudio package
7. ⬜ Compare WER: cache-external vs stateless
8. ⬜ Integrate into production FluidAudio package
9. ⬜ Ship it!

---

## Summary

Both repositories are now in sync with:
- ✅ Correct EOS token (3, not 151643)
- ✅ Cache-external decoder implementation
- ✅ .mlmodelc compilation support
- ✅ 11.95% WER achieved
- ✅ Ready for production integration

**Status**: Ready for FluidAudio Swift package testing and integration
