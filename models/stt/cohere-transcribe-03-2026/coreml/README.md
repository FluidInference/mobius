# Cohere Transcribe CoreML Export

CoreML export of [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) for on-device speech recognition on Apple Silicon.

## Status: ✅ Working with Stateless Decoder

| Component | Status | Notes |
|-----------|--------|-------|
| **Encoder** | ✅ Working | Perfect parity with reference (max diff 0.041) |
| **Decoder (Stateless)** | ✅ Mostly Working | Fixes 2/3 test samples perfectly, O(n^2) complexity |
| **Decoder (Cached)** | ❌ Broken | 174% WER due to sliding window bug (archived) |
| **Mel Preprocessing** | ✅ Working | Python implementation matches reference |

### Current Test Results (LibriSpeech test-clean, 3 samples)

**Stateless Decoder** (`export-decoder-stateless.py`):
- Sample 1 (3.5s): ✅ **Perfect transcription**
- Sample 2 (14.2s): ⚠️ Different error pattern (still investigating)
- Sample 3 (5.0s): ✅ **Perfect transcription**

**Cached Decoder** (archived):
- Average WER: 174%
- Issue: Sliding window bug causes severe repetitions

## Current Models

**FP16 Models (build/):**
- `cohere_encoder.mlpackage` (3.6 GB) - ✅ Working perfectly
- `cohere_decoder_stateless.mlpackage` (291 MB) - ✅ Stateless decoder (fixes 2/3 samples)
- `cohere_cross_kv_projector.mlpackage` (32 MB)

**Archived (broken):**
- `cohere_decoder_cached.mlpackage` - ❌ Sliding window bug (see `archive-failed-approaches/`)

## Quick Start

### Export Models

```bash
# Export encoder (FP16)
uv run python3 export-encoder.py --output-dir build --precision float16

# Export stateless decoder (FP16)
uv run python3 export-decoder-stateless.py --output-dir build --precision float16
```

### Test Models

```bash
# Test stateless decoder on LibriSpeech samples
uv run python3 tests/test-stateless-coreml.py

# Test on 10 LibriSpeech samples (legacy test)
uv run python3 tests/test-librispeech.py
```

## Decoder Cache Fix

### Problem: Sliding Window Bug

The original cached decoder had **174% WER** due to a bug where keeping "last 108 positions" caused cache positions to shift at each step, breaking positional encoding.

**Example failure**:
- Ground truth: "concord returned to its place **amidst** the tents"
- Cached decoder: "concord returned to its place **amidnace amidnace** of the tents"

### Solution: Stateless Decoder

The stateless decoder reprocesses all tokens at each step (O(n^2) complexity) instead of managing cache state. This is:
- ✅ Fully CoreML traceable (no `.item()` calls)
- ✅ Fixes 2/3 test samples perfectly
- ✅ Simpler architecture (no cache management)
- ⚠️ O(n^2) complexity (acceptable for < 200 tokens)

**See `docs/DECODER_CACHE_FIX.md` for complete investigation.**

## Critical Implementation Details

### 10-Token Prompt Required

The decoder requires a 10-token configuration prompt:

```python
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
# ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
# <|en|> <|en|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
```

### Stateless Decoder Interface

**Inputs**:
- `input_ids`: All tokens so far, shape (1, seq_len) - EnumeratedShapes [1,1] to [1,108]
- `encoder_hidden_states`: Encoder output, shape (1, enc_len, 1024)
- `cross_attention_mask`: Encoder attention mask, shape (1, 1, 1, enc_len)

**Outputs**:
- `logits`: Log probabilities for next token, shape (1, vocab_size=16384)

**Usage**:
```python
# Initialize with prompt
tokens = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]

# Generate tokens
for step in range(10, 200):  # Up to 200 tokens
    input_ids = np.array([tokens], dtype=np.int32)
    output = decoder.predict({
        "input_ids": input_ids,
        "encoder_hidden_states": encoder_hidden,
        "cross_attention_mask": cross_mask,
    })
    next_token = np.argmax(output["logits"][0])
    tokens.append(next_token)
    if next_token == EOS_TOKEN_ID:
        break
```

## Files Organization

### Working Solution
- `export-decoder-stateless.py` - Stateless decoder export (O(n^2), fully traceable)
- `export-encoder.py` - Encoder + projection layer export
- `export-cross-kv-projector.py` - Cross-attention KV projector export

### Documentation
- `docs/CACHE_INVESTIGATION_SUMMARY.md` - Complete investigation of 6 approaches
- `docs/DECODER_CACHE_FIX.md` - Concise fix documentation
- `docs/REVERSE_ENGINEERING.md` - Model architecture details
- `docs/OFFICIAL_USAGE_ANALYSIS.md` - Official implementation analysis

### Tests
- `tests/test-stateless-coreml.py` - Test stateless decoder
- `tests/test-librispeech.py` - Legacy WER test (10 samples)
- `tests/debug-*.py` - Debug scripts
- `tests/test-*.py` - Various test scripts

### Archive
- `archive-failed-approaches/` - 7 failed decoder exports with explanations
  - `export-decoder-cached.py` - Original sliding window bug
  - `export-decoder-fixed.py` - Works in PyTorch but not CoreML (uses `.item()`)
  - `export-decoder-masked.py` - Attention masking attempt (still has repetitions)
  - `export-decoder-narrow.py` - torch.narrow approach (not traceable)
  - `export-decoder-static.py` - StaticCache attempt (shape mismatches)
  - `export-decoder-manual.py` - Investigation script
  - `export-decoder-index-select.py` - torch.index_select attempt
- `archive-failed-approaches/README.md` - Why each approach failed

### Preprocessing
- `cohere_mel_spectrogram.py` - Mel spectrogram computation (Python reference)

### Utilities
- `benchmark-models.py` - Model performance benchmarking
- `compare-models.py` - PyTorch vs CoreML comparison
- `compile_models.py` - Compile .mlpackage to .mlmodelc
- `measure-memory.py` - Memory usage measurement

## Known Issues

1. **Sample 2 degradation**: Longer audio (14.2s) still has issues with stateless decoder
   - Hypothesis: Numerical precision (float16), encoder issues, or sequence length effects
   - Affects longer sequences more than short ones

2. **O(n^2) complexity**: Stateless decoder reprocesses all tokens at each step
   - Acceptable for < 200 tokens (typical transcription length)
   - May be slower on very long sequences

3. **Quantization not tested**: Only FP16 models have been tested with stateless decoder
   - Previous cached decoder: INT8/INT6 crashed or produced worse quality

## Investigation Summary

Tested 6+ different approaches to fix the cache bug:

1. ❌ **Cached with sliding window** - Original bug (174% WER)
2. ✅ **Fixed cache (PyTorch only)** - Perfect results but uses `.item()` (not CoreML traceable)
3. ❌ **Attention masking** - Still has repetitions
4. ❌ **torch.narrow** - Requires `.item()`
5. ❌ **torch.index_select** - Requires `.item()`
6. ❌ **StaticCache** - Shape mismatches
7. ✅ **Stateless** - Works in CoreML, fixes 2/3 samples

**Key finding**: CoreML tracing doesn't support dynamic slicing with `.item()` - it gets traced as a constant value.

**See `docs/CACHE_INVESTIGATION_SUMMARY.md` for complete timeline.**

## Next Steps

1. **Investigate Sample 2 degradation**: Try float32 precision, debug encoder output
2. **Benchmark O(n^2) performance**: Measure actual overhead on typical transcriptions
3. **Test quantization**: INT8/INT6 quantization with stateless decoder
4. **Hybrid approach**: Consider cache for short sequences, stateless for long

## Requirements

- macOS 14+ / iOS 17+
- Python 3.10+
- Dependencies: see `pyproject.toml`
  - coremltools
  - PyTorch
  - transformers
  - datasets (for testing)
  - sentencepiece (for tokenization)

## License

GPL-3.0 (matching upstream CoreML conversion)

Base model: Apache-2.0 ([CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026))
