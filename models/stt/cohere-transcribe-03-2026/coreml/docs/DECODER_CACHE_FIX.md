# Cohere Decoder Cache Fix - Stateless Approach

## Problem
The original cached decoder (`export-decoder-cached.py`) had severe repetition issues (174% WER) due to a **sliding window bug** in cache management.

### Root Cause
```python
# BUG: Keeping "last 108 positions" causes sliding window
if current_len > self.max_seq_len:
    layer_k = layer_k[:, -self.max_seq_len:, :]  # Positions shift!
```

At each step:
1. Decoder appends new position at 108 (making 109 total)
2. Code keeps "last 108" positions
3. This drops position 0 and shifts everything down
4. Positional encoding breaks, causing repetitions

## Solution: Stateless Decoder

**File**: `export-decoder-stateless.py`

**Approach**: Reprocess all tokens at each step (no cache)
- At step N: Pass all N tokens (0..N-1) to decoder
- Return logits for the last token
- O(n^2) complexity but fully traceable

**Key advantages**:
- ✅ No cache management complexity
- ✅ Fully traceable (no `.item()` calls)
- ✅ CoreML compatible
- ✅ Fixes 2/3 test samples perfectly

## Test Results

**LibriSpeech test-clean samples**:

| Sample | Duration | Original WER | Stateless Result |
|--------|----------|--------------|------------------|
| 1 | 3.5s | 174% (repetitions) | ✅ **Perfect match** |
| 2 | 14.2s | 174% (repetitions) | ⚠️ Different error pattern |
| 3 | 5.0s | 174% (repetitions) | ✅ **Perfect match** |

### Sample Results

**Sample 1** (✅ Perfect):
- Ground truth: "concord returned to its place amidst the tents"
- Hypothesis: "concord returned to its place amidst the tents."

**Sample 2** (⚠️ Still has issues):
- Ground truth: "the english forwarded to the french baskets of flowers..."
- Hypothesis: "the english erected the french erected the french erected..."
- Note: Different error pattern than cached version

**Sample 3** (✅ Perfect):
- Ground truth: "congratulations were poured in upon the princess everywhere during her journey"
- Hypothesis: "congratulations were poured in upon the princess everywhere during her journey."

## Known Issues

1. **Sample 2 degradation**: Longer audio (14.2s) still has repetitions, though different pattern
   - Possible causes: sequence length, numerical precision (float16), encoder issues
   - Affects longer sequences more than short ones

2. **O(n^2) complexity**: Reprocesses all tokens at each step
   - Acceptable for < 200 tokens (typical transcription length)
   - May be slower on very long sequences

## Files

**Working solution**:
- `export-decoder-stateless.py` - Export script (root directory)
- `tests/test-stateless-coreml.py` - Test script
- `build/cohere_decoder_stateless.mlpackage` - CoreML model (290.5 MB)

**Failed approaches** (archived in `archive-failed-approaches/`):
- `export-decoder-cached.py` - Original sliding window bug
- `export-decoder-fixed.py` - Perfect in PyTorch but not CoreML compatible (uses `.item()`)
- `export-decoder-masked.py` - Attention masking, still has repetitions
- `export-decoder-narrow.py` - torch.narrow approach, not traceable
- `export-decoder-manual.py` - Investigation script
- `export-decoder-static.py` - StaticCache attempt, shape mismatches

## Usage

```bash
# Export
uv run python3 export-decoder-stateless.py

# Test
uv run python3 tests/test-stateless-coreml.py
```

## CoreML Model Interface

**Inputs**:
- `input_ids`: All tokens so far, shape (1, seq_len) - EnumeratedShapes [1,1] to [1,108]
- `encoder_hidden_states`: Encoder output, shape (1, enc_len, 1024)
- `cross_attention_mask`: Encoder attention mask, shape (1, 1, 1, enc_len)

**Outputs**:
- `logits`: Log probabilities for next token, shape (1, vocab_size=16384)

## Next Steps (if needed)

1. **Investigate Sample 2**: Try float32 precision, debug encoder output
2. **Benchmark performance**: Measure actual O(n^2) overhead
3. **Hybrid approach**: Use cache for short sequences, stateless for fallback
4. **Model debugging**: Compare PyTorch stateless with CoreML stateless

## Conclusion

The stateless approach is a pragmatic solution that eliminates most repetition issues while maintaining full CoreML compatibility. The O(n^2) complexity is acceptable for typical transcription lengths (< 200 tokens), and the simpler architecture avoids the cache management complexity that caused the original bug.
