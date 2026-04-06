# Cohere Decoder Cache Investigation - Complete Summary

## Problem Statement

The Cohere Transcribe CoreML decoder was producing severe repetitions, resulting in 174% WER on LibriSpeech test-clean samples.

**Example failures**:
- Ground truth: "concord returned to its place **amidst** the tents"
- Hypothesis: "concord returned to its place **amidnace amidnace** of the tents"

## Investigation Timeline

### Phase 1: Isolate the Bug Location

**Test**: Created `test-pytorch-wrapper.py` to compare PyTorch wrapper vs CoreML
- PyTorch wrapper WER: 174.03%
- CoreML WER: 174.03%
- **Finding**: Bug is in the wrapper implementation, NOT in CoreML conversion

### Phase 2: Debug Cache Behavior

**Test**: Created `debug-pytorch-wrapper.py` to trace cache filling
**Finding**: Cache fills in REVERSE order!
```
Step 0: Non-zero positions: [107]
Step 1: Non-zero positions: [106, 107]
Step 2: Non-zero positions: [105, 106, 107]
```

**Root Cause Identified**: Sliding window in cache extraction
```python
# BUG: Keeps "last 108 positions" - causes sliding window
if current_len > self.max_seq_len:
    layer_k = layer_k[:, -self.max_seq_len:, :]  # Drops position 0!
```

**Why this breaks**:
1. At step N, decoder appends token at position 108 (makes 109 total)
2. Code keeps "last 108" → drops position 0, keeps 1..108
3. Next step: token goes to 108 again, drops position 1, keeps 2..109
4. Positions shift with each step → breaks positional encoding → repetitions

### Phase 3: Fix in PyTorch

**File**: `export-decoder-fixed.py`

**Fix**: Only pass filled cache positions (0:step), not full cache
```python
step_int = int(step.item())
for layer_idx in range(self.num_layers):
    if step_int > 0:
        # Only pass positions 0..step-1
        layer_k = cache_k[layer_idx:layer_idx+1, :, :step_int, :]
        layer_v = cache_v[layer_idx:layer_idx+1, :, :step_int, :]
        self_attention_cache.update(layer_k, layer_v, layer_idx)
```

**Test**: `test-fixed-pytorch.py`
**Result**: ✅ **PERFECT** - All 3 samples transcribed correctly, 0 repetitions

### Phase 4: Try to Export Fixed Version to CoreML

**Problem**: The fix uses `.item()` which is not traceable
```python
step_int = int(step.item())  # ⚠️ Gets traced as constant value
```

**Result**: CoreML model only outputs "." (2 tokens)
**Reason**: Tracer converts `:step.item()` to `:0` (constant)

**Attempts to fix**:
1. ❌ `torch.jit.script` - Model too complex, fails
2. ❌ `torch.narrow` - Still needs `.item()` for length
3. ❌ `torch.index_select` - Still needs `.item()` for indices

### Phase 5: Try Attention Masking

**File**: `export-decoder-masked.py`

**Approach**: Pass full cache, use attention mask to hide unused positions
```python
# Pass full 108 positions to DynamicCache
for layer_idx in range(self.num_layers):
    layer_k = cache_k[layer_idx:layer_idx+1, :, :, :]  # All 108
    self_attention_cache.update(layer_k, layer_v, layer_idx)

# Mask positions > step
should_mask = pos_range > step_exp
self_attention_mask = torch.where(should_mask, -inf, 0.0)
```

**Result**: ❌ Still has repetitions
**Reason**: Passing full 108-position cache creates inconsistency with actual sequence length

### Phase 6: Stateless Approach (Final Solution)

**File**: `export-decoder-stateless.py`

**Approach**: Reprocess all tokens at each step (no cache)
```python
def forward(self, input_ids, encoder_hidden_states, cross_attention_mask):
    """
    At step N: input_ids contains all N tokens (0..N-1)
    Process them all, return logits for last token
    """
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    # No cache!
    decoder_outputs, _ = self.decoder(
        input_ids=input_ids,
        positions=positions,
        encoder_hidden_states=encoder_hidden_states,
        past_key_values=None,  # ← Key difference
        ...
    )

    # Return logits for last token
    last_hidden = decoder_outputs[:, -1:, :]
    return self.log_softmax(last_hidden).squeeze(1)
```

**Test**: `test-stateless-coreml.py`

**Results**:
| Sample | Duration | Ground Truth | Result |
|--------|----------|--------------|--------|
| 1 | 3.5s | "concord...amidst the tents" | ✅ **Perfect** |
| 2 | 14.2s | "the english forwarded..." | ⚠️ "erected the french erected..." |
| 3 | 5.0s | "congratulations were poured..." | ✅ **Perfect** |

**Trade-offs**:
- ✅ Fixes 2/3 samples perfectly
- ✅ Fully traceable (no `.item()`)
- ✅ Simpler architecture
- ⚠️ O(n^2) complexity (acceptable for < 200 tokens)
- ⚠️ Sample 2 still has issues (different error pattern)

## Technical Deep Dive

### Why Sliding Window Breaks

**Incorrect cache management**:
```
Step 0: Cache[0..107] = [X, _, _, ..., _]  (X at position 0)
Step 1: Decoder writes to position 108 → [X, Y, _, ..., _, Z]
        Keep last 108 → [Y, _, _, ..., _, Z] (X dropped, positions shift!)
Step 2: Model thinks Y is at position 0, but it was position 1 → confusion
```

**Correct cache management** (as in export-decoder-fixed.py):
```
Step 0: Pass empty cache, decoder writes to position 0 → [X, _, _, ..., _]
Step 1: Pass cache[0:1] (just X), decoder writes to position 1 → [X, Y, _, ..., _]
Step 2: Pass cache[0:2] (X, Y), decoder writes to position 2 → [X, Y, Z, ..., _]
```

Positions stay stable → no confusion → no repetitions

### Why CoreML Can't Handle Dynamic Slicing

**PyTorch trace** is a static execution trace:
```python
step = torch.tensor([5])
slice = cache[:, :, :step.item(), :]  # Traced as [:, :, :5, :]
```

When you later call with `step=10`, CoreML still uses `:5` because that's what was traced.

**Solutions that don't work**:
- `torch.narrow(tensor, dim, start, length.item())` - `.item()` is still traced as constant
- `torch.index_select(tensor, dim, indices)` - Creating indices needs `.item()`
- `torch.jit.script` - Requires simpler model code, Transformers is too complex

**Solution that works**: No dynamic operations at all (stateless)

### Why Stateless Works

**No dynamic operations**:
```python
# Input shape is dynamic (EnumeratedShapes), but operations are all static
input_ids: (1, seq_len)  # seq_len varies 1..108
positions = torch.arange(seq_len)  # Computed from input shape
```

CoreML can handle dynamic input shapes, just not dynamic indexing operations.

## Files Organization

### Working Solution
- `export-decoder-stateless.py` - Export script (root directory)
- `tests/test-stateless-coreml.py` - Test script
- `build/cohere_decoder_stateless.mlpackage` - CoreML model (290.5 MB)

### Documentation
- `docs/CACHE_INVESTIGATION_SUMMARY.md` - This file
- `docs/DECODER_CACHE_FIX.md` - Concise fix documentation
- `docs/REVERSE_ENGINEERING.md` - Model architecture documentation
- `docs/OFFICIAL_USAGE_ANALYSIS.md` - Official implementation analysis

### Archive
- `archive-failed-approaches/` - All failed attempts with explanations
- `archive-failed-approaches/README.md` - Why each approach failed

### Tests
- `tests/` - Test and debug scripts

## Lessons Learned

1. **Test in PyTorch first**: Isolate wrapper bugs from CoreML conversion issues
2. **CoreML tracing is strict**: No `.item()`, no dynamic control flow
3. **Simple is better**: Stateless O(n^2) beats complex O(n) with dynamic slicing
4. **Debug systematically**: Print cache state at each step to understand behavior
5. **Document failures**: Archive failed approaches with explanations for future reference

## Remaining Issues

**Sample 2 degradation** (14.2s audio):
- Stateless approach shows different error pattern than cached version
- "erected the french erected the french..." vs "flowers of flowers of..."
- Hypothesis: Numerical precision (float16), encoder issues, or sequence length effects
- Affects longer sequences more than short ones

**Potential future work**:
1. Test float32 precision
2. Debug encoder output for Sample 2
3. Compare PyTorch stateless with CoreML stateless behavior
4. Hybrid approach: cache for short sequences, stateless for long

## Conclusion

The **stateless decoder** is the pragmatic solution:
- Simple architecture eliminates cache management complexity
- Fully CoreML compatible (no dynamic operations)
- Fixes 2/3 test samples perfectly
- O(n^2) complexity acceptable for typical transcription lengths

The root cause was a **sliding window bug** where keeping "last 108 positions" caused positions to shift, breaking positional encoding. The fix (only pass filled positions) works perfectly in PyTorch but can't be expressed in CoreML's static execution model, so we adopted a stateless approach instead.
