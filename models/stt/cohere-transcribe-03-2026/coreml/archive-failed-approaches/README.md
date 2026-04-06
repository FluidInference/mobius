# Failed Approaches Archive

This directory contains decoder export attempts that didn't work. Archived for reference.

## Why These Failed

### export-decoder-cached.py
**Issue**: Sliding window bug - keeps "last 108 positions" causing cache positions to shift
**Result**: 174% WER, severe repetitions ("amidnace amidnace", "flowers of flowers of...")
**Root cause**:
```python
# Keeps last 108, drops position 0, shifts everything down
layer_k = layer_k[:, -self.max_seq_len:, :]
```

### export-decoder-fixed.py
**Issue**: Uses `.item()` for dynamic slicing - not traceable in CoreML
**Result**: ✅ Perfect in PyTorch (0% errors), ❌ Broken in CoreML (only outputs ".")
**Root cause**:
```python
step_int = int(step.item())  # Gets traced as constant!
layer_k = cache_k[layer_idx:layer_idx+1, :, :step_int, :]  # Becomes :0
```

### export-decoder-masked.py
**Issue**: Attention masking approach - passes full cache with masking
**Result**: Still has repetitions (Sample 1: "amidnace amidnace")
**Root cause**: Passing full 108-position cache creates positional inconsistencies with actual sequence length

### export-decoder-narrow.py
**Issue**: torch.narrow requires `.item()` for length parameter
**Result**: Not traceable, same issue as export-decoder-fixed.py
**Root cause**:
```python
actual_len = torch.clamp(step, min=torch.tensor(1, device=device))
layer_k = torch.narrow(layer_k_full, dim=2, start=0, length=int(actual_len.item()))
```

### export-decoder-static.py
**Issue**: StaticCache incompatible with model architecture
**Result**: Shape mismatch errors during decoder forward pass
**Error**: `RuntimeError: output with shape [1, 8, 1, 1] doesn't match the broadcast shape [1, 8, 1, 109]`

### export-decoder-manual.py
**Note**: Investigation script showing decoder can run without cache (past_key_values=None)
**Purpose**: Validated that stateless approach is feasible

### export-decoder-index-select.py
**Issue**: torch.index_select still requires `.item()` for indices
**Result**: Incomplete, same traceability issues as other dynamic slicing approaches

## Test Scripts

### test-pytorch-wrapper.py
**Purpose**: Test if wrapper has repetitions in PyTorch before CoreML conversion
**Finding**: Confirmed bug is in wrapper (174% WER in PyTorch), not CoreML conversion

### test-fixed-pytorch.py
**Purpose**: Validate fixed wrapper works in PyTorch
**Result**: ✅ Perfect transcriptions in PyTorch (proved fix is correct)

### test-fixed-coreml.py
**Purpose**: Test fixed version in CoreML
**Result**: ❌ Only outputs "." (2 tokens) - dynamic slicing not traceable

### test-masked-coreml.py
**Purpose**: Test attention masking approach in CoreML
**Result**: ❌ Still has repetitions - approach didn't work

### debug-pytorch-wrapper.py
**Purpose**: Debug cache behavior step-by-step
**Finding**: Cache fills in REVERSE order (107 → 106 → 105...) due to sliding window

## Key Learnings

1. **Sliding window is the bug**: Keeping "last 108 positions" breaks positional encoding
2. **CoreML doesn't support dynamic slicing**: `.item()` gets traced as constant
3. **torch.jit.script doesn't work**: Model too complex for scripting
4. **Attention masking insufficient**: Need to pass correct sequence length to DynamicCache
5. **Stateless approach works**: O(n^2) but fully traceable and fixes most cases

## Working Solution

See `export-decoder-stateless.py` and `DECODER_CACHE_FIX.md` in parent directory.
