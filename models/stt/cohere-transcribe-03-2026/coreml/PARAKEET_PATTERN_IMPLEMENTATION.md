# Cohere Decoder - Parakeet Pattern Implementation

## Summary

Brandon's recommendation: "for parakeet we just passed it in manually each loop and tracked the state outside of the coreml decoder"

This means:
- ✅ Cache managed in Swift (outside CoreML model)
- ✅ Cache passed IN as model inputs
- ✅ Updated cache returned OUT as model outputs
- ✅ No `register_buffer()` or State API needed
- ✅ Works on macOS 14

## Current Status

Steve has already implemented most of this pattern! Looking at `CohereAsrManager.swift` (lines 156-255):

```swift
// Cache inputs
"cache_k": MLFeatureValue(multiArray: cacheK),
"cache_v": MLFeatureValue(multiArray: cacheV),

// Cache outputs
let newCacheK = decoderOutput.featureValue(for: "new_cache_k")
let newCacheV = decoderOutput.featureValue(for: "new_cache_v")

// Update for next iteration
for i in 0..<cacheSize {
    cacheK[i] = newCacheK[i]
    cacheV[i] = newCacheV[i]
}
```

This IS the Parakeet pattern!

## The Problem

The issue is likely in the **export script** - the decoder model needs to:
1. Take cache as inputs (Steve already has this)
2. Update cache properly without using `.item()` (THIS is where it's stuck)
3. Return updated cache as outputs (Steve already has this)

## Solution: Use `attention_mask.shape[-1]` to Infer Position

The key insight from Qwen3's stateful decoder (see `export-decoder-stateful.py` lines 96-100):

```python
# Infer position from attention_mask shape (NO .item() needed!)
end_step = attention_mask.shape[-1]  # Current sequence length
past_kv_len = end_step - 1  # Positions already in cache
```

This works because:
- `attention_mask` is a **dynamic input** with `RangeDim`
- Its shape grows from `[1,1,1,1]` → `[1,1,1,2]` → `[1,1,1,108]`
- Using `.shape[-1]` is traceable (no `.item()` call!)
- We can use `past_kv_len` to index into cache without tracing issues

## Implementation Steps

### 1. Export Script (`export-decoder-cache-external.py`)

**Current approach (WRONG - uses `.item()`):**
```python
past_len_int = int(past_kv_len.item())  # ❌ Gets traced as constant!
k_cache[:, :, past_len_int:end_step, :] = key
```

**Fixed approach (use attention_mask shape):**
```python
# Inputs:
# - attention_mask: [1, 1, 1, end_step] with RangeDim(1, 108)
# - NO separate past_kv_len input needed!

end_step = attention_mask.shape[-1]  # Dynamic, traceable
past_kv_len = end_step - 1           # Derived from shape

# Now we can slice without .item():
k_cache_new = k_cache.clone()
k_cache_new[:, :, past_kv_len:end_step, :] = key
```

### 2. Swift Side (`CohereAsrManager.swift`)

**Changes needed:**

1. Remove `step` input, use `attention_mask` instead:

```swift
// OLD:
let stepArray = try MLMultiArray(shape: [1], dataType: .int32)
stepArray[0] = NSNumber(value: step)
"step": MLFeatureValue(multiArray: stepArray),

// NEW:
let attentionMask = createAttentionMask(currentSeqLen: step + 1)
"attention_mask": MLFeatureValue(multiArray: attentionMask),
```

2. Implement `createAttentionMask`:

```swift
func createAttentionMask(currentSeqLen: Int) -> MLMultiArray {
    // Shape: [1, 1, 1, currentSeqLen]
    // Size grows each step: [1,1,1,1] -> [1,1,1,2] -> [1,1,1,3] ...
    let mask = try! MLMultiArray(shape: [1, 1, 1, NSNumber(value: currentSeqLen)], dataType: .float32)
    // Fill with zeros (all positions valid for causal attention)
    for i in 0..<currentSeqLen {
        mask[[0, 0, 0, i] as [NSNumber]] = 0.0
    }
    return mask
}
```

## Architecture Comparison

### Qwen3 (register_buffer - macOS 15+ only)
```python
class StatefulQwen3:
    def __init__(self):
        # Cache INSIDE model (State API)
        self.register_buffer("k_cache", torch.zeros(...))

    def forward(self, input, attention_mask):
        # Cache persists between calls
        k_cache[:, :, past_kv_len:end_step, :] = key
```

### Parakeet / Cohere (cache-external - macOS 14+)
```python
class CacheExternal:
    def forward(self, input, k_cache, v_cache, attention_mask):
        # Cache passed IN
        k_cache_new = k_cache.clone()
        k_cache_new[:, :, past_kv_len:end_step, :] = key
        # Cache passed OUT
        return logits, k_cache_new, v_cache_new
```

## Files

- ✅ `CohereDecoderState.swift` - Created (manages cache arrays)
- ✅ `CohereModelInference.swift` - Created (runDecoder helper)
- ⚠️  `export-decoder-cache-external.py` - Created but needs testing
- ⚠️  `CohereAsrManager.swift` - Exists, needs minor updates for attention_mask

## Next Steps

1. Test `export-decoder-cache-external.py` to verify it works
2. Update `CohereAsrManager.swift` to use `attention_mask` instead of `step`
3. Test end-to-end inference
4. Compare with stateless decoder (already works but O(n²))

## Key Insight

The **fundamental problem** Steve is hitting is:
- Can't use `.item()` because it gets traced as a constant
- Can't use dynamic slicing with variable indices in CoreML

The **solution**:
- Use `attention_mask.shape[-1]` which IS dynamic and traceable
- Swift grows the attention_mask size each step
- Decoder infers position from mask size, not from explicit counter

This is exactly how Qwen3 solved it, and it applies to Cohere's cache-external pattern too!
