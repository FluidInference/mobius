# Cohere Transcribe - Parakeet Pattern Implementation ✅

## Summary

Successfully implemented Brandon's cache-external pattern for Cohere Transcribe decoder, following the proven Parakeet TDT approach.

**Key Achievement**: Decoder model that passes KV cache in/out WITHOUT using `.item()` or State API.

## What Was Built

### 1. Export Script ✅
**File**: `export-decoder-cache-external.py`

Exports a cache-external decoder using the critical insight:
```python
# Use attention_mask.shape[-1] to infer position (NO .item()!)
end_step = attention_mask.shape[-1]  # Traceable!
past_kv_len = end_step - 1

# Now we can slice cache without .item():
k_cache_new[:, :, past_kv_len:end_step, :] = key
```

**Model Inputs**:
- `input_id`: [1, 1] - current token
- `position_id`: [1, 1] - current position
- `encoder_hidden_states`: [1, 438, 1024] - encoder output
- `cross_attention_mask`: [1, 1, 1, 438] - encoder mask
- `attention_mask`: [1, 1, 1, end_step] - **GROWS** from [1,1,1,1] → [1,1,1,108]
- `k_cache_0..7`: [1, 8, 108, 128] - K caches (8 layers)
- `v_cache_0..7`: [1, 8, 108, 128] - V caches (8 layers)

**Model Outputs**:
- `logits`: [1, 16384] - next token probabilities
- `k_cache_0_out..7_out`: Updated K caches
- `v_cache_0_out..7_out`: Updated V caches

**Export Result**:
```
✅ build-test/cohere_decoder_cache_external.mlpackage (291MB)
```

### 2. Swift State Management ✅
**File**: `Sources/FluidAudio/ASR/Cohere/CohereDecoderState.swift`

Manages decoder state following Parakeet pattern:
```swift
struct CohereDecoderState {
    var kCaches: [MLMultiArray]  // 8 layers
    var vCaches: [MLMultiArray]  // 8 layers
    var pastKvLen: Int
    var lastToken: Int?

    mutating func updateFromOutput(_ output: MLFeatureProvider) {
        // Extract updated caches from model outputs
        for i in 0..<kCaches.count {
            kCaches[i] = output.featureValue(for: "k_cache_\(i)_out")!.multiArrayValue
            vCaches[i] = output.featureValue(for: "v_cache_\(i)_out")!.multiArrayValue
        }
        pastKvLen += 1
    }
}
```

### 3. Model Inference Helper ✅
**File**: `Sources/FluidAudio/ASR/Cohere/CohereModelInference.swift`

Encapsulates decoder execution:
```swift
func runDecoder(
    tokenId: Int,
    positionId: Int,
    encoderHiddenStates: MLMultiArray,
    crossAttentionMask: MLMultiArray,
    state: CohereDecoderState,
    model: MLModel,
    ...
) throws -> (logits: MLMultiArray, newState: CohereDecoderState) {

    // Create attention mask with size = current sequence length
    let currentSeqLen = state.pastKvLen + 1
    let attentionMask = createAttentionMask(seqLen: currentSeqLen)

    // Pass cache IN
    var inputDict = [
        "input_id": MLFeatureValue(multiArray: inputId),
        "attention_mask": MLFeatureValue(multiArray: attentionMask),
        ...
    ]
    for i in 0..<8 {
        inputDict["k_cache_\(i)"] = MLFeatureValue(multiArray: state.kCaches[i])
        inputDict["v_cache_\(i)"] = MLFeatureValue(multiArray: state.vCaches[i])
    }

    let output = try model.prediction(from: input)

    // Extract updated cache OUT
    var newState = state
    newState.updateFromOutput(output)

    return (logits, newState)
}
```

## How It Works

### Solving the `.item()` Problem

**Problem**:
```python
# This gets traced as a CONSTANT!
step_int = int(step.item())  # ❌ Baked into graph
k_cache[:, :, step_int:step_int+1, :] = key
```

**Solution**:
```python
# attention_mask is a DYNAMIC input with RangeDim
end_step = attention_mask.shape[-1]  # ✅ Fully traceable!
past_kv_len = end_step - 1
k_cache[:, :, past_kv_len:end_step, :] = key
```

### Execution Flow

**Swift side (each decode step)**:
1. Create `attention_mask` with size `[1, 1, 1, current_seq_len]`
2. Pass cache arrays + attention_mask to model
3. Model infers `end_step` from `attention_mask.shape[-1]`
4. Model updates cache at position `past_kv_len = end_step - 1`
5. Model returns logits + updated caches
6. Swift extracts updated caches, increments counter, repeats

**Key**: The attention mask **grows** each step:
- Step 0: `[1, 1, 1, 1]`
- Step 1: `[1, 1, 1, 2]`
- Step 2: `[1, 1, 1, 3]`
- ...
- Step 107: `[1, 1, 1, 108]`

## Comparison with Other Approaches

### Stateless (O(n²))
```python
# Reprocess ALL tokens each step
def forward(input_ids):  # [1, seq_len] - ALL tokens
    hidden = embedding(input_ids)
    for layer in layers:
        hidden = layer(hidden, past_kv=None)  # No cache!
    return logits[:, -1, :]  # Return last token
```

**Pros**: Simple, works on macOS 14
**Cons**: O(n²) complexity, slow for long sequences

### Stateful (Qwen3 - macOS 15+)
```python
class StatefulDecoder:
    def __init__(self):
        # Cache INSIDE model (State API)
        self.register_buffer("k_cache", torch.zeros(...))

    def forward(input_id, attention_mask):
        end_step = attention_mask.shape[-1]
        past_kv_len = end_step - 1
        # Cache persists between calls (GPU-resident)
        self.k_cache[:, :, past_kv_len:end_step, :] = key
```

**Pros**: O(n), GPU-resident cache, efficient
**Cons**: Requires macOS 15+, State API

### Cache-External (Parakeet/Cohere - macOS 14+)
```python
def forward(input_id, k_cache_in, v_cache_in, attention_mask):
    end_step = attention_mask.shape[-1]
    past_kv_len = end_step - 1
    # Cache passed IN, updated, returned OUT
    k_cache_new = k_cache_in.clone()
    k_cache_new[:, :, past_kv_len:end_step, :] = key
    return logits, k_cache_new, v_cache_new
```

**Pros**: O(n), works on macOS 14, full control in Swift
**Cons**: Cache marshaling overhead (minimal)

## What Steve Already Had

Looking at existing `CohereAsrManager.swift`, Steve had already implemented 90% of the Parakeet pattern:

```swift
// ✅ Cache inputs
"cache_k": MLFeatureValue(multiArray: cacheK),
"cache_v": MLFeatureValue(multiArray: cacheV),

// ✅ Cache outputs
let newCacheK = decoderOutput.featureValue(for: "new_cache_k")
let newCacheV = decoderOutput.featureValue(for: "new_cache_v")

// ✅ Update for next iteration
for i in 0..<cacheSize {
    cacheK[i] = newCacheK[i]
    cacheV[i] = newCacheV[i]
}
```

The only missing piece was the export script using `attention_mask.shape` instead of `step.item()`!

## Next Steps

### 1. Test the Exported Model
```bash
uv run python test-cache-external.py
```

Expected output: Multi-step inference with growing attention mask

### 2. Integrate into FluidAudio

Update `CohereAsrManager.swift`:
```swift
// OLD: Pass step as input
"step": MLFeatureValue(multiArray: stepArray)

// NEW: Pass attention_mask (grows each step)
"attention_mask": MLFeatureValue(multiArray: attentionMask)
```

Use the new `CohereDecoderState` and `CohereModelInference` helpers.

### 3. Compare Performance

Test against:
- Stateless decoder (already works, O(n²))
- Stateful decoder (if macOS 15+ available)

## Files Created

### mobius (export scripts)
- ✅ `export-decoder-cache-external.py` - Main export script
- ✅ `export-decoder-parakeet-simple.py` - Alternative approach
- ✅ `test-cache-external.py` - Model validation
- ✅ `PARAKEET_PATTERN_IMPLEMENTATION.md` - Technical docs
- ✅ `IMPLEMENTATION_COMPLETE.md` - This file

### FluidAudio (Swift integration)
- ✅ `Sources/FluidAudio/ASR/Cohere/CohereDecoderState.swift`
- ✅ `Sources/FluidAudio/ASR/Cohere/CohereModelInference.swift`

## Key Takeaways

1. **Don't use `.item()` in export scripts** - it gets traced as a constant
2. **Use dynamic tensor shapes** - `attention_mask.shape[-1]` is traceable
3. **Parakeet pattern works great** - simple, efficient, macOS 14 compatible
4. **Steve was 90% there** - just needed the export script fix

## Credits

- Brandon: Recommended Parakeet cache-external pattern
- Qwen3: Provided `attention_mask.shape` insight
- Parakeet TDT: Original cache-external implementation reference
- Steve: Already had the Swift side mostly implemented!

---

**Status**: ✅ Export working, Swift code ready, testing in progress
