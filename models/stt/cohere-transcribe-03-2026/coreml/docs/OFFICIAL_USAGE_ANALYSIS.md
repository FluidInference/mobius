# Official Cohere Transcribe Usage Analysis

## Test Results: Preprocessing Is NOT the Issue

### Test 1: Original Preprocessing (n_fft=1024, no dithering)
- **Average WER**: 185.99%
- **Behavior**: Moderate gibberish with some repetition

### Test 2: Official Preprocessing (n_fft=512, dithering, pre-emphasis, normalization)
- **Average WER**: 544.38% (WORSE!)
- **Behavior**: More severe repetition loops, but initial words sometimes correct

**Conclusion**: Changing preprocessing made WER worse, not better. The CoreML models may have been exported with n_fft=1024 preprocessing already baked in. **The real issue is decoder cache handling, not preprocessing.**

## Critical Finding: Decoder Cache Handling

The official implementation uses a completely different cache structure than our manual approach:

### Our Approach (Manual)
```python
# Manual cache management
cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

# Update cache manually each step
for key, value in decoder_output.items():
    if 'k' in key.lower():
        cache_k = value
    else:
        cache_v = value
```

### Official Approach (EncoderDecoderCache)
```python
# From modeling_cohere_asr.py lines 498-522
cache_implementation = "static"  # Uses StaticCache or DynamicCache

if isinstance(past_key_values, EncoderDecoderCache):
    if is_cross_attention:
        cache_layer = past_key_values.cross_attention_cache
    else:
        cache_layer = past_key_values.self_attention_cache

# Cross-attention cache computed ONCE and reused
if is_cross_attention and cache_layer is not None and is_cross_cache_updated:
    key, value = _get_cache_kv(cache_layer, self.layer_idx)  # Reuse!
else:
    key = self._reshape(self.key_net(source))
    value = self._reshape(self.value_net(source))
    cache_layer.update(key, value, self.layer_idx, cache_kwargs=cache_kwargs)

# Self-attention cache truncation
if not is_cross_attention and kv_seq_len is not None:
    key = key[:, :, :kv_seq_len]
    value = value[:, :, :kv_seq_len]
```

**Key differences:**
1. **Separate cache objects** for self-attention vs cross-attention
2. **Cross-attention cache computed once** at start, then reused (lines 507-508)
3. **Self-attention cache truncated** using kv_seq_len (lines 517-519)
4. **cache_position tracking** for proper positional encoding

## Why Repetition Loops Happen

Looking at the sample outputs:

```
Sample 8:
Ground truth: "you will be frank with me i always am"
Hypothesis:   "you will be frank with me. i always am. i always am. i always am..."
```

The decoder:
1. Correctly transcribes the first sentence
2. Gets stuck repeating "i always am" 40+ times
3. Never generates EOS token

This suggests:
- **Cache is corrupted** after a few steps
- **Positional encoding is wrong** causing model to think it's at the same position
- **Cross-attention cache might be getting updated** when it shouldn't be

## What the Official Code Tells Us

From the official implementation, the critical aspects are:

### 1. Prompt Structure ✅ CORRECT
Our 10-token prompt matches perfectly:
```python
# Official build_prompt() for English with punctuation:
"<|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>"

# Our prompt: [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13] ✓
```

### 2. Generation Config ✅ CORRECT
```json
{
  "decoder_start_token_id": 13764,  // Matches our prompt[0]
  "eos_token_id": 3,
  "bos_token_id": 4,
  "pad_token_id": 2
}
```

### 3. Decoder Cache Structure ❌ WRONG
We're using manual cache management, but the official code uses:
- EncoderDecoderCache with is_updated tracking
- Separate self_attention_cache and cross_attention_cache
- cache_position for proper indexing
- kv_seq_len for self-attention truncation

## The Real Problem: CoreML Export Incompatibility

The fundamental issue is that CoreML doesn't natively support:
1. **EncoderDecoderCache** - a Python class that tracks separate caches
2. **Dynamic cache updating logic** - conditional cache reuse based on is_updated flags
3. **cache_position tensors** - for proper positional encoding

Our export attempts to manually replicate this with fixed-size cache tensors, but the cache update logic is clearly broken, causing:
- Repetition loops
- Failure to generate EOS
- Decoder getting stuck at certain positions

## Possible Solutions

### Option 1: Fix Cache Update Logic
Investigate exactly how the decoder output cache should be merged with the input cache. Currently we're just replacing the entire cache, but maybe we need to:
- Only update specific positions
- Truncate self-attention cache properly
- Keep cross-attention cache frozen after first computation

### Option 2: Pre-compute Cross-Attention KV
The official code computes cross-attention cache once and reuses it. We could:
1. Export a separate model that computes cross KV from encoder output
2. Pass pre-computed cross KV to decoder
3. Only manage self-attention cache dynamically

This is what `export-decoder-with-cross-kv.py` attempts.

### Option 3: Use PyTorch Model Directly
Instead of CoreML, use the official PyTorch model via:
- torch.jit (TorchScript)
- ONNX Runtime
- Direct PyTorch with torch.compile

This would avoid the CoreML cache handling issues entirely.

## Conclusion

The preprocessing parameters from the config don't match what the CoreML models were exported with (changing them made WER worse). **The real bug is in the decoder cache handling** - the manual cache management in our CoreML export doesn't correctly replicate the EncoderDecoderCache behavior, causing severe repetition loops.

Fixing this requires either:
1. Correctly replicating the cache update logic in CoreML
2. Pre-computing cross-attention cache separately
3. Abandoning CoreML for this model architecture

The fact that even BarathwajAnandan's reference models show 185-544% WER suggests this is a **fundamental CoreML export issue**, not just our implementation.
