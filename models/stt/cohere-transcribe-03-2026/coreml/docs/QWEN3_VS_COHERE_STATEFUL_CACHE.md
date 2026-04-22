# Qwen3 vs Cohere: Stateful Cache Comparison

## Summary

Qwen3 successfully implemented stateful KV cache using `register_buffer()` and achieved 1.6% WER. However, **directly applying this approach to Cohere is challenging** due to fundamental architectural differences. The stateless decoder already fixes 2/3 samples - the remaining Sample 2 issue may not be cache-related.

## Architecture Comparison

| Aspect | Qwen3-ASR | Cohere Transcribe |
|--------|-----------|-------------------|
| **Layers** | 28 | 8 |
| **Attention** | GQA (16 Q heads, 8 KV heads) | Standard (8 Q heads, 8 KV heads) |
| **Head dim** | 128 | 128 |
| **QK norms** | Yes | No |
| **Model class** | Standard `Qwen3Model` | Custom `CohereAsrForConditionalGeneration` |
| **Decoder** | `Qwen3Model.layers` | `TransformerDecoderWrapper` with custom layers |
| **Layer type** | Standard transformer | Custom `TransformerDecoderLayer` |
| **Attention module** | `Qwen3Attention` | Custom `DecoderAttention` |
| **Cache interface** | `past_key_values` (tuples of K/V tensors) | `past_key_values` + `cache_position` + `kv_seq_len` |

## Qwen3's Stateful Approach (Proven Working)

### Key Technique: Manual Attention with Stateful Buffers

```python
class StatefulQwen3Decoder(nn.Module):
    def __init__(self, layers, max_seq_len=512):
        super().__init__()
        self.layers = layers

        # Register 56 state buffers (28 layers x K + V)
        # CoreML states MUST be fp16
        for i in range(NUM_LAYERS):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16),
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16),
            )

    def forward(self, hidden_states, position_cos, position_sin, attention_mask):
        # For each layer:
        for i in range(NUM_LAYERS):
            # 1. Project Q, K, V manually
            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)

            # 2. Apply RoPE manually
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            # 3. In-place cache update (CoreML detects as state mutation)
            k_cache[:, :, past_kv_len:end_step, :] = k.half()
            v_cache[:, :, past_kv_len:end_step, :] = v.half()

            # 4. Read cache and cast to fp32
            k_full = k_cache[:, :, :end_step, :].float()
            v_full = v_cache[:, :, :end_step, :].float()

            # 5. Manual scaled dot-product attention
            attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_full)

            # 6. Continue with MLP...
```

### Critical Success Factors

1. **Manual attention computation**: Qwen3 doesn't use the model's built-in attention - it manually implements Q/K/V projection, RoPE, attention, and output projection
2. **Direct layer access**: Simple `model.layers[i]` provides direct access to attention weights
3. **Standard architecture**: Qwen3 uses standard Hugging Face Transformers structure
4. **Cache padding**: Avoids the 112-126 dimension bug zone by padding to HEAD_DIM (128)

## Why Direct Application to Cohere is Challenging

### 1. Custom Architecture

Cohere uses a custom `TransformerDecoderWrapper` with non-standard modules:

```python
# Cohere structure
model.transf_decoder                           # TransformerDecoderWrapper
├── _embedding                                  # TransformerDecoderEmbedding
│   ├── token_embedding                        # Embedding
│   ├── position_embedding                     # FixedPositionalEncoding
│   └── layer_norm                             # LayerNorm
└── _decoder                                    # TransformerDecoderCore
    ├── layers                                  # ModuleList of TransformerDecoderLayer
    │   ├── layer_norm_1                       # LayerNorm
    │   ├── first_sub_layer                    # DecoderAttention (self-attention)
    │   ├── layer_norm_2                       # LayerNorm
    │   ├── second_sub_layer                   # DecoderAttention (cross-attention)
    │   ├── layer_norm_3                       # LayerNorm
    │   └── third_sub_layer                    # DecoderFeedForward
    └── final_layer_norm                       # LayerNorm
```

**vs Qwen3's standard structure:**

```python
model.layers[i]                                # Standard TransformerDecoderLayer
├── input_layernorm                            # RMSNorm
├── self_attn                                  # Qwen3Attention
│   ├── q_proj, k_proj, v_proj, o_proj        # Direct weight access
│   └── q_norm, k_norm (optional)             # QK layer norms
├── post_attention_layernorm                   # RMSNorm
└── mlp                                         # Qwen3MLP
    └── gate_proj, up_proj, down_proj          # Direct weight access
```

### 2. Nested Module Access

To manually implement attention for Cohere, we'd need to:

1. Access `layer.first_sub_layer` (which is `DecoderAttention`, not standard attention)
2. Extract internal Q/K/V projection weights from `DecoderAttention`
3. Understand Cohere's custom attention implementation
4. Re-implement it with stateful buffers

**Problem**: `DecoderAttention` is a custom module - we don't know its internal structure without reading the source code from Hugging Face cache.

### 3. Cross-Attention Complexity

Cohere has both self-attention AND cross-attention in each layer:
- `first_sub_layer`: Self-attention (needs KV cache)
- `second_sub_layer`: Cross-attention (encoder-decoder, no cache needed)

Qwen3 only has self-attention in the decoder (it's audio encoder + LLM, no cross-attention).

### 4. Position Encoding Differences

- **Qwen3**: Uses RoPE (rotary position embeddings) - precomputed cos/sin tensors passed as inputs
- **Cohere**: Uses `FixedPositionalEncoding` - likely learned or sinusoidal, embedded in the model

To apply Qwen3's approach, we'd need to extract and reimplement Cohere's position encoding logic.

## Alternative: Leverage Existing Cache Interface

Cohere's decoder layers already support `past_key_values`:

```python
layer.forward(
    hidden_states=hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    past_key_values=past_kv,  # ← Already supported!
    cache_position=cache_pos,
    kv_seq_len=kv_len
)
```

This means we could:

1. **Use stateful buffers for `past_key_values`** instead of manually implementing attention
2. Call the existing decoder layers with cached KV tensors
3. Update cache in-place after each layer

### Conceptual Approach

```python
class StatefulCohereDecoder(nn.Module):
    def __init__(self, decoder, max_seq_len=108):
        super().__init__()
        self.decoder = decoder

        # 16 buffers (8 layers x K + V)
        for i in range(8):
            self.register_buffer(f"k_cache_{i}", torch.zeros(..., dtype=torch.float16))
            self.register_buffer(f"v_cache_{i}", torch.zeros(..., dtype=torch.float16))

    def forward(self, input_id, encoder_hidden_states, cross_mask, step):
        # Get embeddings
        embeddings = self.decoder._embedding(...)

        hidden_states = embeddings
        for i, layer in enumerate(self.decoder._decoder.layers):
            k_cache = getattr(self, f"k_cache_{i}")
            v_cache = getattr(self, f"v_cache_{i}")

            # Call layer with past_key_values
            out = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=(k_cache, v_cache),  # ← Use our buffers
                cache_position=step,
                kv_seq_len=step+1
            )

            # Extract outputs (need to check return format)
            hidden_states = out[0]
            new_k, new_v = out[1]  # Assuming it returns updated cache

            # Update cache in-place
            k_cache[:, :, step:step+1, :] = new_k.half()
            v_cache[:, :, step:step+1, :] = new_v.half()

        # Final norm + projection
        ...
```

### Challenge: Return Format Unknown

We don't know what format the decoder layer returns after processing. Does it return:
- `(hidden_states,)` - just output
- `(hidden_states, (new_k, new_v))` - output + updated cache
- Something else?

We'd need to test this empirically.

## Current Status: Stateless Decoder Works

The stateless decoder (O(n^2), no cache) already achieves:

| Sample | Duration | Result |
|--------|----------|--------|
| 1 | 3.5s | ✅ **Perfect** |
| 2 | 14.2s | ⚠️ **Degraded** (different error pattern) |
| 3 | 5.0s | ✅ **Perfect** |

**Key observation**: Sample 2 fails even with stateless approach. This suggests the issue may NOT be cache-related.

## Possible Causes for Sample 2 Failure

1. **Encoder issues**: 14.2s is longest audio - encoder may have numerical issues at longer sequences
2. **Numerical precision**: fp16 accumulation errors over longer sequences
3. **Sequence length effects**: Model may have been trained on shorter examples
4. **Model quality**: May simply be a harder sample for the model

## Recommendations

### Option 1: Debug Sample 2 with Stateless Decoder

Since Sample 2 fails even without cache, investigate:
1. Compare encoder output (CoreML vs PyTorch) at each time step
2. Check for fp16 overflow or numerical instability
3. Try fp32 compute precision for encoder
4. Test with different audio lengths to find threshold

### Option 2: Implement Stateful Cache (High Effort)

To apply Qwen3's technique:
1. Read Cohere's `DecoderAttention` source code from HF cache
2. Manually implement attention with stateful buffers
3. Handle both self-attention and cross-attention
4. Re-implement position encoding
5. Test and debug extensively

**Estimated effort**: 1-2 days of work, uncertain success rate

### Option 3: Hybrid Approach (Lower Effort)

Use Cohere's existing cache interface with stateful buffers:
1. Test what format `layer.forward()` returns with `past_key_values`
2. Create stateful buffers and pass to layers
3. Update in-place after each layer
4. Avoid manual attention reimplementation

**Estimated effort**: 4-6 hours, medium success rate

## Conclusion

**Qwen3's stateful cache approach is proven but not directly applicable to Cohere** due to:
- Custom architecture with non-standard modules
- Nested module access requirements
- Cross-attention complexity
- Position encoding differences

**The stateless decoder already works well** (2/3 samples perfect). The Sample 2 issue may not be cache-related.

**Recommended next step**: Debug Sample 2 failure with stateless decoder first. If it's truly a cache issue, then attempt Option 3 (hybrid approach) rather than full manual reimplementation.

## Appendix: Qwen3's Cache-Length Bug

Qwen3 discovered a critical CoreML bug at cache dimensions 112-126 (just below HEAD_DIM=128):

```
cache_len=110:  h_diff=8.2    (normal)
cache_len=111:  h_diff=7.9    (normal)
cache_len=112:  h_diff=35.4   (!!!)
cache_len=113:  h_diff=89.2   (!!!)
cache_len=120:  h_diff=207.1  (!!!)
cache_len=126:  h_diff=42.3   (!!!)
cache_len=127:  h_diff=7.4    (normal)
```

**Solution**: Cache padding to skip the bad zone.

If we implement stateful cache for Cohere, we MUST implement this padding workaround:

```python
# After initial prompt, pad cache to HEAD_DIM to avoid 112-126 bug zone
if current_cache_len < HEAD_DIM:
    # Pad with zeros to reach HEAD_DIM
    # Mask padded positions with -1e9 in attention
```

This was critical for Qwen3's success (10.4% → 1.6% WER).
