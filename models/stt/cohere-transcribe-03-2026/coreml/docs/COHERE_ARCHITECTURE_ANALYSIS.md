# Cohere Transcribe Architecture Analysis

## Summary

**Good news**: Cohere's architecture is **much simpler than expected**. Implementing stateful cache using Qwen3's approach is **highly feasible** with moderate effort (4-8 hours, not 1-2 days).

## DecoderAttention Implementation (Lines 467-531)

### Structure

```python
class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, layer_idx):
        super().__init__()
        self.hidden_size = hidden_size  # 1024
        self.num_heads = num_heads      # 8
        self.layer_idx = layer_idx
        self.head_dim = hidden_size // num_heads  # 128
        self.scale = self.head_dim**-0.5  # 1/sqrt(128)

        # Simple Linear projections (easy to extract!)
        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, context_states=None, attention_mask=None,
                past_key_values=None, cache_position=None, is_cross_attention=False,
                kv_seq_len=None):
        # 1. Project query
        query = self._reshape(self.query_net(hidden_states))

        # 2. Determine source (self-attention vs cross-attention)
        source = hidden_states if context_states is None else context_states

        # 3. Handle cache
        if past_key_values is not None:
            # Extract cache layer
            if isinstance(past_key_values, EncoderDecoderCache):
                if is_cross_attention:
                    cache_layer = past_key_values.cross_attention_cache
                else:
                    cache_layer = past_key_values.self_attention_cache
            elif isinstance(past_key_values, DynamicCache):
                cache_layer = past_key_values

        # 4. Project K, V and update cache
        key = self._reshape(self.key_net(source))
        value = self._reshape(self.value_net(source))

        if cache_layer is not None:
            cache_kwargs = None
            if not is_cross_attention and cache_position is not None:
                cache_kwargs = {"cache_position": cache_position}

            # Update cache (THIS IS WHERE WE'LL INJECT STATEFUL BUFFERS)
            key, value = cache_layer.update(key, value, self.layer_idx,
                                           cache_kwargs=cache_kwargs)

            # For StaticCache, truncate to kv_seq_len
            if not is_cross_attention and kv_seq_len is not None:
                key = key[:, :, :kv_seq_len]
                value = value[:, :, :kv_seq_len]

        # 5. Attention (uses PyTorch's built-in efficient implementation)
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scale
        )

        # 6. Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(hidden_states.shape[0], hidden_states.shape[1], self.hidden_size)
        )
        return self.out_projection(attn_output)

    def _reshape(self, x):
        # [batch, time, hidden] -> [batch, heads, time, head_dim]
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
```

### Key Observations

1. **Simple projections**: Q/K/V are just `nn.Linear` - direct weight access
2. **Standard attention**: Uses `F.scaled_dot_product_attention` (PyTorch built-in)
3. **Cache interface**: Uses `cache_layer.update()` - we can replace with stateful buffers
4. **No RoPE**: Position encoding handled separately via lookup table
5. **Standard head_dim**: 128 (same as Qwen3)

## Position Encoding (Lines 448-464)

### FixedPositionalEncoding

```python
class FixedPositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_sequence_length=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length

        # Precompute sinusoidal position encodings
        pos_enc = torch.zeros(max_sequence_length, hidden_size)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        coef = -math.log(10000.0) / hidden_size
        div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.div_(math.sqrt(hidden_size))  # Scale by 1/sqrt(d_model)

        # Store as buffer (non-trainable parameter)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, position_ids):
        # Simple lookup: select rows from pos_enc buffer
        return torch.index_select(self.pos_enc, 0, position_ids.reshape(-1)).reshape(*position_ids.shape, -1)
```

**This is MUCH simpler than Qwen3's RoPE!**

- Qwen3: Applies rotations to Q and K at every layer (requires cos/sin inputs)
- Cohere: Simple lookup table, added once at embedding layer

We can easily include this in our stateful wrapper.

## TransformerDecoderLayer (Lines 548-596)

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, inner_size, num_heads, layer_idx, hidden_act="relu"):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.first_sub_layer = DecoderAttention(hidden_size, num_heads, layer_idx=layer_idx)  # Self-attn
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.second_sub_layer = DecoderAttention(hidden_size, num_heads, layer_idx=layer_idx)  # Cross-attn
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.third_sub_layer = DecoderFeedForward(hidden_size, inner_size, hidden_act=hidden_act)

    def forward(self, hidden_states, encoder_hidden_states=None,
                self_attention_mask=None, cross_attention_mask=None,
                past_key_values=None, cache_position=None, kv_seq_len=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)
        self_out = self.first_sub_layer(
            hidden_states,
            context_states=None,  # Self-attention
            attention_mask=self_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            is_cross_attention=False,
            kv_seq_len=kv_seq_len,
        )
        hidden_states = residual + self_out

        # Cross-attention
        residual = hidden_states
        hidden_states = self.layer_norm_2(hidden_states)
        cross_out = self.second_sub_layer(
            hidden_states,
            context_states=encoder_hidden_states,  # Cross-attention
            attention_mask=cross_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            is_cross_attention=True,
        )
        hidden_states = residual + cross_out

        # FFN
        residual = hidden_states
        hidden_states = self.layer_norm_3(hidden_states)
        hidden_states = residual + self.third_sub_layer(hidden_states)

        return hidden_states
```

**Standard transformer decoder layer** with:
- Self-attention (needs KV cache)
- Cross-attention (no cache needed - encoder is static)
- Feed-forward network

## Implementing Stateful Cache: The Plan

### Approach: Manual Attention with Stateful Buffers (Qwen3 Style)

```python
class StatefulCohereDecoder(nn.Module):
    def __init__(self, decoder_wrapper, max_seq_len=108):
        super().__init__()

        # Store original modules
        self.embedding = decoder_wrapper._embedding
        self.layers = decoder_wrapper._decoder.layers
        self.final_norm = decoder_wrapper._decoder.final_layer_norm
        self.num_layers = len(self.layers)

        # Register 16 state buffers (8 layers x K + V for self-attention only)
        for i in range(self.num_layers):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, 8, max_seq_len, 128, dtype=torch.float16),
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, 8, max_seq_len, 128, dtype=torch.float16),
            )

    def forward(self, input_id, encoder_hidden_states, cross_attention_mask, step):
        # 1. Get embeddings (includes position encoding)
        positions = step.unsqueeze(0)  # Current position
        hidden_states = self.embedding(input_id, positions)

        # 2. Process through decoder layers
        for layer_idx, layer in enumerate(self.layers):
            k_cache = getattr(self, f"k_cache_{layer_idx}")
            v_cache = getattr(self, f"v_cache_{layer_idx}")

            # --- Self-attention with stateful cache ---
            residual = hidden_states
            hidden_states = layer.layer_norm_1(hidden_states)

            # Manual self-attention computation
            hidden_states = self._manual_self_attention(
                hidden_states=hidden_states,
                attention_module=layer.first_sub_layer,
                k_cache=k_cache,
                v_cache=v_cache,
                step=step,
            )
            hidden_states = residual + hidden_states

            # --- Cross-attention (no cache) ---
            residual = hidden_states
            hidden_states = layer.layer_norm_2(hidden_states)
            cross_out = layer.second_sub_layer(
                hidden_states,
                context_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                past_key_values=None,  # No cache for cross-attention
                is_cross_attention=True,
            )
            hidden_states = residual + cross_out

            # --- FFN ---
            residual = hidden_states
            hidden_states = layer.layer_norm_3(hidden_states)
            hidden_states = residual + layer.third_sub_layer(hidden_states)

        # 3. Final norm
        hidden_states = self.final_norm(hidden_states)

        return hidden_states

    def _manual_self_attention(self, hidden_states, attention_module,
                                k_cache, v_cache, step):
        """Manually compute self-attention with stateful KV cache."""
        step_int = int(step.item())
        end_step = step_int + 1

        # 1. Project Q, K, V
        query = attention_module.query_net(hidden_states)
        key = attention_module.key_net(hidden_states)
        value = attention_module.value_net(hidden_states)

        # 2. Reshape to multi-head
        query = attention_module._reshape(query)   # [1, 8, 1, 128]
        key = attention_module._reshape(key)       # [1, 8, 1, 128]
        value = attention_module._reshape(value)   # [1, 8, 1, 128]

        # 3. In-place cache update (CoreML detects as state mutation)
        k_cache[:, :, step_int:end_step, :] = key.half()
        v_cache[:, :, step_int:end_step, :] = value.half()

        # 4. Read valid cache entries and cast to fp32
        k_full = k_cache[:, :, :end_step, :].float()
        v_full = v_cache[:, :, :end_step, :].float()

        # 5. Create causal mask
        causal_mask = torch.zeros(1, 1, 1, end_step, device=hidden_states.device)

        # 6. Attention (use PyTorch's built-in, same as Cohere)
        attn_output = F.scaled_dot_product_attention(
            query, k_full, v_full,
            attn_mask=causal_mask,
            dropout_p=0.0,
            scale=attention_module.scale
        )

        # 7. Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(hidden_states.shape[0], hidden_states.shape[1], attention_module.hidden_size)
        )
        return attention_module.out_projection(attn_output)
```

## Complexity Comparison

| Task | Qwen3 (28 layers, GQA, RoPE) | Cohere (8 layers, standard, lookup) |
|------|------------------------------|--------------------------------------|
| **Manual Q/K/V projection** | Extract from `q_proj`, `k_proj`, `v_proj` | Extract from `query_net`, `key_net`, `value_net` |
| **Position encoding** | Apply RoPE rotation to Q and K | Simple embedding lookup (already done) |
| **Attention computation** | Manual matmul + softmax | Use `F.scaled_dot_product_attention` |
| **GQA head expansion** | Repeat KV heads (8 → 16) | Not needed (8 Q heads = 8 KV heads) |
| **QK norms** | Apply layer norms to Q and K | Not needed |
| **Cross-attention** | Not present | Need to handle separately (no cache) |

**Verdict**: Cohere is significantly simpler than Qwen3.

## Implementation Checklist

### Phase 1: Basic Stateful Decoder (4 hours)

- [ ] Create `StatefulCohereDecoder` wrapper class
- [ ] Register 16 state buffers (8 layers × K/V)
- [ ] Extract embedding module (includes position encoding)
- [ ] Implement manual self-attention with stateful cache
- [ ] Handle cross-attention (pass-through, no cache)
- [ ] Handle FFN (pass-through)
- [ ] Export to CoreML with `ct.StateType`

### Phase 2: Testing (2 hours)

- [ ] Test with simple inputs (trace validation)
- [ ] Compare outputs: eager vs traced vs CoreML
- [ ] Test on LibriSpeech samples

### Phase 3: Cache Padding (if needed) (2 hours)

- [ ] Implement cache padding to 128 (HEAD_DIM) to avoid 112-126 bug zone
- [ ] Update attention mask to hide padded positions
- [ ] Test on previously failing samples

## Expected Outcomes

### Optimistic (70% probability)
- Stateful decoder works on first try
- WER: 1.6% (matching PyTorch baseline)
- Sample 2 still has issues (likely not cache-related)
- Speed: 2-3x faster than stateless (O(n^2) → O(n))

### Realistic (25% probability)
- Stateful decoder works after debugging
- Encounters cache-length bug (needs padding workaround)
- WER: 2-5% after fixes
- Sample 2 may improve or may still fail

### Pessimistic (5% probability)
- Fundamental CoreML incompatibility we haven't discovered
- Falls back to stateless decoder

## Next Steps

1. **Implement Phase 1** (4 hours): Create stateful decoder export script
2. **Test Phase 2** (2 hours): Validate on LibriSpeech
3. **If needed, Phase 3** (2 hours): Implement cache padding

Total estimated effort: **6-8 hours** (vs initial estimate of 1-2 days)

## Conclusion

**Cohere's architecture is simpler than expected**, making stateful cache implementation **highly feasible**:

✅ Simple Linear projections (easy weight access)
✅ Standard attention (can use `F.scaled_dot_product_attention`)
✅ Simple position encoding (lookup table, not RoPE)
✅ No GQA (8 heads = 8 heads)
✅ No QK norms
⚠️ Cross-attention needs separate handling (but no cache needed)

The main complexity is properly managing the two attention mechanisms (self + cross) in each layer, but this is straightforward once the pattern is established.
