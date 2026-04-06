# Stateless vs Stateful Decoder: Why Simpler is Better

This document explains why we created a **stateless decoder** (Parakeet approach) in addition to the stateful decoder, and why it might actually be the better choice.

## TL;DR

**Stateless decoder** (like Parakeet):
- ✅ Simpler code (no cache management)
- ✅ Works on macOS 14 (no State API requirement)
- ✅ Can compile to `.mlmodelc` for better ANE optimization
- ✅ Easier to debug
- ⚠️ ~10x more compute at step 108 (but acceptable for 108 token limit)

**Stateful decoder** (original):
- ✅ O(n) complexity (theoretically faster)
- ❌ Requires macOS 15+ (CoreML State API)
- ❌ Can't compile to `.mlmodelc` (stuck with `.mlpackage`)
- ❌ Complex cache management (more bugs)
- ❌ Harder to debug

**Verdict**: For 108 token limit, stateless is probably better for most use cases.

---

## Background: Why Did We Use Stateful?

We followed Cohere's upstream implementation, which uses:
- Transformer decoder with self-attention
- KV cache to avoid recomputing attention for previous tokens
- Stateful design with CoreML State API

This seemed like the "modern" approach, using Apple's latest APIs.

**But**: Parakeet proved that stateless works great for ASR decoders!

---

## Decoder Comparison

### Architecture

**Stateful Decoder** (`export-decoder-stateful.py`):
```python
class StatefulCohereDecoder(nn.Module):
    def __init__(self, decoder_wrapper, lm_head, max_seq_len=108):
        super().__init__()

        # Register 16 cache buffers (8 layers × K/V)
        for i in range(num_layers):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, 8, 108, 128, dtype=torch.float16),
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, 8, 108, 128, dtype=torch.float16),
            )

    def forward(self, input_id, encoder_hidden, ...):
        # Infer position from attention_mask shape
        past_kv_len = attention_mask.shape[-1] - 1

        # Update cache in-place at specific position
        k_cache[:, :, past_kv_len:end_step, :] = key.half()
        v_cache[:, :, past_kv_len:end_step, :] = value.half()

        # Read full cache and compute attention
        k_full = k_cache[:, :, :end_step, :].float()
        attn_output = F.scaled_dot_product_attention(query, k_full, ...)
```

**Lines of code**: ~250
**Complexity**: High (cache slicing, type conversion, position tracking)
**State buffers**: 16 (8 layers × 2)

---

**Stateless Decoder** (`export-decoder-stateless.py`):
```python
class StatelessCohereDecoder(nn.Module):
    def __init__(self, decoder_wrapper, lm_head):
        super().__init__()

        # Just store modules - NO cache buffers!
        self.embedding = decoder_wrapper._embedding
        self.layers = decoder_wrapper._decoder.layers
        self.final_norm = decoder_wrapper._decoder.final_layer_norm
        self.lm_head = lm_head

    def forward(self, input_ids, encoder_hidden, ...):
        # Process ALL tokens (not just new one)
        hidden_states = self.embedding(input_ids, position_ids)

        for layer in self.layers:
            # Just call the original modules with use_cache=False
            self_attn_out = layer.first_sub_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                past_key_values=None,  # No cache!
            )
            # ... rest is standard transformer layer

        return logits
```

**Lines of code**: ~140
**Complexity**: Low (just forward pass)
**State buffers**: 0

---

## Performance Comparison

### Computational Complexity

**Stateful**:
- Step 1: 1 token → O(1) attention
- Step 50: 1 token → O(1) attention
- Step 108: 1 token → O(1) attention
- **Total**: O(n) where n = sequence length

**Stateless**:
- Step 1: 1 token → O(1) attention
- Step 50: 50 tokens → O(50²) attention
- Step 108: 108 tokens → O(108²) attention
- **Total**: O(n²) where n = sequence length

**At 108 tokens**:
- Stateful: ~108 attention operations
- Stateless: ~11,664 attention operations
- **Ratio**: ~100x more compute

**But**: ANE is FAST at matrix operations. The real question is wall-clock time, not operation count.

### Memory Usage

**Stateful**:
- 16 cache buffers: 8 layers × 2 (K/V) × (1, 8, 108, 128) × fp16
- **Cache size**: ~1.7 MB total
- **Advantage**: Memory-efficient

**Stateless**:
- No cache buffers
- Recomputes everything from scratch
- **Memory**: Just model weights + activations
- **Advantage**: Simpler memory model

### ANE Optimization

**Stateful**:
- `.mlpackage` format (ML Program)
- Cannot compile to `.mlmodelc`
- **ANE utilization**: Good, but not optimal

**Stateless**:
- `.mlpackage` format initially
- **Can compile to `.mlmodelc`** (like Parakeet!)
- **ANE utilization**: Better (compiled format)

```bash
# Compile stateless decoder to .mlmodelc
xcrun coremlcompiler compile \
    cohere_decoder_stateless.mlpackage \
    output_dir/

# Result: cohere_decoder_stateless.mlmodelc
# Better ANE optimization, faster load time
```

**This might completely offset the O(n²) overhead!**

### macOS Version Support

| Decoder | macOS 14 | macOS 15+ |
|---------|----------|-----------|
| **Stateful** | ❌ No (needs State API) | ✅ Yes |
| **Stateless** | ✅ Yes | ✅ Yes |

**Stateless works on macOS 14** - huge advantage for broader device support.

---

## Quality Comparison

Both should produce **identical results** (same model weights, same architecture).

The only difference is **how** they compute attention:
- Stateful: Cached attention (efficient)
- Stateless: Recomputed attention (inefficient but correct)

**Expected WER**: ~16-17% on LibriSpeech test-clean (both)

---

## Real-World Performance Estimate

For **108 token sequence** on **Apple M1/M2/M3**:

### Stateful Decoder
- Step 1-10: ~5ms per step
- Step 50: ~5ms per step
- Step 108: ~5ms per step
- **Total latency**: ~540ms for 108 tokens

### Stateless Decoder (.mlpackage)
- Step 1-10: ~10ms per step
- Step 50: ~50ms per step (50 tokens to process)
- Step 108: ~108ms per step (108 tokens to process)
- **Total latency**: ~3-4 seconds for 108 tokens

### Stateless Decoder (.mlmodelc, compiled)
- Step 1-10: ~5ms per step (better ANE optimization)
- Step 50: ~25ms per step (ANE acceleration)
- Step 108: ~54ms per step (ANE acceleration)
- **Total latency**: ~1.5-2 seconds for 108 tokens

**Hypothesis**: Compiled stateless might be only **2-3x slower** than stateful, not 100x!

And for typical transcription (20-40 tokens), the difference might be **negligible**.

---

## Debugging and Maintainability

### Stateful Decoder Issues

From our development experience:

1. **Cache truncation bugs** (multiple iterations to fix)
2. **Position tracking** (had to infer from attention_mask shape)
3. **Type conversions** (fp32 → fp16 for cache, back to fp32 for attention)
4. **Slice indexing** (had to avoid `.item()` for CoreML tracing)
5. **State mutation detection** (CoreML needs to detect in-place updates)

**Bug count during development**: 7+ cache-related bugs

### Stateless Decoder Issues

**Bug count during development**: TBD (but expect close to 0)

No cache = no cache bugs!

---

## Use Case Recommendations

### When to Use Stateful

- ✅ You need **minimum latency** (real-time transcription)
- ✅ You're on **macOS 15+** (State API available)
- ✅ You're generating **long sequences** (>50 tokens regularly)
- ✅ You don't mind **complexity** (willing to debug cache issues)

### When to Use Stateless

- ✅ You want **maximum compatibility** (macOS 14+)
- ✅ You want **simpler code** (easier to maintain)
- ✅ You want **better ANE optimization** (can compile to .mlmodelc)
- ✅ Your sequences are typically **short** (<40 tokens)
- ✅ You're okay with **slightly higher latency** (but maybe not much!)

### For Production: Stateless Probably Better

Reasons:
1. **Works on more devices** (macOS 14+)
2. **Fewer bugs** (no cache management)
3. **Better optimization** (compilable to .mlmodelc)
4. **Good enough performance** (for 108 token limit)

The **O(n²) complexity is a red herring** when:
- Sequence is short (108 max)
- ANE is fast at matrix ops
- Compiled .mlmodelc provides better optimization

---

## Benchmark Results

### Stateful Decoder (macOS 15+)

**LibriSpeech test-clean** (10 samples):
- Average WER: 16.44%
- Perfect matches: 50%
- Good (<30% WER): 80%
- Average latency: ~600ms per sample

### Stateless Decoder (macOS 14+)

**LibriSpeech test-clean** (10 samples):
- Average WER: [TODO: Run test]
- Perfect matches: [TODO]
- Good (<30% WER): [TODO]
- Average latency (.mlpackage): [TODO]
- Average latency (.mlmodelc): [TODO]

---

## Parakeet Precedent

**Parakeet TDT** uses a **stateless decoder**:
- RNN-T decoder (LSTM/GRU-based)
- No KV cache needed (RNN architecture)
- Compiled to `.mlmodelc`
- Excellent performance on ANE

**Key insight**: For ASR with bounded output length, stateless works great!

**Qwen3 also has stateless variant** for simpler use cases.

---

## Conclusion

We **over-engineered** the Cohere decoder by using the stateful approach.

**Stateless decoder** (Parakeet approach):
- Simpler
- More compatible (macOS 14+)
- Better optimized (compilable to .mlmodelc)
- Probably "good enough" performance for 108 tokens

**Recommendation**:
- Default to **stateless** for most use cases
- Use **stateful** only if you need absolute minimum latency

The complexity trade-off isn't worth it for marginal performance gains on short sequences.

---

## Next Steps

1. **Test stateless decoder**:
   ```bash
   uv run exports/export-decoder-stateless.py
   uv run test_stateless_decoder.py
   ```

2. **Compile to .mlmodelc**:
   ```bash
   xcrun coremlcompiler compile \
       build/cohere_decoder_stateless.mlpackage \
       build/
   ```

3. **Benchmark both**:
   - Stateful (.mlpackage, macOS 15+)
   - Stateless (.mlpackage, macOS 14+)
   - Stateless (.mlmodelc, macOS 14+)

4. **Compare**:
   - Quality (WER)
   - Latency
   - Memory usage
   - ANE utilization

5. **Choose default** based on results

My prediction: **Stateless .mlmodelc will be the winner** for most use cases.
