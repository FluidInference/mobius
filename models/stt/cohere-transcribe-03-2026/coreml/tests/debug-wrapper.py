#!/usr/bin/env python3
"""Debug the wrapper to see what's in updated_cache."""

import torch
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

print("="*70)
print("Wrapper Debug")
print("="*70)

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()

decoder = model.transf_decoder
log_softmax = model.log_softmax

# Manual forward pass
input_id = torch.tensor([[13764]], dtype=torch.long)
encoder_hidden = torch.randn(1, 376, 1024)
step = torch.tensor([0], dtype=torch.int32)

print("\nBuilding cache...")
self_attention_cache = DynamicCache()
cross_attention_cache = DynamicCache()
past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

print(f"Initial self_attention_cache has {len(self_attention_cache.key_cache)} layers")
print(f"Initial cross_attention_cache has {len(cross_attention_cache.key_cache)} layers")

# Positions
positions_input = step.view(1, 1).long()

# Attention masks
self_attention_mask = None  # Will be created by decoder
cross_mask = torch.ones(1, 376)

print("\nCalling decoder...")
with torch.no_grad():
    decoder_outputs, updated_cache = decoder(
        input_ids=input_id,
        positions=positions_input,
        encoder_hidden_states=encoder_hidden,
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_mask,
        past_key_values=past_key_values,
        cache_position=None,
        kv_seq_len=None,
    )

print(f"\nDecoder output shape: {decoder_outputs.shape}")
print(f"Updated cache type: {type(updated_cache)}")
print(f"Updated cache self_attention: {type(updated_cache.self_attention_cache)}")
print(f"Updated cache cross_attention: {type(updated_cache.cross_attention_cache)}")

self_attn_cache = updated_cache.self_attention_cache
print(f"\nSelf-attention cache has {len(self_attn_cache.key_cache)} layers")

if len(self_attn_cache.key_cache) > 0:
    print(f"Layer 0 key_cache shape: {self_attn_cache.key_cache[0].shape}")
    print(f"Layer 0 value_cache shape: {self_attn_cache.value_cache[0].shape}")

    # Check if non-zero
    layer_k = self_attn_cache.key_cache[0]
    layer_v = self_attn_cache.value_cache[0]
    k_norm = torch.sqrt(torch.sum(layer_k**2))
    v_norm = torch.sqrt(torch.sum(layer_v**2))
    print(f"Layer 0 key norm: {k_norm.item():.6f}")
    print(f"Layer 0 value norm: {v_norm.item():.6f}")
else:
    print("❌ Self-attention cache is EMPTY!")
