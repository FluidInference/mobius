#!/usr/bin/env python3
"""Export Cohere decoder with manual cache handling - no DynamicCache."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq

print("="*70)
print("Cohere Decoder Export - Manual Cache Management")
print("="*70)

print("\n[1/3] Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()
print("   ✓ Loaded")

print("\n[2/3] Testing inference without cache...")
# Test if we can run the decoder without past_key_values (no cache)
decoder = model.transf_decoder
log_softmax = model.log_softmax

input_id = torch.tensor([[13764]], dtype=torch.long)
encoder_hidden = torch.randn(1, 376, 1024)
positions = torch.tensor([[0]], dtype=torch.long)
cross_mask = torch.ones(1, 376)

with torch.no_grad():
    # Call decoder WITHOUT cache
    decoder_outputs, _ = decoder(
        input_ids=input_id,
        positions=positions,
        encoder_hidden_states=encoder_hidden,
        self_attention_mask=None,
        cross_attention_mask=cross_mask,
        past_key_values=None,  # No cache
        cache_position=None,
        kv_seq_len=None,
    )
    logits = log_softmax(decoder_outputs)

print(f"   ✓ No-cache inference works! Logits shape: {logits.shape}")

print("\n[3/3] Analysis:")
print("   The decoder CAN run without cache (past_key_values=None)")
print("   This means we could use a stateless approach:")
print("   - Store all previous tokens")
print("   - Re-process them all at each step (no cache)")
print("   - Slower but avoids cache complexity")
print("\n   Alternative: Store generated tokens and reprocess on each step")
print("   This is O(n^2) but simpler and might be acceptable for < 200 tokens")

print("\n" + "="*70)
print("Conclusion: Stateless decoding is possible")
print("="*70)
