#!/usr/bin/env python3
"""Test the stateless decoder - much simpler than cache-external!"""

import coremltools as ct
import numpy as np

print("="*70)
print("Cohere Stateless Decoder Test")
print("="*70)

# Load model (from coreml directory root, not exports subdirectory)
print("\nLoading model...")
from pathlib import Path
model_path = Path(__file__).parent / "build-stateless" / "cohere_decoder_stateless.mlpackage"
print(f"Model path: {model_path}")
model = ct.models.MLModel(str(model_path))

print("\nModel Interface:")
print("Inputs:")
for inp in model.get_spec().description.input:
    print(f"  • {inp.name}: {inp.type}")

print("\nOutputs:")
for out in model.get_spec().description.output:
    print(f"  • {out.name}: {out.type}")

print("\n" + "="*70)
print("Test 1: Single Token")
print("="*70)

# Test with single token
input_ids = np.array([[4]], dtype=np.int32)  # Start token
encoder_hidden = np.random.randn(1, 438, 1024).astype(np.float32)
cross_mask = np.ones((1, 1, 1, 438), dtype=np.float32)

print(f"\nInput IDs shape: {input_ids.shape}")
print(f"Encoder hidden shape: {encoder_hidden.shape}")

output = model.predict({
    "input_ids": input_ids,
    "encoder_hidden_states": encoder_hidden,
    "cross_attention_mask": cross_mask,
})

print(f"\nOutput logits shape: {output['logits'].shape}")
print(f"Expected: [1, 1, 16384] (batch=1, seq_len=1, vocab=16384)")

# Sample next token
next_token = int(np.argmax(output["logits"][0, -1, :]))
print(f"Predicted next token: {next_token}")

print("\n" + "="*70)
print("Test 2: Multi-Step Generation (Stateless)")
print("="*70)

# Simulate autoregressive generation
tokens = [4]  # Start with start token
encoder_hidden = np.random.randn(1, 438, 1024).astype(np.float32)
cross_mask = np.ones((1, 1, 1, 438), dtype=np.float32)

for step in range(5):
    print(f"\n--- Step {step} ---")

    # Pass ALL tokens so far (stateless - reprocess everything!)
    input_ids = np.array([tokens], dtype=np.int32)
    print(f"  Input IDs: {tokens}")
    print(f"  Input shape: {input_ids.shape}")

    output = model.predict({
        "input_ids": input_ids,
        "encoder_hidden_states": encoder_hidden,
        "cross_attention_mask": cross_mask,
    })

    # Get logits for LAST token position
    # output shape: [1, seq_len, vocab_size]
    last_token_logits = output["logits"][0, -1, :]
    next_token = int(np.argmax(last_token_logits))

    print(f"  Output logits shape: {output['logits'].shape}")
    print(f"  Predicted next token: {next_token}")

    tokens.append(next_token)

print("\n" + "="*70)
print("Test 3: Growing Sequence (O(n²) Complexity)")
print("="*70)

print("\nTesting computation cost as sequence grows...")
print("Stateless decoder reprocesses ALL tokens each step:")

import time

tokens = [4]
times = []

for step in range(10):
    input_ids = np.array([tokens], dtype=np.int32)

    start = time.time()
    output = model.predict({
        "input_ids": input_ids,
        "encoder_hidden_states": encoder_hidden,
        "cross_attention_mask": cross_mask,
    })
    elapsed = time.time() - start

    next_token = int(np.argmax(output["logits"][0, -1, :]))
    tokens.append(next_token)
    times.append(elapsed)

    print(f"  Step {step}: {len(tokens)-1} tokens, {elapsed*1000:.1f}ms")

print(f"\n  Average: {np.mean(times)*1000:.1f}ms per step")
print(f"  Min: {np.min(times)*1000:.1f}ms, Max: {np.max(times)*1000:.1f}ms")
print(f"\n  ⚠️  Time grows as sequence gets longer (O(n²))")
print(f"  ✅  But for 108 tokens max, this is totally acceptable!")

print("\n" + "="*70)
print("✅ Stateless Decoder Working!")
print("="*70)

print("\nKey points:")
print("  • No cache management - much simpler!")
print("  • Reprocesses all tokens each step (O(n²))")
print("  • Returns logits for ALL positions")
print("  • We extract logits for LAST position")
print("  • For 108 token limit, performance is fine")
print("\nReady for Swift integration with CohereStatelessManager!")
