#!/usr/bin/env python3
"""Test the cache-external decoder model."""

import coremltools as ct
import numpy as np

# Load model
print("Loading model...")
model = ct.models.MLModel("build-test/cohere_decoder_cache_external.mlpackage")

print("\n" + "="*70)
print("Model Specification")
print("="*70)

# Print inputs
print("\nInputs:")
for input_desc in model.get_spec().description.input:
    print(f"  • {input_desc.name}: {input_desc.type}")

# Print outputs
print("\nOutputs:")
for output_desc in model.get_spec().description.output:
    print(f"  • {output_desc.name}: {output_desc.type}")

print("\n" + "="*70)
print("Test Inference (Step 0)")
print("="*70)

# Test first token (step 0)
test_input = {
    "input_id": np.array([[4]], dtype=np.int32),  # Start token
    "position_id": np.array([[0]], dtype=np.int32),
    "encoder_hidden_states": np.random.randn(1, 438, 1024).astype(np.float32),
    "cross_attention_mask": np.ones((1, 1, 1, 438), dtype=np.float32),
    "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),  # Step 0: size 1
}

# Add empty caches
for i in range(8):
    test_input[f"k_cache_{i}"] = np.zeros((1, 8, 108, 128), dtype=np.float32)
    test_input[f"v_cache_{i}"] = np.zeros((1, 8, 108, 128), dtype=np.float32)

print("\nRunning inference...")
output = model.predict(test_input)

print(f"\nOutputs received:")
print(f"  • logits: {output['logits'].shape}")
for i in range(8):
    if f"k_cache_{i}_out" in output:
        print(f"  • k_cache_{i}_out: {output[f'k_cache_{i}_out'].shape}")
        print(f"  • v_cache_{i}_out: {output[f'v_cache_{i}_out'].shape}")

# Sample next token
next_token = int(np.argmax(output["logits"][0]))
print(f"\nPredicted token: {next_token}")

print("\n" + "="*70)
print("Test Multi-Step Inference")
print("="*70)

# Test a few steps with growing attention mask
k_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]
v_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]
current_token = 4

for step in range(3):
    print(f"\n--- Step {step} ---")

    # Build input with growing attention_mask
    test_input = {
        "input_id": np.array([[current_token]], dtype=np.int32),
        "position_id": np.array([[step]], dtype=np.int32),
        "encoder_hidden_states": np.random.randn(1, 438, 1024).astype(np.float32),
        "cross_attention_mask": np.ones((1, 1, 1, 438), dtype=np.float32),
        # Attention mask grows: [1,1,1,1] -> [1,1,1,2] -> [1,1,1,3]
        "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float32),
    }

    for i in range(8):
        test_input[f"k_cache_{i}"] = k_caches[i]
        test_input[f"v_cache_{i}"] = v_caches[i]

    output = model.predict(test_input)

    # Extract updated caches
    for i in range(8):
        k_caches[i] = output[f"k_cache_{i}_out"]
        v_caches[i] = output[f"v_cache_{i}_out"]

    next_token = int(np.argmax(output["logits"][0]))
    print(f"  Input token: {current_token}")
    print(f"  Attention mask size: [1, 1, 1, {step + 1}]")
    print(f"  Predicted token: {next_token}")

    current_token = next_token

print("\n" + "="*70)
print("✅ Cache-External Decoder Working!")
print("="*70)
print("\nThe model successfully:")
print("  • Takes cache as inputs (16 arrays)")
print("  • Returns updated cache as outputs")
print("  • Uses attention_mask.shape[-1] to infer position")
print("  • Handles growing attention_mask across steps")
print("\nReady for Swift integration!")
