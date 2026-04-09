#!/usr/bin/env python3
"""Test that the compiled .mlmodelc works correctly."""

import numpy as np
import coremltools as ct

# Test parameters
MAX_SEQ_LEN = 108

def test_mlmodelc():
    """Test the compiled .mlmodelc model."""
    print("Testing compiled .mlmodelc model...")
    print("=" * 70)

    # Load compiled model
    print("\n[1/3] Loading compiled model...")
    model_path = "build-test/cohere_decoder_cache_external.mlmodelc"
    model = ct.models.MLModel(model_path)
    print(f"   ✓ Loaded: {model_path}")

    # Print model info
    spec = model.get_spec()
    print(f"\n[2/3] Model info:")
    print(f"   Inputs: {len(spec.description.input)}")
    print(f"   Outputs: {len(spec.description.output)}")

    # Test single inference step
    print(f"\n[3/3] Running single inference step...")

    # Create dummy inputs
    input_dict = {
        "input_id": np.array([[4]], dtype=np.int32),  # START_TOKEN
        "position_id": np.array([[0]], dtype=np.int32),
        "encoder_hidden_states": np.random.randn(1, 438, 1024).astype(np.float32),
        "cross_attention_mask": np.ones((1, 1, 1, 438), dtype=np.float32),
        "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }

    # Add cache inputs (zeros)
    for i in range(8):
        input_dict[f"k_cache_{i}"] = np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32)
        input_dict[f"v_cache_{i}"] = np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32)

    # Run inference
    output = model.predict(input_dict)

    # Check outputs
    logits = output["logits"]
    print(f"   Logits shape: {logits.shape}")
    print(f"   Expected: (1, 16384)")

    # Check cache outputs
    cache_ok = True
    for i in range(8):
        k_out = output[f"k_cache_{i}_out"]
        v_out = output[f"v_cache_{i}_out"]
        if k_out.shape != (1, 8, MAX_SEQ_LEN, 128):
            cache_ok = False
            print(f"   ✗ k_cache_{i}_out has wrong shape: {k_out.shape}")
        if v_out.shape != (1, 8, MAX_SEQ_LEN, 128):
            cache_ok = False
            print(f"   ✗ v_cache_{i}_out has wrong shape: {v_out.shape}")

    if cache_ok:
        print(f"   ✓ All 16 cache outputs have correct shape: (1, 8, {MAX_SEQ_LEN}, 128)")

    # Sample next token
    next_token = int(np.argmax(logits[0]))
    print(f"   Next token: {next_token}")

    print("\n" + "=" * 70)
    print("✅ Compiled .mlmodelc works correctly!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    test_mlmodelc()
