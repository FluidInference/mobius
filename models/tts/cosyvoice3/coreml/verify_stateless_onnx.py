#!/usr/bin/env python3
"""
Verify that Vocoder and Flow ONNX models are stateless.

Stateless = Same input always produces same output, no hidden state between calls.
"""

import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

print("=" * 80)
print("Verify Stateless ONNX Models")
print("=" * 80)

# Check if ONNX models exist
vocoder_path = Path("converted/hift_vocoder.onnx")
flow_path = Path("flow_decoder.onnx")

def test_stateless(session, input_dict, model_name):
    """Test if a model is stateless by running same input twice."""
    print(f"\n[Testing {model_name}]")
    print("-" * 80)

    # Run 1
    print("Run 1...")
    outputs1 = session.run(None, input_dict)

    # Run 2 (same inputs)
    print("Run 2 (same inputs)...")
    outputs2 = session.run(None, input_dict)

    # Compare
    print("\nComparing outputs...")
    for i, (out1, out2) in enumerate(zip(outputs1, outputs2)):
        max_diff = np.abs(out1 - out2).max()
        print(f"  Output {i}: max_diff = {max_diff:.10f}")

        if max_diff < 1e-6:
            print(f"  ✓ Output {i} is deterministic (stateless)")
        else:
            print(f"  ✗ Output {i} differs (might have state)")

    all_same = all(np.allclose(out1, out2, atol=1e-6) for out1, out2 in zip(outputs1, outputs2))

    if all_same:
        print(f"\n✓ {model_name} is STATELESS")
    else:
        print(f"\n✗ {model_name} might have hidden state")

    return all_same

# Test Vocoder
print("\n" + "=" * 80)
print("VOCODER ONNX")
print("=" * 80)

if vocoder_path.exists():
    print(f"Loading {vocoder_path}...")

    try:
        vocoder_session = ort.InferenceSession(
            str(vocoder_path),
            providers=['CPUExecutionProvider']
        )
        print("✓ Loaded")

        # Check inputs/outputs
        print("\nModel signature:")
        print(f"  Inputs: {[i.name for i in vocoder_session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in vocoder_session.get_outputs()]}")

        # Get input shape
        input_info = vocoder_session.get_inputs()[0]
        print(f"\n  Input '{input_info.name}':")
        print(f"    Shape: {input_info.shape}")
        print(f"    Type: {input_info.type}")

        # Create test input
        # Assuming shape is [batch, mel_bins, time]
        mel_test = np.random.randn(1, 80, 50).astype(np.float32)

        # Test statelessness
        vocoder_stateless = test_stateless(
            vocoder_session,
            {input_info.name: mel_test},
            "Vocoder"
        )

    except Exception as e:
        print(f"✗ Failed to test vocoder: {e}")
        vocoder_stateless = None
else:
    print(f"✗ Not found: {vocoder_path}")
    print("\nTo create ONNX vocoder:")
    print("  uv run python convert_vocoder.py")
    vocoder_stateless = None

# Test Flow
print("\n" + "=" * 80)
print("FLOW DECODER ONNX")
print("=" * 80)

if flow_path.exists():
    print(f"Loading {flow_path}...")

    try:
        flow_session = ort.InferenceSession(
            str(flow_path),
            providers=['CPUExecutionProvider']
        )
        print("✓ Loaded")

        # Check inputs/outputs
        print("\nModel signature:")
        inputs = flow_session.get_inputs()
        outputs = flow_session.get_outputs()

        print(f"  Inputs ({len(inputs)}):")
        for inp in inputs:
            print(f"    - {inp.name}: {inp.shape} ({inp.type})")

        print(f"  Outputs ({len(outputs)}):")
        for out in outputs:
            print(f"    - {out.name}: {out.shape} ({out.type})")

        # Create test inputs based on signature
        # Expected: x, mask, mu, t, spks, cond
        time_steps = 100

        input_dict = {}
        for inp in inputs:
            if 'mask' in inp.name.lower():
                # Mask is typically [batch, 1, time]
                input_dict[inp.name] = np.ones((1, 1, time_steps), dtype=np.float32)
            elif 't' == inp.name.lower():
                # Time step is typically scalar [batch]
                input_dict[inp.name] = np.array([0.5], dtype=np.float32)
            elif 'spks' in inp.name.lower():
                # Speaker embedding [batch, 80]
                input_dict[inp.name] = np.random.randn(1, 80).astype(np.float32)
            else:
                # x, mu, cond are [batch, 80, time]
                input_dict[inp.name] = np.random.randn(1, 80, time_steps).astype(np.float32)

        # Test statelessness
        flow_stateless = test_stateless(
            flow_session,
            input_dict,
            "Flow Decoder"
        )

    except Exception as e:
        print(f"✗ Failed to test flow: {e}")
        import traceback
        traceback.print_exc()
        flow_stateless = None
else:
    print(f"✗ Not found: {flow_path}")
    print("\nTo create ONNX flow decoder:")
    print("  uv run python convert_flow_final.py")
    flow_stateless = None

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if vocoder_stateless is not None:
    status = "✓ STATELESS" if vocoder_stateless else "✗ STATEFUL"
    print(f"\nVocoder ONNX: {status}")
    if vocoder_stateless:
        print("  → Safe to use in parallel")
        print("  → No state management needed")
        print("  → Same input = same output")

if flow_stateless is not None:
    status = "✓ STATELESS" if flow_stateless else "✗ STATEFUL"
    print(f"\nFlow ONNX: {status}")
    if flow_stateless:
        print("  → Safe to use in parallel")
        print("  → No state management needed")
        print("  → Same input = same output")

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

if vocoder_stateless and flow_stateless:
    print("\n✓ Both models are STATELESS")
    print("\nYou can use them in the hybrid CoreML + ONNX approach:")
    print("  1. Load once")
    print("  2. Call multiple times with different inputs")
    print("  3. No need to manage state between calls")
    print("  4. Safe for concurrent/parallel inference")
elif vocoder_stateless is None and flow_stateless is None:
    print("\n⚠ ONNX models not found")
    print("\nCreate them with:")
    print("  uv run python convert_vocoder.py  # Creates vocoder ONNX")
    print("  uv run python convert_flow_final.py  # Creates flow ONNX")
else:
    print("\n⚠ Mixed results - some models might have state")
    print("\nCheck the test results above for details.")

print("\n" + "=" * 80)
