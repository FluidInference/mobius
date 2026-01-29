"""Verify Mimi streaming decoder CoreML model.

NOTE: The Mimi decoder uses in-place state mutations (state[:] = ...) in its
streaming convolution layers (StreamingConv1d, StreamingConvTranspose1d).
coremltools cannot convert these in-place operations directly.

The existing mimi_decoder.mlpackage was converted using a custom traceable
wrapper that rewrites all streaming ops as functional (returning new tensors
instead of mutating in place). This requires rewriting the forward pass of:
  - StreamingConv1d.forward()       (conv.py)
  - StreamingConvTranspose1d.forward()  (conv.py)
  - MimiTransformerLayer attention cache updates  (mimi_transformer.py)

The model has 26 streaming state tensors (see traceable_mimi_decoder.py for
the full list) and produces 1920 audio samples per frame at 24kHz.

To regenerate mimi_decoder.mlpackage:
1. Create a functional TraceableMimiDecoder that avoids all in-place ops
2. Trace with sequence_length=256 for attention caches
3. Convert with compute_precision=FLOAT32, target=iOS17

Input:  latent [1, 512, 1]  +  26 state tensors
Output: audio  [1, 1, 1920] +  26 updated state tensors

This script verifies the existing model instead of converting, since direct
conversion is not possible without the functional wrapper.
"""
import sys
import os
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)


def verify():
    """Verify the existing mimi_decoder.mlpackage is iOS 17 compatible."""
    import coremltools as ct

    model_path = os.path.join(_COREML_DIR, "mimi_decoder.mlpackage")

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        print("The mimi_decoder.mlpackage must be converted using a custom")
        print("functional wrapper — see this file's docstring for details.")
        return False

    print(f"Loading {model_path}...")
    mlmodel = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = mlmodel.get_spec()

    # Check spec version (7 = iOS 17 / macOS 14)
    spec_version = spec.specificationVersion
    print(f"\nSpec version: {spec_version}")
    if spec_version <= 7:
        print("  ✓ Compatible with iOS 17+ (spec version ≤ 7)")
    else:
        print(f"  ✗ Requires newer OS (spec version {spec_version})")

    # Check for SDPA ops (would cause BNNS crash on iOS)
    model_desc = str(spec)
    has_sdpa = "scaled_dot_product_attention" in model_desc.lower()
    print(f"\nScaled dot product attention (SDPA): {'FOUND ✗' if has_sdpa else 'None ✓'}")
    if has_sdpa:
        print("  WARNING: SDPA ops will cause BNNS crash on iOS.")

    # Print input/output summary
    print(f"\n=== INPUTS ({len(spec.description.input)}) ===")
    for inp in spec.description.input:
        if inp.type.HasField('multiArrayType'):
            shape = list(inp.type.multiArrayType.shape)
            print(f"  {inp.name}: {shape}")

    print(f"\n=== OUTPUTS ({len(spec.description.output)}) ===")
    for out in spec.description.output:
        if out.type.HasField('multiArrayType'):
            shape = list(out.type.multiArrayType.shape)
            print(f"  {out.name}: {shape}")

    # Quick inference test
    print("\nRunning inference test...")
    test_inputs = {}
    for inp in spec.description.input:
        if inp.type.HasField('multiArrayType'):
            shape = list(inp.type.multiArrayType.shape)
            test_inputs[inp.name] = np.zeros(shape, dtype=np.float32)

    try:
        out = mlmodel.predict(test_inputs)
        print(f"  ✓ Inference succeeded — {len(out)} outputs")

        # Check audio output shape
        for key, val in out.items():
            if hasattr(val, 'shape') and len(val.shape) == 3 and val.shape[-1] == 1920:
                print(f"  ✓ Audio output '{key}': {val.shape} (1920 samples = 80ms at 24kHz)")
                break
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return False

    ios17_ok = spec_version <= 7 and not has_sdpa
    print(f"\n{'✓ Model is iOS 17 compatible' if ios17_ok else '✗ Model needs reconversion for iOS 17'}")
    return ios17_ok


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
