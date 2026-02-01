"""Convert Mimi streaming decoder to CoreML.

Traces the TraceableMimiDecoder (which bakes in denormalize + quantize)
and converts to CoreML .mlpackage. Strips zero-length state tensors
from the spec to avoid Espresso crash on iOS.

Input:  latent [1, 32]       +  23 state tensors (26 minus 3 zero-length)
Output: audio  [1, 1, 1920]  +  23 updated state tensors
"""
import torch
import numpy as np
import coremltools as ct
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))  # for: from traceable_* import ...

from traceable_mimi_decoder import TraceableMimiDecoder, MIMI_STATE_SPEC


def strip_zero_length_io(mlpackage_path):
    """Remove zero-length tensor inputs/outputs from a saved CoreML mlpackage.

    Three Mimi state tensors have a zero-length dimension (kernel_size=1
    streaming conv layers with 0 padding). CoreML Espresso crashes on
    zero-element blobs, so we strip them from the spec.

    Must operate on a saved .mlpackage (not in-memory) because mlProgram
    models require the weights directory when loading from spec.
    """
    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = mlmodel.get_spec()

    # Find zero-length input/output names
    zero_inputs = set()
    for inp in spec.description.input:
        if inp.type.HasField('multiArrayType'):
            shape = list(inp.type.multiArrayType.shape)
            if 0 in shape:
                zero_inputs.add(inp.name)

    zero_outputs = set()
    for out in spec.description.output:
        if out.type.HasField('multiArrayType'):
            shape = list(out.type.multiArrayType.shape)
            if 0 in shape:
                zero_outputs.add(out.name)

    if not zero_inputs and not zero_outputs:
        print("No zero-length tensors to strip.")
        return

    print(f"Stripping {len(zero_inputs)} zero-length inputs: {zero_inputs}")
    print(f"Stripping {len(zero_outputs)} zero-length outputs: {zero_outputs}")

    # Remove from spec
    inputs_to_keep = [inp for inp in spec.description.input
                      if inp.name not in zero_inputs]
    outputs_to_keep = [out for out in spec.description.output
                       if out.name not in zero_outputs]

    del spec.description.input[:]
    spec.description.input.extend(inputs_to_keep)

    del spec.description.output[:]
    spec.description.output.extend(outputs_to_keep)

    # Save modified spec back (with weights dir from the mlpackage)
    weights_dir = os.path.join(mlpackage_path, "Data",
                               "com.apple.CoreML", "weights")
    updated = ct.models.MLModel(spec, weights_dir=weights_dir,
                                compute_units=ct.ComputeUnit.CPU_AND_GPU)
    updated.save(mlpackage_path)


def convert():
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    print("Creating traceable Mimi decoder (with denormalize + quantize baked in)...")
    traceable = TraceableMimiDecoder.from_tts_model(model)
    traceable.eval()

    # Build example inputs from MIMI_STATE_SPEC
    print("Creating example inputs...")
    latent = torch.randn(1, 32)
    state_tensors = []
    ct_inputs = [ct.TensorType(name="latent", shape=(1, 32))]

    for name, shape in MIMI_STATE_SPEC:
        t = torch.zeros(*shape)
        state_tensors.append(t)
        ct_inputs.append(ct.TensorType(name=name, shape=tuple(shape)))

    example_inputs = (latent,) + tuple(state_tensors)

    print(f"Tracing with {len(state_tensors)} state tensors...")
    with torch.no_grad():
        traced = torch.jit.trace(traceable, example_inputs)

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = os.path.join(_COREML_DIR, "mimi_decoder.mlpackage")
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    # NOTE: Zero-length I/O stripping is skipped for mlProgram format.
    # The 3 zero-length state tensors (res{0,1,2}_conv1_prev) are kept in the
    # model spec. The Swift side provides them as empty MLMultiArrays.
    # Stripping from the spec description causes "Model and main function must
    # have same number of inputs and states" because the MIL function still
    # references them.

    # Print I/O summary
    spec = mlmodel.get_spec()
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

    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    out = coreml_model.predict(test_inputs)
    print(f"  Inference succeeded — {len(out)} outputs")

    for key, val in out.items():
        if hasattr(val, 'shape') and len(val.shape) == 3 and val.shape[-1] == 1920:
            print(f"  Audio output '{key}': {val.shape} (1920 samples = 80ms at 24kHz)")
            break

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    convert()
