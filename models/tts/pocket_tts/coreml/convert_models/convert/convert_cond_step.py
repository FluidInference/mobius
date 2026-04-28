"""Convert conditioning step model to CoreML."""
import argparse
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
sys.path.insert(0, _SCRIPT_DIR)  # for: from _language_arg import ...
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))  # for: from traceable_* import ...

from _language_arg import (
    add_compute_args,
    add_language_arg,
    build_output_dir,
    resolve_compute_precision,
    resolve_compute_units,
)
from traceable_cond_step import TraceableCondStep


def convert(language: str, compute_precision: str = "fp16", compute_units: str = "ALL"):
    print(f"Loading model (language={language})...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    cond_step = TraceableCondStep.from_flowlm(model.flow_lm, max_seq_len=512)
    cond_step.eval()
    num_layers = cond_step.num_layers
    print(f"num_layers={num_layers}")

    # Example inputs
    conditioning = torch.randn(1, 1, 1024)
    cache = torch.full((2, 1, 512, 16, 64), float('nan'))
    pos = torch.zeros(1)

    example_inputs_list = [conditioning]
    for _ in range(num_layers):
        example_inputs_list.append(cache.clone())
        example_inputs_list.append(pos.clone())
    example_inputs = tuple(example_inputs_list)

    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(cond_step, example_inputs)

    # NOTE: Declaring `dtype=np.float32` on every TensorType (inputs + outputs)
    # forces the IO contract to fp32 even when `compute_precision=fp16`.
    # Without this, CoreML promotes IO to fp16, and the macOS MLE5 binder
    # rejects fp16 MLMultiArrays from Swift ("Invalid heap allocated handle").
    # Anonymous (unnamed) `outputs=` makes coremltools insert fp16→fp32 cast
    # ops at the boundary so internal compute stays fp16 (perf) while IO is
    # Swift-friendly fp32.
    print("Converting to CoreML (precision={}, IO=fp32)...".format(compute_precision))
    inputs = [ct.TensorType(name="conditioning", shape=(1, 1, 1024), dtype=np.float32)]
    for i in range(num_layers):
        inputs.append(ct.TensorType(
            name=f"cache{i}", shape=(2, 1, 512, 16, 64), dtype=np.float32))
        inputs.append(ct.TensorType(
            name=f"position{i}", shape=(1,), dtype=np.float32))

    n_outputs = 2 * num_layers  # interleaved (cache_out, position_out) per layer
    outputs = [ct.TensorType(dtype=np.float32) for _ in range(n_outputs)]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cond_step.mlpackage")
    mlmodel.save(output_path)
    print(f"Saved to {output_path} (precision={compute_precision})")

    # Print outputs
    spec = mlmodel.get_spec()
    print("\n=== OUTPUTS ===")
    for out in spec.description.output:
        if out.type.HasField('multiArrayType'):
            print(f"  {out.name}: {list(out.type.multiArrayType.shape)}")

    # Quick test
    print(f"\nTesting (compute_units={compute_units})...")
    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))
    test_inputs = {
        'conditioning': np.random.randn(1, 1, 1024).astype(np.float32),
    }
    for i in range(num_layers):
        test_inputs[f'cache{i}'] = np.zeros((2, 1, 512, 16, 64), dtype=np.float32)
        test_inputs[f'position{i}'] = np.array([0.0], dtype=np.float32)
    out = coreml_model.predict(test_inputs)
    print(f"Output keys: {len(out)}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    args = parser.parse_args()
    convert(args.language, args.compute_precision, args.compute_units)
