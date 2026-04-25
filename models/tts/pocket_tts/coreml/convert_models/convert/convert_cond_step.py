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

from _language_arg import add_language_arg, build_output_dir
from traceable_cond_step import TraceableCondStep


def convert(language: str):
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

    print("Converting to CoreML...")
    inputs = [ct.TensorType(name="conditioning", shape=(1, 1, 1024))]
    for i in range(num_layers):
        inputs.append(ct.TensorType(name=f"cache{i}", shape=(2, 1, 512, 16, 64)))
        inputs.append(ct.TensorType(name=f"position{i}", shape=(1,)))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cond_step.mlpackage")
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")

    # Print outputs
    spec = mlmodel.get_spec()
    print("\n=== OUTPUTS ===")
    for out in spec.description.output:
        if out.type.HasField('multiArrayType'):
            print(f"  {out.name}: {list(out.type.multiArrayType.shape)}")

    # Quick test
    print("\nTesting...")
    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
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
    args = parser.parse_args()
    convert(args.language)
