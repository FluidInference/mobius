"""Convert conditioning step model to CoreML."""
import torch
import numpy as np
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_cond_step import TraceableCondStep


def convert():
    print("Loading model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    cond_step = TraceableCondStep.from_flowlm(model.flow_lm, max_seq_len=200)
    cond_step.eval()

    # Example inputs
    conditioning = torch.randn(1, 1, 1024)
    cache = torch.full((2, 1, 200, 16, 64), float('nan'))
    pos = torch.zeros(1)

    example_inputs = (
        conditioning,
        cache, pos, cache.clone(), pos.clone(),
        cache.clone(), pos.clone(), cache.clone(), pos.clone(),
        cache.clone(), pos.clone(), cache.clone(), pos.clone(),
    )

    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(cond_step, example_inputs)

    print("Converting to CoreML...")
    inputs = [ct.TensorType(name="conditioning", shape=(1, 1, 1024))]
    for i in range(6):
        inputs.append(ct.TensorType(name=f"cache{i}", shape=(2, 1, 200, 16, 64)))
        inputs.append(ct.TensorType(name=f"position{i}", shape=(1,)))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "cond_step.mlpackage"
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
    for i in range(6):
        test_inputs[f'cache{i}'] = np.zeros((2, 1, 200, 16, 64), dtype=np.float32)
        test_inputs[f'position{i}'] = np.array([0.0], dtype=np.float32)
    out = coreml_model.predict(test_inputs)
    print(f"Output keys: {len(out)}")
    print("Done!")


if __name__ == "__main__":
    convert()
