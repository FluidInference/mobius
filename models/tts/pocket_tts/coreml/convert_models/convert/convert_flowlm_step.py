"""Convert FlowLM step model to CoreML."""
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
from traceable_flowlm_step import TraceableFlowLMStep


def convert_flowlm_step(language: str, compute_precision: str = "fp16", compute_units: str = "ALL"):
    print(f"Loading model (language={language})...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    print("Creating traceable step model...")
    max_seq_len = 512
    step_model = TraceableFlowLMStep.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    step_model.eval()
    num_layers = step_model.num_layers
    print(f"num_layers={num_layers}")

    print("Creating example inputs...")
    B = 1
    T = 1
    H = 16
    D = 64

    sequence = torch.randn(B, T, 32)
    bos_emb = model.flow_lm.bos_emb.data

    # Create example caches and positions
    caches = []
    positions = []
    for i in range(num_layers):
        cache = torch.zeros(2, B, max_seq_len, H, D)
        # Fill some positions with data (simulating voice/text conditioning)
        cache[:, :, :136, :, :] = torch.randn(2, B, 136, H, D)
        caches.append(cache)
        positions.append(torch.tensor([136.0]))

    print("Tracing model...")
    trace_inputs = [sequence, bos_emb]
    for cache, pos in zip(caches, positions):
        trace_inputs.append(cache)
        trace_inputs.append(pos)
    with torch.no_grad():
        traced = torch.jit.trace(step_model, tuple(trace_inputs))

    # NOTE: Force fp32 IO contract via `dtype=np.float32` on every TensorType
    # plus anonymous fp32 outputs. With `compute_precision=fp16`, internal
    # ops still run in fp16 (the perf win we want) but coremltools inserts
    # fp16↔fp32 cast ops at the IO boundary so Swift can drive the model
    # with `MLMultiArrayDataType.float32` buffers. Without this, the macOS
    # MLE5 binder rejects fp16 MLMultiArrays ("Invalid heap allocated handle").
    print("Converting to CoreML (precision={}, IO=fp32)...".format(compute_precision))
    inputs = [
        ct.TensorType(name="sequence", shape=(1, 1, 32), dtype=np.float32),
        ct.TensorType(name="bos_emb", shape=(32,), dtype=np.float32),
    ]
    for i in range(num_layers):
        inputs.append(ct.TensorType(
            name=f"cache{i}", shape=(2, 1, max_seq_len, H, D), dtype=np.float32))
        inputs.append(ct.TensorType(
            name=f"position{i}", shape=(1,), dtype=np.float32))

    # Outputs: (x, is_eos, cache0_out, position0_out, ..., cacheN_out, positionN_out)
    n_outputs = 2 + 2 * num_layers
    ct_outputs = [ct.TensorType(dtype=np.float32) for _ in range(n_outputs)]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "flowlm_step.mlpackage")
    print(f"Saving to {output_path} (precision={compute_precision})...")
    mlmodel.save(output_path)

    # Test
    print(f"\nTesting CoreML model (compute_units={compute_units})...")
    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))

    # Create test inputs
    test_seq = np.random.randn(1, 1, 32).astype(np.float32)
    test_bos = bos_emb.numpy().astype(np.float32)

    test_caches = {}
    test_positions = {}
    for i in range(num_layers):
        cache = np.zeros((2, 1, max_seq_len, H, D), dtype=np.float32)
        cache[:, :, :136, :, :] = np.random.randn(2, 1, 136, H, D).astype(np.float32)
        test_caches[f'cache{i}'] = cache
        test_positions[f'position{i}'] = np.array([136.0], dtype=np.float32)

    outputs = coreml_model.predict({
        'sequence': test_seq,
        'bos_emb': test_bos,
        **test_caches,
        **test_positions,
    })

    print(f"Output keys: {list(outputs.keys())}")
    for k, v in outputs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    args = parser.parse_args()
    convert_flowlm_step(args.language, args.compute_precision, args.compute_units)
