"""Convert traceable flow decoder to CoreML."""
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
from traceable_flow_decoder import TraceableFlowDecoder


def convert_flow_decoder(language: str, compute_precision: str = "fp16", compute_units: str = "ALL"):
    print(f"Loading model (language={language})...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    print("Creating traceable flow decoder...")
    flow_decoder = TraceableFlowDecoder.from_flowlm(model.flow_lm)
    flow_decoder.eval()

    print("Creating example inputs...")
    transformer_out = torch.randn(1, 1024)
    latent = torch.randn(1, 32)
    s = torch.tensor([[0.0]])
    t = torch.tensor([[0.125]])

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(flow_decoder, (transformer_out, latent, s, t))

    # NOTE: Force fp32 IO contract via `dtype=np.float32` on every TensorType
    # plus an anonymous fp32 output. With `compute_precision=fp16`, internal
    # ops still run in fp16 but coremltools inserts fp16↔fp32 cast ops at the
    # IO boundary so Swift can drive the model with `MLMultiArrayDataType.float32`
    # buffers. Without this, the macOS MLE5 binder rejects fp16 MLMultiArrays
    # ("Invalid heap allocated handle").
    print("Converting to CoreML (precision={}, IO=fp32)...".format(compute_precision))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="transformer_out", shape=(1, 1024), dtype=np.float32),
            ct.TensorType(name="latent", shape=(1, 32), dtype=np.float32),
            ct.TensorType(name="s", shape=(1, 1), dtype=np.float32),
            ct.TensorType(name="t", shape=(1, 1), dtype=np.float32),
        ],
        outputs=[ct.TensorType(dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "flow_decoder.mlpackage")
    print(f"Saving to {output_path} (precision={compute_precision})...")
    mlmodel.save(output_path)

    # Test
    print(f"\nTesting CoreML model (compute_units={compute_units})...")
    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))

    test_transformer = np.random.randn(1, 1024).astype(np.float32)
    test_latent = np.random.randn(1, 32).astype(np.float32)
    test_s = np.array([[0.0]], dtype=np.float32)
    test_t = np.array([[0.125]], dtype=np.float32)

    outputs = coreml_model.predict({
        'transformer_out': test_transformer,
        'latent': test_latent,
        's': test_s,
        't': test_t,
    })

    print(f"Output keys: {list(outputs.keys())}")
    velocity = list(outputs.values())[0]
    print(f"Velocity shape: {velocity.shape}")
    print(f"Velocity range: [{velocity.min():.4f}, {velocity.max():.4f}]")

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    args = parser.parse_args()
    convert_flow_decoder(args.language, args.compute_precision, args.compute_units)
