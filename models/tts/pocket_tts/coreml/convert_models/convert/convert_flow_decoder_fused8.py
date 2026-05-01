"""Convert the fused-8-step flow decoder to CoreML, then verify parity.

Compares against the per-step `TraceableFlowDecoder` reference loop on
random conditioning inputs. Reports max abs diff between:
    - PyTorch fused single-call vs PyTorch reference 8-call loop
    - CoreML fused single-call vs PyTorch reference 8-call loop

Usage:
    uv run --no-project --python 3.10 \
        --with "pocket-tts>=1.0.3" \
        --with "coremltools>=8.0" \
        --with "safetensors>=0.4.0" \
        --with "sentencepiece>=0.2.1" \
        --with "scipy>=1.5.0" \
        --with "numpy>=2" \
        --with "torch>=2.5.0" \
        --with "huggingface_hub>=0.10" \
        --with "einops>=0.4.0" \
        python convert_flow_decoder_fused8.py
"""
import os
import sys

import coremltools as ct
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))

from traceable_flow_decoder import TraceableFlowDecoder  # noqa: E402
from traceable_flow_decoder_fused8 import FusedFlowDecoder8  # noqa: E402


def reference_loop(per_step_decoder, transformer_out, latent_init, num_steps=8):
    """Reference: 8 separate per-step calls, exactly mirrors current Swift loop."""
    z = latent_init.clone()
    dt = 1.0 / num_steps
    with torch.no_grad():
        for i in range(num_steps):
            s = torch.tensor([[i * dt]], dtype=torch.float32)
            t = torch.tensor([[(i + 1) * dt]], dtype=torch.float32)
            v = per_step_decoder(transformer_out, z, s, t)
            z = z + v * dt
    return z


def main():
    print("Loading PocketTTS model (default English)...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    print("Building per-step reference and fused-8 decoders...")
    per_step = TraceableFlowDecoder.from_flowlm(model.flow_lm).eval()
    fused = FusedFlowDecoder8.from_flowlm(model.flow_lm).eval()

    print("Constructing example inputs (deterministic seed=0)...")
    torch.manual_seed(0)
    transformer_out = torch.randn(1, 1024)
    latent_init = torch.randn(1, 32)

    # ---- PyTorch parity check ----
    print("\n[PyTorch] Reference 8-step loop vs fused single call...")
    with torch.no_grad():
        ref = reference_loop(per_step, transformer_out, latent_init)
        fused_out = fused(transformer_out, latent_init)
    pt_diff = (ref - fused_out).abs().max().item()
    print(f"  max abs diff (PyTorch ref vs fused): {pt_diff:.3e}")

    # ---- Trace + convert ----
    print("\nTracing fused-8 decoder...")
    with torch.no_grad():
        traced = torch.jit.trace(fused, (transformer_out, latent_init))

    print("Converting to CoreML (FP32, iOS17)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="transformer_out", shape=(1, 1024)),
            ct.TensorType(name="latent_init", shape=(1, 32)),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "flow_decoder_fused8.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    # ---- CoreML parity check ----
    print("\nLoading CoreML model on CPU+GPU...")
    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

    print("Predicting through fused CoreML graph...")
    outputs = coreml_model.predict({
        "transformer_out": transformer_out.numpy().astype(np.float32),
        "latent_init": latent_init.numpy().astype(np.float32),
    })
    coreml_out = list(outputs.values())[0]
    print(f"  CoreML output shape: {coreml_out.shape}")
    cm_diff = float(np.abs(coreml_out - ref.numpy()).max())
    print(f"  max abs diff (PyTorch ref vs CoreML fused): {cm_diff:.3e}")

    # ---- Summary ----
    print("\n=== PARITY SUMMARY ===")
    print(f"  PyTorch ref vs PyTorch fused: {pt_diff:.3e}")
    print(f"  PyTorch ref vs CoreML  fused: {cm_diff:.3e}")
    print("(Healthy thresholds: PyTorch parity ~0, CoreML parity < 1e-3 for FP32)")
    print("\nDone!")


if __name__ == "__main__":
    main()
