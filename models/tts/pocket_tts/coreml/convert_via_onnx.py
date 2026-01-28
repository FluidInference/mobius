#!/usr/bin/env python3
"""Convert PocketTTS components via ONNX (bypasses beartype JIT issues)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path

# Disable beartype at import time
import beartype
beartype.BeartypeConf.is_disabled = True

print("Loading PocketTTS model...")
from pocket_tts import TTSModel

# Load the model
model = TTSModel.load_model()
model.eval()

print(f"Model loaded. Sample rate: {model.sample_rate}")

# ============================================================================
# Simple wrappers (beartype will be disabled)
# ============================================================================
class TextEncoderONNX(nn.Module):
    def __init__(self, conditioner):
        super().__init__()
        # LUTConditioner is just an embedding lookup: 4001 tokens -> 1024 dim
        self.embed = conditioner.embed

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Direct embedding lookup (no projection needed)
        return self.embed(tokens)

class FlowDecoderONNX(nn.Module):
    def __init__(self, flow_net, ldim):
        super().__init__()
        self.flow_net = flow_net
        self.ldim = ldim

    def forward(self, transformer_out: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s = torch.zeros_like(noise[..., :1])
        t = torch.ones_like(noise[..., :1])
        flow_dir = self.flow_net(transformer_out, s, t, noise)
        return noise + flow_dir

class EOSDetectorONNX(nn.Module):
    def __init__(self, out_eos, threshold=-4.0):
        super().__init__()
        self.out_eos = out_eos
        self.threshold = threshold

    def forward(self, transformer_out: torch.Tensor) -> torch.Tensor:
        logit = self.out_eos(transformer_out)
        return (logit > self.threshold).float()

# ============================================================================
# Export via ONNX
# ============================================================================
def convert_via_onnx(wrapper, dummy_inputs, name, input_names, output_names, dynamic_axes=None):
    print(f"\n--- Converting {name} via ONNX ---")
    wrapper.eval()

    onnx_path = f"{name}.onnx"

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  Exported to {onnx_path}")

    # Convert ONNX to CoreML
    mlmodel = ct.convert(
        onnx_path,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlpackage_path = f"pocket_tts_{name}.mlpackage"
    mlmodel.save(mlpackage_path)
    print(f"  Saved: {mlpackage_path}")

    # Cleanup ONNX file
    os.remove(onnx_path)

    return mlmodel

if __name__ == "__main__":
    os.makedirs("coreml", exist_ok=True)
    os.chdir("coreml")

    print("=" * 60)
    print("PocketTTS CoreML Conversion via ONNX")
    print("=" * 60)

    results = []

    # 1. Text Encoder
    try:
        text_enc = TextEncoderONNX(model.flow_lm.conditioner)
        dummy_tokens = torch.zeros((1, 32), dtype=torch.int64)
        convert_via_onnx(
            text_enc,
            dummy_tokens,
            "text_encoder",
            input_names=["tokens"],
            output_names=["embeddings"],
            dynamic_axes={"tokens": {1: "seq_len"}, "embeddings": {1: "seq_len"}}
        )
        results.append(("Text Encoder", "✓"))
    except Exception as e:
        print(f"  Error: {e}")
        results.append(("Text Encoder", f"✗ {str(e)[:40]}"))

    # 2. Flow Decoder
    try:
        dim = model.flow_lm.dim
        ldim = model.flow_lm.ldim
        flow_dec = FlowDecoderONNX(model.flow_lm.flow_net, ldim)
        dummy_trans = torch.randn(1, dim)
        dummy_noise = torch.randn(1, ldim)
        convert_via_onnx(
            flow_dec,
            (dummy_trans, dummy_noise),
            "flow_decoder",
            input_names=["transformer_out", "noise"],
            output_names=["latent"],
        )
        results.append(("Flow Decoder", "✓"))
    except Exception as e:
        print(f"  Error: {e}")
        results.append(("Flow Decoder", f"✗ {str(e)[:40]}"))

    # 3. EOS Detector
    try:
        eos = EOSDetectorONNX(model.flow_lm.out_eos)
        dummy = torch.randn(1, model.flow_lm.dim)
        convert_via_onnx(
            eos,
            dummy,
            "eos_detector",
            input_names=["transformer_out"],
            output_names=["is_eos"],
        )
        results.append(("EOS Detector", "✓"))
    except Exception as e:
        print(f"  Error: {e}")
        results.append(("EOS Detector", f"✗ {str(e)[:40]}"))

    print("\n" + "=" * 60)
    print("Conversion Results:")
    for name, status in results:
        print(f"  {name}: {status}")
    print("=" * 60)

    # Model sizes
    print("\nModel Sizes:")
    import glob
    for pkg in sorted(glob.glob("*.mlpackage")):
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(pkg) for f in fn)
        print(f"  {pkg}: {size / 1024 / 1024:.1f} MB")
