#!/usr/bin/env python3
"""Test CTC decoder linear layer in isolation to identify numerical bug."""
from __future__ import annotations

import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import nemo.collections.asr as nemo_asr

# Load model
print("Loading NeMo model...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt_ctc-0.6b-ja", map_location="cpu"
)
asr_model.eval()

# Generate test data
max_samples = 240000
torch.manual_seed(42)
dummy_audio = torch.randn(1, max_samples, dtype=torch.float32)
dummy_length = torch.tensor([max_samples], dtype=torch.int32)

# Run through preprocessor and encoder to get real encoder output
print("\nGenerating encoder output...")
with torch.inference_mode():
    mel, mel_length = asr_model.preprocessor(
        input_signal=dummy_audio, length=dummy_length.long()
    )
    encoded, encoded_length = asr_model.encoder(
        audio_signal=mel, length=mel_length.long()
    )

print(f"Encoder output shape: {encoded.shape}")
print(f"Encoder output range: [{encoded.min():.2f}, {encoded.max():.2f}]")

# Get the CTC decoder module
ctc_decoder = asr_model.ctc_decoder
print(f"\nCTC Decoder type: {type(ctc_decoder)}")
print(f"CTC Decoder module: {ctc_decoder}")

# Inspect the CTC decoder structure
print("\nCTC Decoder attributes:")
for name, module in ctc_decoder.named_modules():
    print(f"  {name}: {type(module).__name__}")

# Find the projection layer (Conv1d or Linear)
# In NeMo CTC decoders, it's typically ctc_decoder.decoder_layers
if hasattr(ctc_decoder, 'decoder_layers'):
    projection_layer = ctc_decoder.decoder_layers
    print(f"\nProjection layer found: {projection_layer}")

    # Check if it's Conv1d or Linear
    first_layer = projection_layer[0] if isinstance(projection_layer, torch.nn.Sequential) else projection_layer
    print(f"Layer type: {type(first_layer).__name__}")

    if hasattr(first_layer, 'weight'):
        print(f"Weight shape: {first_layer.weight.shape}")
        print(f"Bias shape: {first_layer.bias.shape if first_layer.bias is not None else 'None'}")
else:
    print("\nSearching for projection layer...")
    for name, param in ctc_decoder.named_parameters():
        print(f"  {name}: {param.shape}")

# Test 1: Run full CTC decoder (PyTorch)
print("\n=== Test 1: Full CTC Decoder (PyTorch) ===")
with torch.inference_mode():
    ctc_logits = ctc_decoder(encoder_output=encoded)

print(f"CTC logits shape: {ctc_logits.shape}")
print(f"CTC logits range: [{ctc_logits.min():.2f}, {ctc_logits.max():.2f}]")
print(f"CTC logits sample (first timestep, first 10 tokens):")
print(ctc_logits[0, 0, :10].numpy())

# Test 2: Manually apply the projection layer
print("\n=== Test 2: Manual Projection Layer Application ===")
if hasattr(ctc_decoder, 'decoder_layers'):
    projection_layer = ctc_decoder.decoder_layers

    # The CTC decoder expects [B, D, T] and outputs [B, T, V]
    # With Conv1d: [B, D, T] -> Conv1d -> [B, V, T] -> transpose -> [B, T, V]
    with torch.inference_mode():
        # Apply conv1d directly (it expects [B, C, T])
        conv_output = projection_layer(encoded)  # [B, V, T]
        print(f"After Conv1d: {conv_output.shape}")

        # Transpose to [B, T, V]
        manual_logits = conv_output.transpose(1, 2)  # [B, T, V]
        print(f"Manual logits shape: {manual_logits.shape}")
        print(f"Manual logits range: [{manual_logits.min():.2f}, {manual_logits.max():.2f}]")
        print(f"Manual logits sample (first timestep, first 10 tokens):")
        print(manual_logits[0, 0, :10].numpy())

        # Compare with full decoder
        diff = torch.abs(ctc_logits - manual_logits).max()
        print(f"\nDifference between full decoder and manual: {diff:.6e}")

# Test 3: Export ONLY the Conv1d layer to CoreML
print("\n=== Test 3: Export Conv1d Layer to CoreML ===")

class Conv1dOnlyWrapper(torch.nn.Module):
    """Wrapper that ONLY contains the Conv1d projection."""
    def __init__(self, conv_layer):
        super().__init__()
        self.conv = conv_layer

    def forward(self, x):
        # Input: [B, D, T] - encoder features
        # Output: [B, V, T] - logits (not transposed yet)
        return self.conv(x)

if hasattr(ctc_decoder, 'decoder_layers'):
    conv_only = Conv1dOnlyWrapper(ctc_decoder.decoder_layers)

    # Prepare input (encoder output as-is [B, D, T])
    test_input = encoded  # [1, 1024, 188]

    # Trace it
    print(f"Tracing Conv1d layer with input shape: {test_input.shape}")
    with torch.inference_mode():
        conv_traced = torch.jit.trace(conv_only, test_input, strict=False)
    conv_traced.eval()

    # Convert to CoreML
    print("Converting to CoreML...")
    conv_coreml = ct.convert(
        conv_traced,
        inputs=[
            ct.TensorType(
                name="encoder_output",
                shape=test_input.shape,
                dtype=np.float32
            )
        ],
        outputs=[
            ct.TensorType(name="logits_untransposed", dtype=np.float32)
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17,
    )

    # Save it
    output_path = Path("build/Conv1dOnly.mlpackage")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conv_coreml.save(str(output_path))
    print(f"Saved: {output_path}")

    # Test it
    print("\n=== Test 4: Compare Conv1d PyTorch vs CoreML ===")
    with torch.inference_mode():
        pytorch_output = conv_traced(test_input)

    coreml_output = conv_coreml.predict(
        {"encoder_output": test_input.numpy()}
    )["logits_untransposed"]

    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"CoreML output shape: {coreml_output.shape}")
    print(f"PyTorch output range: [{pytorch_output.min():.2f}, {pytorch_output.max():.2f}]")
    print(f"CoreML output range: [{coreml_output.min():.2f}, {coreml_output.max():.2f}]")
    print(f"\nPyTorch sample (first vocab token, first 10 timesteps):")
    print(pytorch_output[0, 0, :10].numpy())
    print(f"CoreML sample (first vocab token, first 10 timesteps):")
    print(coreml_output[0, 0, :10])

    diff = np.abs(pytorch_output.numpy() - coreml_output).max()
    print(f"\n**Max difference: {diff:.6e}**")

    if diff > 1.0:
        print("\n❌ CONV1D LAYER ALONE IS BROKEN!")
        print("This indicates the issue is in the Conv1d weights/conversion itself.")
    else:
        print("\n✅ CONV1D LAYER ALONE WORKS!")
        print("This suggests the issue is in the transpose operation or elsewhere.")

# Test 5: Export Conv1d with transpose
print("\n=== Test 5: Export Conv1d WITH Transpose ===")

class Conv1dTransposeWrapper(torch.nn.Module):
    """Wrapper with Conv1d + transpose (mimics full CTC decoder)."""
    def __init__(self, conv_layer):
        super().__init__()
        self.conv = conv_layer

    def forward(self, encoder_output):
        # Input: [B, D, T] - encoder output
        # Output: [B, T, V] - logits
        x = self.conv(encoder_output)  # [B, V, T]
        logits = x.transpose(1, 2)  # [B, T, V]
        return logits

if hasattr(ctc_decoder, 'decoder_layers'):
    conv_transpose = Conv1dTransposeWrapper(ctc_decoder.decoder_layers)

    # Trace it with encoder output [B, D, T]
    print(f"Tracing Conv1d+transpose with input shape: {encoded.shape}")
    with torch.inference_mode():
        ct_traced = torch.jit.trace(conv_transpose, encoded, strict=False)
    ct_traced.eval()

    # Convert to CoreML
    print("Converting to CoreML...")
    ct_coreml = ct.convert(
        ct_traced,
        inputs=[
            ct.TensorType(
                name="encoder_output",
                shape=encoded.shape,
                dtype=np.float32
            )
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32)
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17,
    )

    # Save it
    output_path = Path("build/Conv1dTranspose.mlpackage")
    ct_coreml.save(str(output_path))
    print(f"Saved: {output_path}")

    # Test it
    print("\n=== Test 6: Compare Conv1d+Transpose PyTorch vs CoreML ===")
    with torch.inference_mode():
        pytorch_output = ct_traced(encoded)

    coreml_output = ct_coreml.predict(
        {"encoder_output": encoded.numpy()}
    )["logits"]

    print(f"PyTorch output range: [{pytorch_output.min():.2f}, {pytorch_output.max():.2f}]")
    print(f"CoreML output range: [{coreml_output.min():.2f}, {coreml_output.max():.2f}]")
    print(f"\nPyTorch sample (first timestep, first 10 tokens):")
    print(pytorch_output[0, 0, :10].numpy())
    print(f"CoreML sample (first timestep, first 10 tokens):")
    print(coreml_output[0, 0, :10])

    diff = np.abs(pytorch_output.numpy() - coreml_output).max()
    print(f"\n**Max difference: {diff:.6e}**")

    if diff > 1.0:
        print("\n❌ CONV1D+TRANSPOSE IS BROKEN!")
        print("This narrows down the issue to either:")
        print("  1. The Conv1d conversion")
        print("  2. The transpose operation conversion")
        print("  3. The interaction between them")
    else:
        print("\n✅ CONV1D+TRANSPOSE WORKS!")
        print("This is very strange - suggests issue is elsewhere in CTC decoder.")

print("\n=== Summary ===")
print("This test isolates the linear projection layer to identify where the numerical bug occurs.")
