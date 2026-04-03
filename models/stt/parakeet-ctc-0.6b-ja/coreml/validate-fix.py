#!/usr/bin/env python3
"""Quick validation that the fixed CTC decoder works correctly."""
import numpy as np
import torch
import coremltools as ct
import nemo.collections.asr as nemo_asr
from pathlib import Path

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

# Run through encoder to get encoder output
print("Generating encoder output...")
with torch.inference_mode():
    mel, mel_length = asr_model.preprocessor(
        input_signal=dummy_audio, length=dummy_length.long()
    )
    encoded, encoded_length = asr_model.encoder(
        audio_signal=mel, length=mel_length.long()
    )

print(f"Encoder output shape: {encoded.shape}")
print(f"Encoder output range: [{encoded.min():.2f}, {encoded.max():.2f}]")

# Test 1: Get RAW logits from decoder_layers (what our wrapper does now)
print("\n=== Test 1: Raw Logits (PyTorch) ===")
with torch.inference_mode():
    conv_output = asr_model.ctc_decoder.decoder_layers(encoded)  # [B, V, T]
    raw_logits_pytorch = conv_output.transpose(1, 2)  # [B, T, V]

print(f"Raw logits shape: {raw_logits_pytorch.shape}")
print(f"Raw logits range: [{raw_logits_pytorch.min():.2f}, {raw_logits_pytorch.max():.2f}]")
print(f"Raw logits sample (first timestep, first 10 tokens):")
print(raw_logits_pytorch[0, 0, :10].numpy())

# Test 2: Load CoreML model and compare
print("\n=== Test 2: Raw Logits (CoreML) ===")
mlmodel = ct.models.MLModel('build/CtcDecoder.mlpackage')
coreml_output = mlmodel.predict({
    'encoder_output': encoded.numpy()
})['ctc_logits']

print(f"CoreML output shape: {coreml_output.shape}")
print(f"CoreML output range: [{coreml_output.min():.2f}, {coreml_output.max():.2f}]")
print(f"CoreML sample (first timestep, first 10 tokens):")
print(coreml_output[0, 0, :10])

# Compare
diff = np.abs(raw_logits_pytorch.numpy() - coreml_output).max()
print(f"\n**Max difference: {diff:.6e}**")

if diff > 1.0:
    print("\n❌ STILL BROKEN - CoreML conversion issue persists")
    print(f"Expected range: [{raw_logits_pytorch.min():.2f}, {raw_logits_pytorch.max():.2f}]")
    print(f"Got range: [{coreml_output.min():.2f}, {coreml_output.max():.2f}]")
else:
    print("\n✅ FIXED! CoreML conversion now works correctly")
    print("The raw logits match between PyTorch and CoreML.")
    print("\nNote: These are RAW logits, not log-softmax.")
    print("Apply log_softmax in post-processing for CTC decoding.")

# Test 3: Verify log_softmax can be applied in post-processing
print("\n=== Test 3: Verify log_softmax application ===")
with torch.inference_mode():
    # What the original decoder would output
    original_output = asr_model.ctc_decoder(encoder_output=encoded)
    print(f"Original CTC decoder output (with log_softmax): [{original_output.min():.2f}, {original_output.max():.2f}]")

    # Apply log_softmax to our raw logits
    log_probs_from_raw = torch.nn.functional.log_softmax(
        raw_logits_pytorch, dim=-1
    )
    print(f"Log-softmax applied to raw logits: [{log_probs_from_raw.min():.2f}, {log_probs_from_raw.max():.2f}]")

    # Compare
    logsoftmax_diff = torch.abs(original_output - log_probs_from_raw).max()
    print(f"Difference: {logsoftmax_diff:.6e}")

    if logsoftmax_diff < 1e-5:
        print("✅ log_softmax(raw_logits) matches original decoder output")
    else:
        print("❌ Mismatch - something is wrong")
