#!/usr/bin/env python3
"""Test the full CoreML pipeline end-to-end."""
import numpy as np
import torch
import coremltools as ct
import nemo.collections.asr as nemo_asr

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

print("\n=== Test 1: NeMo Full Pipeline (Reference) ===")
with torch.inference_mode():
    mel, mel_length = asr_model.preprocessor(
        input_signal=dummy_audio, length=dummy_length.long()
    )
    encoded, encoded_length = asr_model.encoder(
        audio_signal=mel, length=mel_length.long()
    )
    # Get raw logits (not log-softmax)
    conv_output = asr_model.ctc_decoder.decoder_layers(encoded)
    raw_logits_nemo = conv_output.transpose(1, 2)

print(f"NeMo raw logits shape: {raw_logits_nemo.shape}")
print(f"NeMo raw logits range: [{raw_logits_nemo.min():.2f}, {raw_logits_nemo.max():.2f}]")

print("\n=== Test 2: CoreML Full Pipeline ===")
mlmodel = ct.models.MLModel('build/FullPipeline.mlpackage')
coreml_output = mlmodel.predict({
    'audio_signal': dummy_audio.numpy(),
    'audio_length': dummy_length.numpy()
})

logits_coreml = coreml_output['ctc_logits']
length_coreml = coreml_output['encoder_length']

print(f"CoreML logits shape: {logits_coreml.shape}")
print(f"CoreML logits range: [{logits_coreml.min():.2f}, {logits_coreml.max():.2f}]")
print(f"CoreML encoder length: {length_coreml}")

# Compare
diff = np.abs(raw_logits_nemo.numpy() - logits_coreml).max()
print(f"\n**Max difference: {diff:.6e}**")

if diff < 0.1:
    print("✅ FULL PIPELINE WORKS! CoreML matches NeMo.")
    print("\nThe Japanese Parakeet CTC model is now successfully converted to CoreML.")
    print("Apply log_softmax to the output logits for CTC decoding.")
else:
    print(f"❌ Full pipeline has issues (max diff: {diff:.6e})")

print("\n=== Test 3: Verify log_softmax Application ===")
with torch.inference_mode():
    # What the original decoder would output
    original_logprobs = asr_model.ctc_decoder(encoder_output=encoded)

    # Apply log_softmax to our raw logits
    log_probs_from_raw = torch.nn.functional.log_softmax(
        raw_logits_nemo, dim=-1
    )

    diff_logsoftmax = torch.abs(original_logprobs - log_probs_from_raw).max()
    print(f"log_softmax(raw_logits) vs original decoder: {diff_logsoftmax:.6e}")

    if diff_logsoftmax < 1e-5:
        print("✅ log_softmax produces identical results to original decoder")
