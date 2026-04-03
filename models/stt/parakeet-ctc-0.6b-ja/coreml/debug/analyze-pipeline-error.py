#!/usr/bin/env python3
"""Analyze where the numerical error accumulates in the pipeline."""
import numpy as np
import torch
import coremltools as ct
import nemo.collections.asr as nemo_asr

print("Loading models...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt_ctc-0.6b-ja", map_location="cpu"
)
asr_model.eval()

# Generate test data
max_samples = 240000
torch.manual_seed(42)
dummy_audio = torch.randn(1, max_samples, dtype=torch.float32)
dummy_length = torch.tensor([max_samples], dtype=torch.int32)

print("\n=== Step 1: Preprocessor ===")
with torch.inference_mode():
    mel_nemo, mel_length_nemo = asr_model.preprocessor(
        input_signal=dummy_audio, length=dummy_length.long()
    )

prep_model = ct.models.MLModel('build/Preprocessor.mlpackage')
prep_out = prep_model.predict({
    'audio_signal': dummy_audio.numpy(),
    'length': dummy_length.numpy()
})
mel_coreml = prep_out['mel_features']

diff_prep = np.abs(mel_nemo.numpy() - mel_coreml).max()
print(f"Preprocessor max diff: {diff_prep:.6e}")
print(f"NeMo mel range: [{mel_nemo.min():.2f}, {mel_nemo.max():.2f}]")
print(f"CoreML mel range: [{mel_coreml.min():.2f}, {mel_coreml.max():.2f}]")

print("\n=== Step 2: Encoder ===")
with torch.inference_mode():
    encoded_nemo, enc_len_nemo = asr_model.encoder(
        audio_signal=mel_nemo, length=mel_length_nemo.long()
    )

enc_model = ct.models.MLModel('build/Encoder.mlpackage')
enc_out = enc_model.predict({
    'mel_features': mel_nemo.numpy(),
    'mel_length': mel_length_nemo.numpy().astype(np.int32)
})
encoded_coreml = enc_out['encoder_output']

diff_enc = np.abs(encoded_nemo.numpy() - encoded_coreml).max()
print(f"Encoder max diff: {diff_enc:.6e}")
print(f"NeMo encoder range: [{encoded_nemo.min():.2f}, {encoded_nemo.max():.2f}]")
print(f"CoreML encoder range: [{encoded_coreml.min():.2f}, {encoded_coreml.max():.2f}]")

print("\n=== Step 3: CTC Decoder ===")
with torch.inference_mode():
    conv_output = asr_model.ctc_decoder.decoder_layers(encoded_nemo)
    logits_nemo = conv_output.transpose(1, 2)

ctc_model = ct.models.MLModel('build/CtcDecoder.mlpackage')
ctc_out = ctc_model.predict({
    'encoder_output': encoded_nemo.numpy()
})
logits_coreml = ctc_out['ctc_logits']

diff_ctc = np.abs(logits_nemo.numpy() - logits_coreml).max()
print(f"CTC Decoder max diff: {diff_ctc:.6e}")
print(f"NeMo logits range: [{logits_nemo.min():.2f}, {logits_nemo.max():.2f}]")
print(f"CoreML logits range: [{logits_coreml.min():.2f}, {logits_coreml.max():.2f}]")

print("\n=== Step 4: Full Pipeline (Accumulated) ===")
# Use CoreML encoder output as input to CTC decoder
ctc_out_accumulated = ctc_model.predict({
    'encoder_output': encoded_coreml
})
logits_accumulated = ctc_out_accumulated['ctc_logits']

diff_accumulated = np.abs(logits_nemo.numpy() - logits_accumulated).max()
print(f"Accumulated diff (CoreML encoder -> CoreML CTC): {diff_accumulated:.6e}")

# Compare with full pipeline
full_model = ct.models.MLModel('build/FullPipeline.mlpackage')
full_out = full_model.predict({
    'audio_signal': dummy_audio.numpy(),
    'audio_length': dummy_length.numpy()
})
logits_full = full_out['ctc_logits']

diff_full = np.abs(logits_nemo.numpy() - logits_full).max()
print(f"Full pipeline diff: {diff_full:.6e}")

print("\n=== Analysis ===")
print(f"Preprocessor contributes: {diff_prep:.6e}")
print(f"Encoder contributes: {diff_enc:.6e}")
print(f"CTC decoder contributes: {diff_ctc:.6e}")
print(f"Accumulated error: {diff_accumulated:.6e}")
print(f"Full pipeline error: {diff_full:.6e}")

# Calculate relative error
rel_error = diff_full / (logits_nemo.max() - logits_nemo.min())
print(f"\nRelative error: {rel_error:.4f} ({rel_error*100:.2f}%)")

if diff_full < 1.0:
    print("\n✅ Numerical accuracy is EXCELLENT (< 1.0 absolute diff)")
    print("This level of precision is more than sufficient for CTC decoding.")
else:
    print(f"\n⚠️ Higher than expected error: {diff_full:.6e}")
