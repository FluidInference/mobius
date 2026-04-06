#!/usr/bin/env python3
"""Simple comparison: PyTorch vs CoreML on failing sample using high-level APIs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
import sentencepiece as spm
import soundfile as sf
import torch
import tempfile

print("="*70)
print("Compare: Full PyTorch vs CoreML on Failing Sample")
print("="*70)

PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3

# Load CoreML models
print("\n[1/4] Loading CoreML models...")
coreml_encoder = ct.models.MLModel(
    "build/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
coreml_decoder = ct.models.MLModel(
    "build/cohere_decoder_stateful_256.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")
print("   ✓ CoreML models loaded")

# Load PyTorch model
print("\n[2/4] Loading PyTorch model...")
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    pytorch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    pytorch_model.eval()
    processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True)
    print("   ✓ PyTorch model loaded")
except Exception as e:
    print(f"   ❌ Failed to load PyTorch model: {e}")
    exit(1)

# Find the failing sample
print("\n[3/4] Finding failing sample...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
for sample in dataset:
    duration = len(sample['audio']['array']) / 16000.0
    if 23.0 <= duration <= 23.5 and "from the respect paid" in sample['text'].lower():
        break

audio = sample['audio']['array'].astype(np.float32)
ground_truth = sample['text'].lower()
duration = len(audio) / 16000.0

print(f"   ✓ Found sample: {duration:.2f}s")
print(f"   Text: \"{ground_truth[:60]}...\"")

# Process with mel spectrogram
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)
actual_frames = mel.shape[2]

if mel.shape[2] > 3500:
    mel_padded = mel[:, :, :3500]
    actual_frames = 3500
else:
    mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])), mode='constant', constant_values=0)

# Run CoreML pipeline
print("\n[4/4] Comparing pipelines...")
print("\n--- CoreML Pipeline ---")

coreml_encoder_output = coreml_encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([actual_frames], dtype=np.int32)
})

coreml_encoder_hidden = None
for key, value in coreml_encoder_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3:
        coreml_encoder_hidden = value
        break

print(f"Encoder: mean={coreml_encoder_hidden.mean():.6f}, std={coreml_encoder_hidden.std():.6f}")

# Decode with CoreML
cross_attention_mask = np.ones((1, 1, 1, coreml_encoder_hidden.shape[1]), dtype=np.float16)
state = coreml_decoder.make_state()
coreml_tokens = []
last_token = None

for step in range(256):
    if step < len(PROMPT_IDS):
        current_token = PROMPT_IDS[step]
    else:
        current_token = last_token

    decoder_input = {
        "input_id": np.array([[current_token]], dtype=np.int32),
        "encoder_hidden_states": coreml_encoder_hidden.astype(np.float16),
        "cross_attention_mask": cross_attention_mask,
        "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float16),
        "position_ids": np.array([[step]], dtype=np.int32),
    }

    decoder_output = coreml_decoder.predict(decoder_input, state=state)
    next_token = int(np.argmax(decoder_output["logits"][0]))
    last_token = next_token

    if step >= len(PROMPT_IDS) - 1:
        coreml_tokens.append(next_token)
        if next_token == EOS_TOKEN_ID:
            break

coreml_hypothesis = sp.DecodeIds(list(PROMPT_IDS) + coreml_tokens)
for special in ['<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
                '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
                '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
                '<|endoftext|>', '<|en|>']:
    coreml_hypothesis = coreml_hypothesis.replace(special, '')
coreml_hypothesis = coreml_hypothesis.strip().lower()

print(f"Generated {len(coreml_tokens)} tokens")
print(f"Output: \"{coreml_hypothesis[:100]}...\"")

# Run PyTorch pipeline using high-level API
print("\n--- PyTorch Pipeline ---")
with torch.no_grad():
    # Save audio to temp file for processor
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
        sf.write(tmpf.name, audio, 16000)
        tmpfile = tmpf.name

    try:
        # Use processor + model (high-level API)
        result = pytorch_model.transcribe(audio_files=[tmpfile], language="en", processor=processor)
        pytorch_hypothesis = result['text'][0].lower().strip()
    finally:
        Path(tmpfile).unlink()

print(f"Output: \"{pytorch_hypothesis[:100]}...\"")

# Compare
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
print(f"\nGround Truth:")
print(f'  "{ground_truth}"')
print(f"\nCoreML Output:")
print(f'  "{coreml_hypothesis}"')
print(f"\nPyTorch Output:")
print(f'  "{pytorch_hypothesis}"')

# Check transcription quality
gt_start = ground_truth.lower()[:50].replace(".", "").replace(",", "").strip()
pytorch_start = pytorch_hypothesis[:50].replace(".", "").replace(",", "").strip()
coreml_start = coreml_hypothesis[:50].replace(".", "").replace(",", "").strip()

pytorch_matches_gt = gt_start in pytorch_start or pytorch_start in gt_start
coreml_matches_gt = gt_start in coreml_start or coreml_start in gt_start

print(f"\nQuality Assessment:")
print(f"  PyTorch matches ground truth: {'YES' if pytorch_matches_gt else 'NO'}")
print(f"  CoreML matches ground truth: {'YES' if coreml_matches_gt else 'NO'}")

if pytorch_matches_gt and not coreml_matches_gt:
    print(f"\n⚠️  PyTorch is CORRECT, CoreML produces garbage")
    print(f"   → CoreML decoder conversion issue!")
elif coreml_matches_gt and not pytorch_matches_gt:
    print(f"\n⚠️  CoreML is CORRECT, PyTorch produces garbage")
    print(f"   → Unexpected! CoreML decoder is better")
elif not pytorch_matches_gt and not coreml_matches_gt:
    print(f"\n✅ BOTH produce garbage on this sample")
    print(f"   → Model limitation: Weak encoder embeddings cause BOTH decoders to fail")
    print(f"   → This confirms encoder is the root cause, not decoder")
else:
    print(f"\n✓ Both produce correct transcriptions")
    print(f"   → This sample may not be a good test case")

print(f"\n{'='*70}")
