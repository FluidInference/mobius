#!/usr/bin/env python3
"""Test INT8 stateful decoder on LibriSpeech test-clean samples.

This verifies the INT8 quantized models produce acceptable quality.
We test on 10 short samples (3-10s) to verify baseline quality is maintained.
"""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
from jiwer import wer
import json

print("="*70)
print("Testing INT8 Stateful Decoder Quality")
print("="*70)

# Configuration
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 256
NUM_SAMPLES = 10

print("\n[1/5] Loading INT8 CoreML models...")
encoder = ct.models.MLModel("build-35s-int8/cohere_encoder_int8.mlpackage")
decoder = ct.models.MLModel("build-35s-int8/cohere_decoder_stateful_int8.mlpackage")
print("   ✓ Loaded INT8 encoder and decoder")

# Check sizes
encoder_path = "build-35s-int8/cohere_encoder_int8.mlpackage"
decoder_path = "build-35s-int8/cohere_decoder_stateful_int8.mlpackage"
from pathlib import Path
encoder_size = sum(f.stat().st_size for f in Path(encoder_path).rglob('*') if f.is_file()) / 1024**3
decoder_size = sum(f.stat().st_size for f in Path(decoder_path).rglob('*') if f.is_file()) / 1024**2
print(f"   Encoder: {encoder_size:.2f} GB")
print(f"   Decoder: {decoder_size:.1f} MB")
print(f"   Total: {encoder_size:.2f} GB + {decoder_size:.1f} MB")

print("\n[2/5] Loading vocabulary...")
with open("hf-upload/vocab.json") as f:
    vocab = json.load(f)
    vocab = {int(k): v for k, v in vocab.items()}
print(f"   ✓ Loaded vocabulary ({len(vocab)} tokens)")

print(f"\n[3/5] Loading {NUM_SAMPLES} samples from LibriSpeech test-clean...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []
for i, sample in enumerate(dataset):
    if i >= NUM_SAMPLES:
        break
    samples.append(sample)
    duration = len(sample['audio']['array']) / 16000.0
    print(f"   Sample {i+1}: {duration:.2f}s")

print(f"\n[4/5] Testing {len(samples)} samples with INT8 models...")
mel_processor = CohereMelSpectrogram()

results = []
for i, sample in enumerate(samples):
    audio = sample['audio']['array']
    reference = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"\n--- Sample {i+1}/{len(samples)} ({duration:.2f}s) ---")

    # Compute mel
    mel = mel_processor(audio)
    # Use 3500 frames (35 seconds) to get 438 encoder outputs
    if mel.shape[2] > 3500:
        mel_padded = mel[:, :, :3500]
    else:
        mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])))

    # Encode
    encoder_out = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([mel.shape[2]], dtype=np.int32)
    })

    # Find encoder output
    encoder_hidden = None
    for key, value in encoder_out.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden = value
            break

    if encoder_hidden is None:
        print("   ❌ Could not find encoder output")
        continue

    enc_seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, enc_seq_len), dtype=np.float16)

    # Decode with stateful decoder
    state = decoder.make_state()
    tokens = []
    last_token = None

    for step in range(MAX_NEW_TOKENS):
        # For first 10 steps, feed prompt tokens
        if step < len(PROMPT_IDS):
            current_token = PROMPT_IDS[step]
        else:
            current_token = last_token

        input_id = np.array([[current_token]], dtype=np.int32)
        attention_mask = np.zeros((1, 1, 1, step + 1), dtype=np.float16)
        position_ids = np.array([[step]], dtype=np.int32)

        decoder_out = decoder.predict({
            "input_id": input_id,
            "encoder_hidden_states": encoder_hidden.astype(np.float16),
            "attention_mask": attention_mask,
            "cross_attention_mask": cross_mask,
            "position_ids": position_ids,
        }, state=state)

        next_token = int(np.argmax(decoder_out["logits"][0]))
        last_token = next_token

        # Collect tokens after prompt
        if step >= len(PROMPT_IDS) - 1:
            tokens.append(next_token)
            if next_token == EOS_TOKEN_ID:
                break

    # Decode text
    text_tokens = []
    for token_id in tokens:
        if token_id <= 4 or token_id == EOS_TOKEN_ID:
            continue
        token_str = vocab.get(token_id, "")
        if token_str.startswith("<|"):
            continue
        text_tokens.append(token_str)

    hypothesis = "".join(text_tokens).replace("▁", " ").strip().lower()
    error_rate = wer(reference, hypothesis)

    print(f"Reference:  {reference[:70]}...")
    print(f"Hypothesis: {hypothesis[:70]}...")
    print(f"Tokens: {len(tokens)}, WER: {error_rate*100:.1f}%")

    results.append({
        "duration": duration,
        "wer": error_rate,
        "tokens": len(tokens),
        "reference": reference,
        "hypothesis": hypothesis,
        "perfect": error_rate < 0.05
    })

# Summary
print("\n" + "="*70)
print("INT8 QUALITY RESULTS")
print("="*70)
avg_wer = np.mean([r["wer"] for r in results]) * 100
avg_duration = np.mean([r["duration"] for r in results])
perfect_count = sum(1 for r in results if r["perfect"])

print(f"Samples tested: {len(results)}")
print(f"Average duration: {avg_duration:.1f}s")
print(f"Average WER: {avg_wer:.1f}%")
print(f"Perfect matches: {perfect_count} / {len(results)} ({100*perfect_count/len(results):.0f}%)")
print(f"Model size: {encoder_size:.2f} GB (encoder) + {decoder_size:.1f} MB (decoder)")

# Verdict
print("\n" + "="*70)
if avg_wer < 30 and perfect_count >= len(results) * 0.6:
    print("✅ INT8 MODELS READY FOR UPLOAD")
    print("Quality is acceptable (< 30% WER, 60%+ perfect matches)")
elif avg_wer < 50:
    print("⚠️  INT8 MODELS NEED REVIEW")
    print(f"Quality is degraded ({avg_wer:.1f}% WER)")
    print("Consider using FP16 or improving quantization")
else:
    print("❌ INT8 MODELS FAILED")
    print(f"Quality is too poor ({avg_wer:.1f}% WER)")
    print("Do not use these models")

print("="*70)
