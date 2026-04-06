#!/usr/bin/env python3
"""Quick test of FP16 stateful decoder on long audio samples."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
from jiwer import wer

print("="*70)
print("Testing FP16 Stateful Decoder on Long Audio (20+ seconds)")
print("="*70)

# Configuration
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 256
TARGET_DURATION_MIN = 20.0
NUM_SAMPLES = 3

print("\n[1/4] Loading CoreML models...")
encoder = ct.models.MLModel("build/cohere_encoder.mlpackage")
decoder = ct.models.MLModel("build/cohere_decoder_stateful_256.mlpackage")
print("   ✓ Loaded encoder and stateful decoder (256 tokens)")

print("\n[2/4] Loading vocabulary...")
import json
with open("hf-upload/vocab.json") as f:
    vocab = json.load(f)
    vocab = {int(k): v for k, v in vocab.items()}
print(f"   ✓ Loaded vocabulary ({len(vocab)} tokens)")

print(f"\n[3/4] Finding {NUM_SAMPLES} long samples from LibriSpeech test-clean...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []
checked = 0

for sample in dataset:
    duration = len(sample['audio']['array']) / 16000.0
    checked += 1

    if duration >= TARGET_DURATION_MIN:
        samples.append(sample)
        print(f"   Found sample {len(samples)}: {duration:.2f}s")

        if len(samples) >= NUM_SAMPLES:
            break

    if checked >= 500:  # Safety limit
        print(f"   Checked {checked} samples, stopping")
        break

if len(samples) == 0:
    print("   ⚠️  No long samples found, using any available")
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    for i, sample in enumerate(dataset):
        if i >= NUM_SAMPLES:
            break
        samples.append(sample)

print(f"\n[4/4] Testing {len(samples)} samples with stateful decoder...")
mel_processor = CohereMelSpectrogram()

results = []
for i, sample in enumerate(samples):
    audio = sample['audio']['array']
    reference = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"\n--- Sample {i+1}/{len(samples)} ({duration:.2f}s) ---")
    print(f"Reference: {reference[:80]}...")

    # Compute mel
    mel = mel_processor(audio)
    mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])))

    # Encode
    encoder_out = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([mel.shape[2]], dtype=np.int32)
    })

    # Find encoder output (look for 3D tensor)
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
    # Note: Stateful decoder uses CoreML State API, cache is GPU-resident
    state = decoder.make_state()  # Create state object for CoreML State API
    tokens = []
    last_token = None

    for step in range(MAX_NEW_TOKENS):
        # For first 10 steps, feed prompt tokens
        if step < len(PROMPT_IDS):
            current_token = PROMPT_IDS[step]
        else:
            current_token = last_token

        # Input: single token at current position
        input_id = np.array([[current_token]], dtype=np.int32)
        # Attention mask shape determines position: [1, 1, 1, total_seq_len]
        attention_mask = np.zeros((1, 1, 1, step + 1), dtype=np.float16)
        # Position IDs: just the current position
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

    # Decode text (tokens already excludes prompt)
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

    print(f"Hypothesis: {hypothesis[:80]}...")
    print(f"Tokens: {len(tokens)}, WER: {error_rate*100:.1f}%")

    results.append({
        "duration": duration,
        "wer": error_rate,
        "tokens": len(tokens),
        "reference": reference,
        "hypothesis": hypothesis
    })

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
avg_wer = np.mean([r["wer"] for r in results]) * 100
avg_duration = np.mean([r["duration"] for r in results])
print(f"Samples: {len(results)}")
print(f"Average duration: {avg_duration:.1f}s")
print(f"Average WER: {avg_wer:.1f}%")
print(f"Perfect matches: {sum(1 for r in results if r['wer'] < 0.05)} / {len(results)}")
print("\nConclusion: Stateful decoder " + ("✅ WORKS" if avg_wer < 30 else "❌ HAS ISSUES"))
