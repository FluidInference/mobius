#!/usr/bin/env python3
"""Test the STATELESS CoreML decoder (no cache, reprocesses all tokens)."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
import sentencepiece as spm

print("="*70)
print("Test STATELESS CoreML Decoder (No Cache, O(n^2))")
print("="*70)

# Configuration
NUM_SAMPLES = 3
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 200

# Load LibriSpeech samples
print(f"\n[1/4] Loading {NUM_SAMPLES} samples from LibriSpeech test-clean...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []
for i, sample in enumerate(dataset):
    if i >= NUM_SAMPLES:
        break
    samples.append(sample)
print(f"   ✓ Loaded {len(samples)} samples")

# Load models
print("\n[2/4] Loading CoreML models...")
encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
decoder = ct.models.MLModel("build/cohere_decoder_stateless.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
print(f"   ✓ Loaded STATELESS decoder (no cache)")

# Load tokenizer
print("\n[3/4] Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")
print(f"   ✓ Tokenizer loaded")

# Process samples
print(f"\n[4/4] Testing {NUM_SAMPLES} samples...")
mel_processor = CohereMelSpectrogram()

for sample_idx, sample in enumerate(samples):
    print(f"\n   Sample {sample_idx + 1}/{NUM_SAMPLES}:")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"     Duration: {duration:.2f}s")
    print(f"     Ground truth: \"{ground_truth}\"")

    # Compute mel spectrogram
    mel = mel_processor(audio)
    mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant', constant_values=0)

    # Encode
    encoder_output = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([mel.shape[2]], dtype=np.int32)
    })

    encoder_hidden = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden = value
            break

    cross_attention_mask = np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16)

    # Start with prompt tokens
    tokens = list(PROMPT_IDS)

    # Generate new tokens
    for step in range(len(PROMPT_IDS), len(PROMPT_IDS) + MAX_NEW_TOKENS):
        # Pass ALL tokens so far (stateless approach)
        input_ids = np.array([tokens], dtype=np.int32)  # (1, seq_len)

        decoder_input = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden.astype(np.float16),
            "cross_attention_mask": cross_attention_mask,
        }

        decoder_output = decoder.predict(decoder_input)

        # Extract logits
        logits = None
        for key, value in decoder_output.items():
            if hasattr(value, 'shape') and len(value.shape) == 2 and value.shape[1] > 1000:
                logits = value
                break

        next_token = int(np.argmax(logits[0]))
        tokens.append(next_token)

        if next_token == EOS_TOKEN_ID:
            break

    # Decode tokens
    hypothesis = sp.DecodeIds(tokens)

    # Remove special tokens
    special_tokens = [
        '<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
        '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
        '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
        '<|endoftext|>', '<|en|>'
    ]
    for special in special_tokens:
        hypothesis = hypothesis.replace(special, '')
    hypothesis = hypothesis.strip().lower()

    print(f"     Hypothesis:   \"{hypothesis}\"")
    print(f"     Tokens:       {len(tokens) - len(PROMPT_IDS)}")

print("\n" + "="*70)
print("STATELESS CoreML TEST COMPLETE")
print("="*70)
print("✅ If transcriptions are perfect, the stateless approach works!")
print("⚠️  Note: This is O(n^2) - slower but simpler than cache-based approach")
