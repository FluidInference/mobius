#!/usr/bin/env python3
"""Test the FIXED CoreML decoder."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
import sentencepiece as spm

print("="*70)
print("Test FIXED CoreML Decoder")
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
decoder = ct.models.MLModel("build/cohere_decoder_fixed.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
print(f"   ✓ Loaded FIXED decoder")

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

    # Initialize cache
    cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
    cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)
    cross_attention_mask = np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16)

    # Process prompt tokens
    tokens = list(PROMPT_IDS)
    for step, token_id in enumerate(PROMPT_IDS):
        decoder_input = {
            "input_id": np.array([[token_id]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden.astype(np.float16),
            "cache_k": cache_k,
            "cache_v": cache_v,
            "step": np.array([step], dtype=np.int32),
            "cross_attention_mask": cross_attention_mask,
        }

        decoder_output = decoder.predict(decoder_input)

        for key, value in decoder_output.items():
            if hasattr(value, 'shape') and len(value.shape) == 4:
                if 'k' in key.lower() or key.startswith('new_cache_k'):
                    cache_k = value
                else:
                    cache_v = value

    # Generate new tokens
    for step in range(len(PROMPT_IDS), len(PROMPT_IDS) + MAX_NEW_TOKENS):
        decoder_input = {
            "input_id": np.array([[tokens[-1]]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden.astype(np.float16),
            "cache_k": cache_k,
            "cache_v": cache_v,
            "step": np.array([step], dtype=np.int32),
            "cross_attention_mask": cross_attention_mask,
        }

        decoder_output = decoder.predict(decoder_input)

        logits = None
        for key, value in decoder_output.items():
            if hasattr(value, 'shape'):
                if len(value.shape) == 2 and value.shape[1] > 1000:
                    logits = value
                elif len(value.shape) == 4:
                    if 'k' in key.lower() or key.startswith('new_cache_k'):
                        cache_k = value
                    else:
                        cache_v = value

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
print("FIXED CoreML TEST COMPLETE")
print("="*70)
print("✅ If transcriptions are perfect with no repetitions, the fix works!")
