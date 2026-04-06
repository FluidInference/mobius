#!/usr/bin/env python3
"""Debug cache growth to understand the repetition issue."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset

print("="*70)
print("Debug: Cache Growth Analysis")
print("="*70)

# Load one short sample
print("\n[1/4] Loading sample...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
sample = next(iter(dataset))
audio = sample['audio']['array'].astype(np.float32)
ground_truth = sample['text'].lower()
print(f"   Ground truth: \"{ground_truth}\"")

# Load models
print("\n[2/4] Loading models...")
encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
decoder = ct.models.MLModel("build/cohere_decoder_cached.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)

# Process audio
print("\n[3/4] Encoding audio...")
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)
mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant', constant_values=0)
encoder_output = encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([mel.shape[2]], dtype=np.int32)
})
encoder_hidden = None
for key, value in encoder_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3:
        encoder_hidden = value
        break

# Decode with cache debugging
print("\n[4/4] Decoding with cache debugging...")
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3

tokens = list(PROMPT_IDS)
cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

# Process prompt tokens
print("\n   Processing prompt tokens:")
for step, token_id in enumerate(PROMPT_IDS):
    decoder_input = {
        "input_id": np.array([[token_id]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "step": np.array([step], dtype=np.int32),
        "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16),
        "cache_k": cache_k,
        "cache_v": cache_v,
    }

    decoder_output = decoder.predict(decoder_input)

    # Extract cache
    for key, value in decoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 4:
            if 'k' in key.lower() or key == 'new_cache_k':
                new_cache_k = value
            else:
                new_cache_v = value

    # Check how many non-zero positions in cache
    # Sum across layer, head and hidden dims, check which sequence positions are non-zero
    cache_k_norms = np.sqrt(np.sum(new_cache_k**2, axis=(0, 1, 3)))  # (108,)
    cache_v_norms = np.sqrt(np.sum(new_cache_v**2, axis=(0, 1, 3)))  # (108,)

    num_nonzero_k = np.sum(cache_k_norms > 1e-8)  # Lower threshold for FP16
    num_nonzero_v = np.sum(cache_v_norms > 1e-8)

    max_k_norm = np.max(cache_k_norms)
    max_v_norm = np.max(cache_v_norms)

    print(f"     Step {step}: cache_k has {num_nonzero_k} non-zero (max norm: {max_k_norm:.6f}), cache_v has {num_nonzero_v} (max norm: {max_v_norm:.6f})")

    cache_k = new_cache_k
    cache_v = new_cache_v

# Generate a few tokens with debugging
print("\n   Generating tokens:")
for i in range(20):
    step = len(PROMPT_IDS) + i
    decoder_input = {
        "input_id": np.array([[tokens[-1]]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "step": np.array([step], dtype=np.int32),
        "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16),
        "cache_k": cache_k,
        "cache_v": cache_v,
    }

    decoder_output = decoder.predict(decoder_input)

    logits = None
    for key, value in decoder_output.items():
        if hasattr(value, 'shape'):
            if len(value.shape) == 2 and value.shape[1] > 1000:
                logits = value
            elif len(value.shape) == 4:
                if 'k' in key.lower() or key == 'new_cache_k':
                    new_cache_k = value
                else:
                    new_cache_v = value

    next_token = int(np.argmax(logits[0]))
    tokens.append(next_token)

    # Check cache growth
    cache_k_norms = np.sqrt(np.sum(new_cache_k**2, axis=(0, 1, 3)))
    cache_v_norms = np.sqrt(np.sum(new_cache_v**2, axis=(0, 1, 3)))

    num_nonzero_k = np.sum(cache_k_norms > 1e-8)
    num_nonzero_v = np.sum(cache_v_norms > 1e-8)

    max_k_norm = np.max(cache_k_norms)
    max_v_norm = np.max(cache_v_norms)

    print(f"     Step {step}: token={next_token}, cache_k has {num_nonzero_k} non-zero (max: {max_k_norm:.6f}), cache_v has {num_nonzero_v} (max: {max_v_norm:.6f})")

    cache_k = new_cache_k
    cache_v = new_cache_v

    if next_token == EOS_TOKEN_ID:
        print(f"     EOS generated at step {step}")
        break

# Decode tokens
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")
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

print(f"\n{'='*70}")
print(f"Ground truth: \"{ground_truth}\"")
print(f"Hypothesis:   \"{hypothesis}\"")
print(f"{'='*70}")
