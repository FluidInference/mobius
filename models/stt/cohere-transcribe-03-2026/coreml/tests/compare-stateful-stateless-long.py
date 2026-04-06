#!/usr/bin/env python3
"""Compare stateful vs stateless decoder on long audio.

This determines if long audio failure is specific to stateful decoder
or affects both implementations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
import sentencepiece as spm

print("="*70)
print("Compare: Stateful vs Stateless on Long Audio")
print("="*70)

PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3

# Load models
print("\n[1/3] Loading models...")
encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
stateful_decoder = ct.models.MLModel("build/cohere_decoder_stateful_256.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
stateless_decoder = ct.models.MLModel("build/cohere_decoder_stateless.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")
print("   ✓ Models loaded")

# Find one 20-second sample
print("\n[2/3] Finding a 20-second sample...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
for sample in dataset:
    duration = len(sample['audio']['array']) / 16000.0
    if 19.5 <= duration <= 20.5:
        break

audio = sample['audio']['array'].astype(np.float32)
ground_truth = sample['text'].lower()
duration = len(audio) / 16000.0

print(f"   ✓ Found {duration:.2f}s sample")
print(f"   Ground truth: \"{ground_truth[:80]}...\"")

# Encode (same for both)
print("\n[3/3] Testing both decoders...")
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

cross_attention_mask = np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16)

# Test STATEFUL decoder
print("\n--- Stateful Decoder ---")
state = stateful_decoder.make_state()
stateful_tokens = []
last_token = None

for step in range(256):  # Max 256 tokens
    if step < len(PROMPT_IDS):
        current_token = PROMPT_IDS[step]
    else:
        current_token = last_token

    decoder_input = {
        "input_id": np.array([[current_token]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "cross_attention_mask": cross_attention_mask,
        "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float16),
        "position_ids": np.array([[step]], dtype=np.int32),
    }

    decoder_output = stateful_decoder.predict(decoder_input, state=state)
    next_token = int(np.argmax(decoder_output["logits"][0]))
    last_token = next_token

    if step >= len(PROMPT_IDS) - 1:
        stateful_tokens.append(next_token)
        if next_token == EOS_TOKEN_ID:
            print(f"EOS at step {step}")
            break

# Decode stateful
stateful_all_tokens = list(PROMPT_IDS) + stateful_tokens
stateful_hypothesis = sp.DecodeIds(stateful_all_tokens)
for special in ['<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
                '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
                '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
                '<|endoftext|>', '<|en|>']:
    stateful_hypothesis = stateful_hypothesis.replace(special, '')
stateful_hypothesis = stateful_hypothesis.strip().lower()

print(f"Tokens: {len(stateful_tokens)}")
print(f"Output: \"{stateful_hypothesis[:100]}...\"")

# Test STATELESS decoder
print("\n--- Stateless Decoder ---")
# Build full sequence (prompt + tokens we'll generate)
# For stateless, we need to pass all tokens up to current position

# Start with just prompt
input_ids = list(PROMPT_IDS)
stateless_tokens = []

for gen_step in range(200):  # Generate up to 200 tokens
    # Pass all tokens so far
    decoder_input = {
        "input_ids": np.array([input_ids], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "cross_attention_mask": cross_attention_mask,
    }

    decoder_output = stateless_decoder.predict(decoder_input)
    next_token = int(np.argmax(decoder_output["logits"][0]))

    stateless_tokens.append(next_token)
    input_ids.append(next_token)

    if next_token == EOS_TOKEN_ID:
        print(f"EOS at generation step {gen_step}")
        break

# Decode stateless
stateless_hypothesis = sp.DecodeIds(input_ids)
for special in ['<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
                '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
                '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
                '<|endoftext|>', '<|en|>']:
    stateless_hypothesis = stateless_hypothesis.replace(special, '')
stateless_hypothesis = stateless_hypothesis.strip().lower()

print(f"Tokens: {len(stateless_tokens)}")
print(f"Output: \"{stateless_hypothesis[:100]}...\"")

# Compare
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
print(f"\nGround Truth:")
print(f'  "{ground_truth}"')
print(f"\nStateful Output:")
print(f'  "{stateful_hypothesis}"')
print(f"\nStateless Output:")
print(f'  "{stateless_hypothesis}"')

# Check if they match
if stateful_hypothesis == stateless_hypothesis:
    print(f"\n✅ Both decoders produce IDENTICAL output")
    print(f"   → Issue is NOT specific to stateful implementation")
else:
    print(f"\n⚠️  Decoders produce DIFFERENT output")
    print(f"   → Issue may be specific to stateful implementation")

print(f"\n{'='*70}")
