#!/usr/bin/env python3
"""Investigate why certain 20s samples produce garbage while others work.

Compare:
1. One working sample (19.81s: "for general service...")
2. One failing sample (that produces garbage)

To find what's different.
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
print("Investigate: Why Do Some Long Samples Produce Garbage?")
print("="*70)

PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3

# Load models
print("\n[1/4] Loading models...")
encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
stateful_decoder = ct.models.MLModel("build/cohere_decoder_stateful_256.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")
print("   ✓ Models loaded")

# Find specific samples
print("\n[2/4] Finding samples...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

samples_to_find = [
    ("Working", 19.5, 20.5, "for general service"),  # Known working
    ("Failing", 22.0, 23.5, "from the respect paid"),  # Known failing (23.32s)
]

found_samples = {}

for label, min_dur, max_dur, text_snippet in samples_to_find:
    for sample in dataset:
        duration = len(sample['audio']['array']) / 16000.0
        if min_dur <= duration <= max_dur and text_snippet in sample['text'].lower():
            found_samples[label] = sample
            print(f"   ✓ Found {label} sample: {duration:.2f}s")
            print(f"     Text: \"{sample['text'][:60]}...\"")
            break
    # Reset dataset
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

if len(found_samples) < 2:
    print("\n❌ Could not find both samples. Using any 20s samples instead...")
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    found_count = 0
    for sample in dataset:
        duration = len(sample['audio']['array']) / 16000.0
        if 19.5 <= duration <= 23.5:
            label = f"Sample_{found_count + 1}"
            found_samples[label] = sample
            found_count += 1
            print(f"   Using sample {found_count}: {duration:.2f}s")
            if found_count >= 2:
                break

# Process each sample
print("\n[3/4] Processing and comparing...")
mel_processor = CohereMelSpectrogram()

for label, sample in found_samples.items():
    print(f"\n{'='*70}")
    print(f"{label} Sample Analysis")
    print(f"{'='*70}")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"\nDuration: {duration:.2f}s")
    print(f"Ground truth: \"{ground_truth[:80]}...\"")

    # Encode
    mel = mel_processor(audio)
    if mel.shape[2] > 3500:
        mel_padded = mel[:, :, :3500]
        actual_frames = 3500
    else:
        mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])), mode='constant', constant_values=0)
        actual_frames = mel.shape[2]

    encoder_output = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([actual_frames], dtype=np.int32)
    })

    encoder_hidden = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden = value
            break

    # Analyze encoder output
    print(f"\nEncoder output:")
    print(f"  Shape: {encoder_hidden.shape}")
    print(f"  Mean: {encoder_hidden.mean():.6f}")
    print(f"  Std:  {encoder_hidden.std():.6f}")
    print(f"  Min:  {encoder_hidden.min():.6f}")
    print(f"  Max:  {encoder_hidden.max():.6f}")

    # Check for anomalies in encoder output
    has_nan = np.isnan(encoder_hidden).any()
    has_inf = np.isinf(encoder_hidden).any()
    very_small_std = encoder_hidden.std() < 0.1
    very_large_values = (np.abs(encoder_hidden) > 10).sum()

    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Very small std: {very_small_std}")
    print(f"  Values > 10: {very_large_values}")

    # Decode
    cross_attention_mask = np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16)
    state = stateful_decoder.make_state()
    tokens = []
    last_token = None

    # Generate tokens and track first 20 predictions
    first_predictions = []

    for step in range(256):
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
        logits = decoder_output["logits"][0]
        next_token = int(np.argmax(logits))
        last_token = next_token

        if step >= len(PROMPT_IDS) - 1:
            tokens.append(next_token)

            # Track first 20 generated tokens with logit stats
            if len(tokens) <= 20:
                top5_indices = np.argsort(logits)[-5:][::-1]
                top5_probs = np.exp(logits[top5_indices]) / np.exp(logits).sum()
                first_predictions.append({
                    'step': step,
                    'token': next_token,
                    'token_text': sp.DecodeIds([next_token]),
                    'top_prob': float(top5_probs[0]),
                    'logit_max': float(logits.max()),
                    'logit_std': float(logits.std()),
                })

            if next_token == EOS_TOKEN_ID:
                break

    # Decode hypothesis
    all_tokens = list(PROMPT_IDS) + tokens
    hypothesis = sp.DecodeIds(all_tokens)
    for special in ['<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
                    '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
                    '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
                    '<|endoftext|>', '<|en|>']:
        hypothesis = hypothesis.replace(special, '')
    hypothesis = hypothesis.strip().lower()

    print(f"\nGenerated {len(tokens)} tokens")
    print(f"Hypothesis: \"{hypothesis[:100]}...\"")

    # Show first 20 generated tokens
    print(f"\nFirst 20 generated tokens:")
    for pred in first_predictions[:20]:
        print(f"  Step {pred['step']:3d}: token={pred['token']:5d} '{pred['token_text']:15s}' "
              f"prob={pred['top_prob']:.3f} logit_max={pred['logit_max']:6.2f} logit_std={pred['logit_std']:5.2f}")

    # Check for repetition pattern
    if len(tokens) >= 10:
        # Check if same tokens repeat
        token_set = set(tokens[:20])
        unique_ratio = len(token_set) / min(20, len(tokens))
        print(f"\nToken diversity (first 20):")
        print(f"  Unique tokens: {len(token_set)}")
        print(f"  Unique ratio: {unique_ratio:.2f}")
        if unique_ratio < 0.3:
            print(f"  ⚠️  Very low diversity - likely repetitive output!")

# Compare
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")

if len(found_samples) == 2:
    labels = list(found_samples.keys())
    print(f"\nCompare {labels[0]} vs {labels[1]}:")
    print("\nLook for differences in:")
    print("  1. Encoder output statistics (mean, std, anomalies)")
    print("  2. Token generation patterns (diversity, confidence)")
    print("  3. Logit statistics (max, std)")
    print("\nThese differences may explain why one works and one fails.")

print(f"\n{'='*70}")
