#!/usr/bin/env python3
"""Test stateful decoder across different audio lengths to find failure threshold.

This will help identify at what audio duration the model starts failing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
import sentencepiece as spm
import re

print("="*70)
print("Cohere Transcribe - Audio Length Sweep Test")
print("="*70)

# Configuration
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 200
MAX_SEQ_LEN = 256

# Test different length buckets
LENGTH_BUCKETS = [
    (3, 5, "Very Short", 5),
    (8, 12, "Short", 5),
    (15, 18, "Medium", 5),
    (20, 23, "Long", 5),
]

# Load models
print("\n[1/4] Loading CoreML models...")
encoder = ct.models.MLModel(
    "build/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
stateful_decoder = ct.models.MLModel(
    "build/cohere_decoder_stateful_256.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
print("   ✓ Models loaded")

# Load tokenizer
print("\n[2/4] Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")
print("   ✓ Tokenizer loaded")

# Process each length bucket
print("\n[3/4] Testing across audio length ranges...")
mel_processor = CohereMelSpectrogram()

all_results = []

for min_dur, max_dur, bucket_name, num_samples in LENGTH_BUCKETS:
    print(f"\n{'='*70}")
    print(f"Testing: {bucket_name} ({min_dur}-{max_dur}s) - Finding {num_samples} samples...")
    print(f"{'='*70}")

    # Find samples in this range
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    samples = []
    checked = 0

    for sample in dataset:
        duration = len(sample['audio']['array']) / 16000.0
        checked += 1

        if min_dur <= duration <= max_dur:
            samples.append(sample)
            if len(samples) >= num_samples:
                break

        if checked >= 500:
            break

    print(f"   Found {len(samples)} samples (checked {checked})")

    if len(samples) == 0:
        continue

    # Test samples in this bucket
    bucket_results = []

    for sample_idx, sample in enumerate(samples):
        audio = sample['audio']['array'].astype(np.float32)
        ground_truth = sample['text'].lower()
        duration = len(audio) / 16000.0

        # Compute mel spectrogram
        mel = mel_processor(audio)
        if mel.shape[2] > 3500:
            mel_padded = mel[:, :, :3500]
        else:
            mel_padded = np.pad(
                mel,
                ((0, 0), (0, 0), (0, 3500 - mel.shape[2])),
                mode='constant',
                constant_values=0
            )

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

        # Decode
        state = stateful_decoder.make_state()
        tokens = []
        last_token = None
        max_steps = min(MAX_NEW_TOKENS + len(PROMPT_IDS), MAX_SEQ_LEN)

        for step in range(max_steps):
            if step < len(PROMPT_IDS):
                current_token = PROMPT_IDS[step]
            else:
                current_token = last_token

            attention_mask = np.zeros((1, 1, 1, step + 1), dtype=np.float16)
            position_ids = np.array([[step]], dtype=np.int32)

            decoder_input = {
                "input_id": np.array([[current_token]], dtype=np.int32),
                "encoder_hidden_states": encoder_hidden.astype(np.float16),
                "cross_attention_mask": cross_attention_mask,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

            decoder_output = stateful_decoder.predict(decoder_input, state=state)
            logits = decoder_output["logits"]
            next_token = int(np.argmax(logits[0]))
            last_token = next_token

            if step >= len(PROMPT_IDS) - 1:
                tokens.append(next_token)
                if next_token == EOS_TOKEN_ID:
                    break

        # Decode tokens
        all_tokens = list(PROMPT_IDS) + tokens
        hypothesis = sp.DecodeIds(all_tokens)

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

        # Check accuracy (ignoring punctuation)
        gt_clean = re.sub(r'[^\w\s]', '', ground_truth.lower()).strip()
        hyp_clean = re.sub(r'[^\w\s]', '', hypothesis.lower()).strip()
        is_perfect = gt_clean == hyp_clean

        result = {
            'bucket': bucket_name,
            'duration': duration,
            'ground_truth': ground_truth,
            'hypothesis': hypothesis,
            'tokens': len(tokens),
            'perfect': is_perfect,
        }
        bucket_results.append(result)
        all_results.append(result)

        status = "✅" if is_perfect else "❌"
        print(f"   Sample {sample_idx+1}: {duration:.2f}s - {status}")

    # Bucket summary
    perfect_count = sum(1 for r in bucket_results if r['perfect'])
    print(f"\n   {bucket_name} Summary: {perfect_count}/{len(bucket_results)} perfect ({perfect_count/len(bucket_results)*100:.1f}%)")

# Overall summary
print("\n" + "="*70)
print("SUMMARY - Audio Length Sweep")
print("="*70)

for bucket_name in [b[2] for b in LENGTH_BUCKETS]:
    bucket_samples = [r for r in all_results if r['bucket'] == bucket_name]
    if bucket_samples:
        perfect_count = sum(1 for r in bucket_samples if r['perfect'])
        avg_duration = sum(r['duration'] for r in bucket_samples) / len(bucket_samples)
        print(f"\n{bucket_name} ({avg_duration:.1f}s avg):")
        print(f"  Perfect: {perfect_count}/{len(bucket_samples)} ({perfect_count/len(bucket_samples)*100:.1f}%)")

        # Show a sample hypothesis
        if bucket_samples:
            sample = bucket_samples[0]
            print(f"  Example GT:  \"{sample['ground_truth'][:60]}...\"")
            print(f"  Example Hyp: \"{sample['hypothesis'][:60]}...\"")

print(f"\n{'='*70}")
