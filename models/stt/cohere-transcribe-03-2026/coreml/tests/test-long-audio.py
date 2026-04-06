#!/usr/bin/env python3
"""Test the stateful CoreML decoder on longer audio samples (30-40 seconds).

This tests performance on longer-form audio.
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
print("Cohere Transcribe - Long Audio Test (20-28 seconds)")
print("="*70)

# Configuration
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 200
MAX_SEQ_LEN = 256  # Using 256-token decoder
TARGET_DURATION_MIN = 20.0  # seconds
TARGET_DURATION_MAX = 28.0  # seconds (max ~30s due to encoder 3500 frame limit)
NUM_SAMPLES = 10

# Load LibriSpeech test-clean and filter for longer samples
print(f"\n[1/5] Finding {NUM_SAMPLES} samples between {TARGET_DURATION_MIN}-{TARGET_DURATION_MAX}s...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []
checked = 0

for sample in dataset:
    duration = len(sample['audio']['array']) / 16000.0
    checked += 1

    if TARGET_DURATION_MIN <= duration <= TARGET_DURATION_MAX:
        samples.append(sample)
        print(f"   Found sample {len(samples)}: {duration:.2f}s - \"{sample['text'][:80]}...\"")

        if len(samples) >= NUM_SAMPLES:
            break

    if checked % 100 == 0:
        print(f"   Checked {checked} samples, found {len(samples)} matches...")

    if checked >= 1000:  # Safety limit
        print(f"   ⚠️  Reached check limit of 1000 samples")
        break

print(f"   ✓ Found {len(samples)} samples (checked {checked} total)")

if len(samples) == 0:
    print("\n❌ No samples found in target duration range")
    exit(1)

# Load models
print("\n[2/5] Loading CoreML models...")
try:
    encoder = ct.models.MLModel(
        "build/cohere_encoder.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    stateful_decoder = ct.models.MLModel(
        "build/cohere_decoder_stateful_256.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print(f"   ✓ Models loaded")
except Exception as e:
    print(f"   ❌ Error loading models: {e}")
    exit(1)

# Load tokenizer
print("\n[3/5] Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")
print(f"   ✓ Tokenizer loaded")

# Process samples
print(f"\n[4/5] Processing {len(samples)} long audio samples...")
mel_processor = CohereMelSpectrogram()
results = []

for sample_idx, sample in enumerate(samples):
    print(f"\n   Sample {sample_idx + 1}/{len(samples)}:")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"     Duration: {duration:.2f}s")
    print(f"     Ground truth: \"{ground_truth[:100]}...\"")

    # Compute mel spectrogram
    mel = mel_processor(audio)

    # Encoder max is 3500 frames - truncate or pad as needed
    if mel.shape[2] > 3500:
        print(f"       ⚠️  Mel is {mel.shape[2]} frames, truncating to 3500")
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

    # Decode with stateful decoder
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
                print(f"       EOS at step {step}")
                break

    if step >= MAX_SEQ_LEN - 1:
        print(f"       ⚠️  Hit max sequence length ({MAX_SEQ_LEN})")

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

    print(f"     Hypothesis:   \"{hypothesis[:100]}...\"")
    print(f"     Tokens: {len(tokens)}")

    # Simple accuracy check (not WER)
    import re
    gt_clean = re.sub(r'[^\w\s]', '', ground_truth.lower()).strip()
    hyp_clean = re.sub(r'[^\w\s]', '', hypothesis.lower()).strip()
    is_perfect = gt_clean == hyp_clean

    status = "✅ Perfect" if is_perfect else "❌ Has differences"
    print(f"     Status: {status}")

    results.append({
        'sample_idx': sample_idx,
        'duration': duration,
        'ground_truth': ground_truth,
        'hypothesis': hypothesis,
        'tokens': len(tokens),
        'hit_limit': step >= MAX_SEQ_LEN - 1,
        'perfect': is_perfect,
    })

# Summary
print("\n" + "="*70)
print("RESULTS - Long Audio Test (20-28s samples)")
print("="*70)

total_duration = 0
perfect_count = 0
hit_limit_count = 0

for result in results:
    print(f"\nSample {result['sample_idx'] + 1}:")
    print(f"  Duration:     {result['duration']:.2f}s")
    print(f"  Ground truth: \"{result['ground_truth'][:60]}...\"")
    print(f"  Hypothesis:   \"{result['hypothesis'][:60]}...\"")
    print(f"  Tokens:       {result['tokens']}")
    if result['hit_limit']:
        print(f"  ⚠️  Hit max length limit")
        hit_limit_count += 1

    status = "✅ PERFECT" if result['perfect'] else "❌ Different"
    print(f"  Status:       {status}")

    total_duration += result['duration']
    if result['perfect']:
        perfect_count += 1

print(f"\n{'='*70}")
print("SUMMARY - Long Audio")
print(f"{'='*70}")
print(f"Samples:       {len(results)}")
print(f"Total audio:   {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
print(f"Avg duration:  {total_duration/len(results):.2f}s")
print(f"Perfect:       {perfect_count}/{len(results)}")
print(f"Hit limit:     {hit_limit_count}/{len(results)}")
print(f"{'='*70}")

if hit_limit_count > 0:
    print(f"\n⚠️  {hit_limit_count} samples hit the {MAX_SEQ_LEN} token limit")
    print(f"   Consider re-exporting decoder with --max-seq-len 256 for longer audio")
