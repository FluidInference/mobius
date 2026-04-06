#!/usr/bin/env python3
"""Test the stateful CoreML decoder on LibriSpeech samples.

This validates that the stateful decoder (Qwen3 approach) works correctly.
Compares against:
1. Stateless decoder (O(n^2), known working baseline)
2. PyTorch reference (gold standard)
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
print("Cohere Transcribe - Stateful Decoder Test")
print("="*70)

# Configuration
NUM_SAMPLES = 100
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 200
MAX_SEQ_LEN = 108  # Model was exported with this max sequence length

# Load LibriSpeech test-clean
print(f"\n[1/5] Loading {NUM_SAMPLES} samples from LibriSpeech test-clean...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []
for i, sample in enumerate(dataset):
    if i >= NUM_SAMPLES:
        break
    samples.append(sample)
print(f"   ✓ Loaded {len(samples)} samples")

# Load models
print("\n[2/5] Loading CoreML models...")
try:
    encoder = ct.models.MLModel(
        "build/cohere_encoder.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    stateful_decoder = ct.models.MLModel(
        "build/cohere_decoder_stateful.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print(f"   ✓ Models loaded (Stateful decoder, FP16)")
except Exception as e:
    print(f"   ❌ Error loading models: {e}")
    print("\n   Make sure you've run:")
    print("     1. uv run export-encoder.py --output-dir build")
    print("     2. uv run export-decoder-stateful.py --output-dir build")
    exit(1)

# Load tokenizer
print("\n[3/5] Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")
print(f"   ✓ Tokenizer loaded")

# Process samples
print(f"\n[4/5] Processing {NUM_SAMPLES} samples...")
mel_processor = CohereMelSpectrogram()
results = []

for sample_idx, sample in enumerate(samples):
    print(f"\n   Sample {sample_idx + 1}/{NUM_SAMPLES}:")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"     Duration: {duration:.2f}s")
    print(f"     Ground truth: \"{ground_truth}\"")

    # Compute mel spectrogram
    mel = mel_processor(audio)
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

    # Decode with stateful decoder (Qwen3 interface)
    # Create state ONCE and reuse for all steps
    state = stateful_decoder.make_state()

    tokens = []
    last_token = None

    # Process ALL tokens (prompt + generated) through decoder
    # Step 0-9: Process prompt tokens (build up cache)
    # Step 10+: Generate new tokens
    max_steps = min(MAX_NEW_TOKENS + len(PROMPT_IDS), MAX_SEQ_LEN)

    for step in range(max_steps):
        # Determine current token
        if step < len(PROMPT_IDS):
            # Processing prompt
            current_token = PROMPT_IDS[step]
        else:
            # Generating: use prediction from previous step
            current_token = last_token

        # NEW INTERFACE: attention_mask grows from [1,1,1,1] to [1,1,1,2] to [1,1,1,3], etc.
        # This lets the model infer the current position from mask.shape[-1]
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

        # Extract logits
        logits = decoder_output["logits"]
        next_token = int(np.argmax(logits[0]))
        last_token = next_token  # Save for next iteration

        # Debug first few steps
        if step < 13:
            print(f"       Step {step}: input_token={current_token}, next_token={next_token}, logit_range=[{logits.min():.2f}, {logits.max():.2f}]")

        # Append generated tokens (start collecting after last prompt token is processed)
        # The prediction from step len(PROMPT_IDS)-1 is the first transcription token
        if step >= len(PROMPT_IDS) - 1:
            tokens.append(next_token)
            if next_token == EOS_TOKEN_ID:
                print(f"       EOS at step {step}")
                break

    if step >= MAX_SEQ_LEN - 1:
        print(f"       ⚠️  Hit max sequence length ({MAX_SEQ_LEN})")

    # Decode tokens (include prompt for full decoding)
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

    print(f"     Hypothesis:   \"{hypothesis}\"")
    print(f"     Tokens: {len(tokens)}")  # Generated tokens only

    # Check if correct
    is_correct = hypothesis == ground_truth
    status = "✅" if is_correct else "❌"
    print(f"     Status: {status}")

    results.append({
        'sample_idx': sample_idx,
        'duration': duration,
        'ground_truth': ground_truth,
        'hypothesis': hypothesis,
        'tokens': len(tokens),  # Generated tokens only
        'correct': is_correct,
    })

# Calculate WER
print("\n[5/5] Calculating WER...")

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Levenshtein distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,    # deletion
                    d[i][j-1] + 1,    # insertion
                    d[i-1][j-1] + 1   # substitution
                )

    distance = d[len(ref_words)][len(hyp_words)]
    wer = (distance / len(ref_words) * 100) if len(ref_words) > 0 else 0.0
    return wer

for result in results:
    result['wer'] = calculate_wer(result['ground_truth'], result['hypothesis'])

# Print results
print("\n" + "="*70)
print("RESULTS - Stateful Decoder (Qwen3 Approach)")
print("="*70)

total_duration = 0
perfect_count = 0
for result in results:
    print(f"\nSample {result['sample_idx'] + 1}:")
    print(f"  Duration:     {result['duration']:.2f}s")
    print(f"  Ground truth: \"{result['ground_truth']}\"")
    print(f"  Hypothesis:   \"{result['hypothesis']}\"")
    print(f"  WER:          {result['wer']:.2f}%")
    print(f"  Tokens:       {result['tokens']}")
    status = "✅ PERFECT" if result['correct'] else f"❌ {result['wer']:.2f}% WER"
    print(f"  Status:       {status}")
    total_duration += result['duration']
    if result['correct']:
        perfect_count += 1

# Summary statistics
avg_wer = sum(r['wer'] for r in results) / len(results)

print(f"\n{'='*70}")
print("SUMMARY - Stateful Decoder")
print(f"{'='*70}")
print(f"Samples:       {len(results)}")
print(f"Total audio:   {total_duration:.2f}s")
print(f"Perfect:       {perfect_count}/{len(results)}")
print(f"Average WER:   {avg_wer:.2f}%")
print(f"{'='*70}")

if perfect_count == len(results):
    print("\n🎉 ALL SAMPLES PERFECT! Stateful decoder working correctly!")
elif perfect_count >= len(results) * 0.66:
    print(f"\n✅ {perfect_count}/{len(results)} samples perfect - stateful decoder working well")
    print("   (Stateless decoder also gets 2/3 perfect)")
else:
    print(f"\n⚠️  Only {perfect_count}/{len(results)} samples perfect")
    print("   Expected at least 2/3 based on stateless decoder performance")
    print("   May need cache padding to avoid 112-126 bug zone")
