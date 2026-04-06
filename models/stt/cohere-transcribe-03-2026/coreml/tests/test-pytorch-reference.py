#!/usr/bin/env python3
"""Benchmark original Cohere PyTorch model on LibriSpeech test-clean.

This establishes the gold standard baseline for the model's expected performance.
Compare CoreML results against this to identify conversion issues.
"""

import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("="*70)
print("Cohere Transcribe PyTorch Reference - LibriSpeech Test-Clean")
print("="*70)

# Configuration
NUM_SAMPLES = 10
MAX_NEW_TOKENS = 200

# Load LibriSpeech test-clean
print(f"\n[1/5] Loading {NUM_SAMPLES} samples from LibriSpeech test-clean...")
try:
    from datasets import load_dataset
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    samples = []
    for i, sample in enumerate(dataset):
        if i >= NUM_SAMPLES:
            break
        samples.append(sample)
    print(f"   ✓ Loaded {len(samples)} samples")
except Exception as e:
    print(f"   ❌ Error loading LibriSpeech: {e}")
    exit(1)

# Load PyTorch model and processor
print("\n[2/5] Loading PyTorch model...")
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
    print(f"   ✓ Model loaded (FP16, PyTorch)")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    exit(1)

# Load tokenizer
print("\n[3/5] Loading tokenizer...")
try:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("../tokenizer.model")
    print(f"   ✓ Tokenizer loaded")
except Exception as e:
    print(f"   ❌ Error loading tokenizer: {e}")
    exit(1)

# Process samples
print(f"\n[4/5] Processing {NUM_SAMPLES} samples...")
results = []

for sample_idx, sample in enumerate(samples):
    print(f"\n   Sample {sample_idx + 1}/{NUM_SAMPLES}:")

    audio = sample['audio']['array']
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"     Duration: {duration:.2f}s")
    print(f"     Ground truth: \"{ground_truth}\"")

    # Process audio
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # Decode
    hypothesis = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    hypothesis = hypothesis.strip().lower()

    print(f"     Hypothesis:   \"{hypothesis}\"")
    print(f"     Tokens:       {len(generated_ids[0])}")

    results.append({
        'sample_idx': sample_idx,
        'duration': duration,
        'ground_truth': ground_truth,
        'hypothesis': hypothesis,
        'tokens': len(generated_ids[0])
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
print("RESULTS")
print("="*70)

total_duration = 0
for result in results:
    print(f"\nSample {result['sample_idx'] + 1}:")
    print(f"  Duration:     {result['duration']:.2f}s")
    print(f"  Ground truth: \"{result['ground_truth']}\"")
    print(f"  Hypothesis:   \"{result['hypothesis']}\"")
    print(f"  WER:          {result['wer']:.2f}%")
    print(f"  Tokens:       {result['tokens']}")
    total_duration += result['duration']

# Summary statistics
avg_wer = sum(r['wer'] for r in results) / len(results)
median_wer = sorted(r['wer'] for r in results)[len(results) // 2]
min_wer = min(r['wer'] for r in results)
max_wer = max(r['wer'] for r in results)

print(f"\n{'='*70}")
print("SUMMARY - PyTorch Reference (Gold Standard)")
print(f"{'='*70}")
print(f"Samples:      {len(results)}")
print(f"Total audio:  {total_duration:.2f}s")
print(f"Average WER:  {avg_wer:.2f}%")
print(f"Median WER:   {median_wer:.2f}%")
print(f"Min WER:      {min_wer:.2f}%")
print(f"Max WER:      {max_wer:.2f}%")
print(f"{'='*70}")
print("\nUse this as the baseline to compare against CoreML conversions.")
print("Any significant WER increase indicates a conversion issue.")
