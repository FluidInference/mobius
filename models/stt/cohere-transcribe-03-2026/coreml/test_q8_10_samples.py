#!/usr/bin/env python3
"""Test Q8 stateful decoder with punctuation-normalized WER on 10 samples from LibriSpeech test-clean."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "f16"))

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
from jiwer import wer
import json
import re
import time

def normalize_text(text):
    """Remove punctuation and normalize case."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

print("="*70)
print("Cohere Q8 Stateful Decoder - 10 Sample Test (Normalized WER)")
print("="*70)

# Configuration
NUM_SAMPLES = 10
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 200

# Load Q8 models
print("\n[1/4] Loading Q8 CoreML models...")
encoder = ct.models.MLModel("q8/cohere_encoder.mlpackage")
decoder = ct.models.MLModel("q8/cohere_decoder_stateful.mlpackage")
print("   ✓ Q8 models loaded")

# Load vocab
print("\n[2/4] Loading vocabulary...")
with open("f16/vocab.json") as f:
    vocab = {int(k): v for k, v in json.load(f).items()}
print("   ✓ Vocabulary loaded")

# Load LibriSpeech
print(f"\n[3/4] Loading {NUM_SAMPLES} samples from LibriSpeech test-clean...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []
for i, sample in enumerate(dataset):
    if i >= NUM_SAMPLES:
        break
    samples.append(sample)
print(f"   ✓ Loaded {len(samples)} samples")

# Process samples
print(f"\n[4/4] Transcribing {NUM_SAMPLES} samples...")
mel_processor = CohereMelSpectrogram()
results = []
start_time = time.time()

for sample_idx, sample in enumerate(samples):
    sample_start = time.time()

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    # Compute mel spectrogram
    mel = mel_processor(audio)
    mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])))

    # Encode
    encoder_output = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([mel.shape[2]], dtype=np.int32)
    })
    encoder_hidden = encoder_output["hidden_states"]

    # Decode with stateful decoder
    state = decoder.make_state()
    tokens = []

    for step in range(MAX_NEW_TOKENS):
        current_token = PROMPT_IDS[step] if step < len(PROMPT_IDS) else tokens[-1]

        decoder_output = decoder.predict({
            "input_id": np.array([[current_token]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden.astype(np.float16),
            "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float16),
            "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16),
            "position_ids": np.array([[step]], dtype=np.int32),
        }, state=state)

        next_token = int(np.argmax(decoder_output["logits"][0]))
        tokens.append(next_token)

        if next_token == EOS_TOKEN_ID:
            break

    # Decode tokens to text
    text_tokens = []
    for token_id in tokens:
        if token_id <= 4 or token_id == EOS_TOKEN_ID:
            continue
        token_str = vocab.get(token_id, "")
        if token_str.startswith("<|"):
            continue
        text_tokens.append(token_str)

    hypothesis = "".join(text_tokens).replace("▁", " ").strip()

    # Normalize both texts
    ground_truth_norm = normalize_text(ground_truth)
    hypothesis_norm = normalize_text(hypothesis)

    # Calculate WER on normalized text
    sample_wer = wer(ground_truth_norm, hypothesis_norm) * 100

    sample_time = time.time() - sample_start

    print(f"   Sample {sample_idx + 1}/{NUM_SAMPLES}: WER={sample_wer:.2f}% ({sample_time:.2f}s)")

    results.append({
        "duration": duration,
        "ground_truth": ground_truth,
        "hypothesis": hypothesis,
        "wer": sample_wer,
        "processing_time": sample_time,
    })

total_time = time.time() - start_time

# Calculate statistics
print("\n" + "="*70)
print("RESULTS (10 Samples, Q8 Models, Punctuation-Normalized WER)")
print("="*70)

avg_wer = np.mean([r["wer"] for r in results])
median_wer = np.median([r["wer"] for r in results])
perfect_matches = sum(1 for r in results if r["wer"] < 5.0)
good_matches = sum(1 for r in results if r["wer"] < 20.0)
perfect_pct = (perfect_matches / len(results)) * 100
good_pct = (good_matches / len(results)) * 100

print(f"\n📊 Quality Metrics:")
print(f"   Average WER:         {avg_wer:.2f}%")
print(f"   Median WER:          {median_wer:.2f}%")
print(f"   Perfect (WER < 5%):  {perfect_matches}/{len(results)} ({perfect_pct:.1f}%)")
print(f"   Good (WER < 20%):    {good_matches}/{len(results)} ({good_pct:.1f}%)")

print(f"\n⚡ Performance Metrics:")
avg_proc_time = np.mean([r["processing_time"] for r in results])
avg_audio_duration = np.mean([r["duration"] for r in results])
avg_rtfx = avg_proc_time / avg_audio_duration if avg_audio_duration > 0 else 0
print(f"   Avg processing time: {avg_proc_time:.2f}s")
print(f"   Avg audio duration:  {avg_audio_duration:.2f}s")
print(f"   Avg RTFx:            {avg_rtfx:.2f}x")
print(f"   Total time:          {total_time:.1f}s")

# Show sample outputs
print(f"\n📝 Sample outputs:")
for i in range(min(3, len(results))):
    r = results[i]
    print(f"\n   Sample {i+1} (WER: {r['wer']:.2f}%):")
    print(f"   GT:  {r['ground_truth'][:80]}...")
    print(f"   Hyp: {r['hypothesis'][:80]}...")

# Save results to JSON
output_file = "test_q8_10_samples_results.json"
with open(output_file, "w") as f:
    json.dump({
        "quantization": "int8",
        "num_samples": len(results),
        "avg_wer": avg_wer,
        "median_wer": median_wer,
        "perfect_matches": perfect_matches,
        "perfect_pct": perfect_pct,
        "good_matches": good_matches,
        "good_pct": good_pct,
        "avg_rtfx": avg_rtfx,
        "total_time": total_time,
        "results": results,
    }, f, indent=2)
print(f"\n💾 Saved detailed results to: {output_file}")
print()
