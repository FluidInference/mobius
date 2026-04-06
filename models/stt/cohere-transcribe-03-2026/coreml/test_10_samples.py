#!/usr/bin/env python3
"""Quick test of stateful decoder on 10 LibriSpeech samples."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "f16"))

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
from jiwer import wer
import json

print("="*70)
print("Cohere Stateful Decoder - 10 Sample Test")
print("="*70)

# Configuration
NUM_SAMPLES = 10
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 200

# Load models
print("\n[1/4] Loading CoreML models...")
encoder = ct.models.MLModel("f16/cohere_encoder.mlpackage")
decoder = ct.models.MLModel("f16/cohere_decoder_stateful.mlpackage")
print("   ✓ Models loaded")

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

for sample_idx, sample in enumerate(samples):
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
    sample_wer = wer(ground_truth, hypothesis) * 100
    
    print(f"\n   Sample {sample_idx + 1}/{NUM_SAMPLES} ({duration:.1f}s):")
    print(f"     Ground truth: {ground_truth}")
    print(f"     Hypothesis:   {hypothesis}")
    print(f"     WER: {sample_wer:.2f}%")
    
    results.append({
        "duration": duration,
        "ground_truth": ground_truth,
        "hypothesis": hypothesis,
        "wer": sample_wer
    })

# Calculate statistics
print("\n" + "="*70)
print("RESULTS")
print("="*70)

avg_wer = np.mean([r["wer"] for r in results])
perfect_matches = sum(1 for r in results if r["wer"] < 5.0)
perfect_pct = (perfect_matches / len(results)) * 100

print(f"\nAverage WER: {avg_wer:.2f}%")
print(f"Perfect matches (WER < 5%): {perfect_matches}/{len(results)} ({perfect_pct:.1f}%)")
print(f"\nPer-sample WER:")
for i, r in enumerate(results):
    status = "✅" if r["wer"] < 5.0 else "⚠️" if r["wer"] < 20.0 else "❌"
    print(f"  {status} Sample {i+1}: {r['wer']:.2f}% ({r['duration']:.1f}s)")
