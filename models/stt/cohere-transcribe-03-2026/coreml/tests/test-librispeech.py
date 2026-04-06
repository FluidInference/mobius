#!/usr/bin/env python3
"""Test Cohere Transcribe FP16 models on LibriSpeech test-clean.

This script evaluates the FP16 CoreML models on 10 samples from LibriSpeech test-clean
and reports Word Error Rate (WER).
"""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram

print("="*70)
print("Cohere Transcribe - LibriSpeech Test-Clean WER Test")
print("="*70)

# Configuration
NUM_SAMPLES = 10
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 200

# Load LibriSpeech test-clean
print(f"\n[1/6] Loading {NUM_SAMPLES} samples from LibriSpeech test-clean...")
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

# Load models
print("\n[2/6] Loading CoreML models...")
try:
    encoder = ct.models.MLModel(
        "build/cohere_encoder.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    decoder = ct.models.MLModel(
        "build/cohere_decoder_cached.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print(f"   ✓ Models loaded (FP16)")
except Exception as e:
    print(f"   ❌ Error loading models: {e}")
    exit(1)

# Load tokenizer
print("\n[3/6] Loading tokenizer...")
try:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("../tokenizer.model")
    print(f"   ✓ Tokenizer loaded")
except Exception as e:
    print(f"   ❌ Error loading tokenizer: {e}")
    exit(1)

# Process samples
print(f"\n[4/6] Processing {NUM_SAMPLES} samples...")
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
        ((0, 0), (0, 0), (0, 3001 - mel.shape[2])),
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
    
    # Decode with 10-token prompt
    tokens = list(PROMPT_IDS)
    cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
    cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)
    
    # Process prompt tokens
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
        
        for key, value in decoder_output.items():
            if hasattr(value, 'shape') and len(value.shape) == 4:
                if 'k' in key.lower() or key == 'new_cache_k':
                    cache_k = value
                else:
                    cache_v = value
    
    # Generate new tokens
    for step in range(len(PROMPT_IDS), len(PROMPT_IDS) + MAX_NEW_TOKENS):
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
    print(f"     Tokens: {len(tokens) - len(PROMPT_IDS)}")
    
    results.append({
        'sample_idx': sample_idx,
        'duration': duration,
        'ground_truth': ground_truth,
        'hypothesis': hypothesis,
        'tokens': len(tokens) - len(PROMPT_IDS)
    })

# Calculate WER
print("\n[5/6] Calculating WER...")

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
print("\n[6/6] Results:")
print("\n" + "="*70)

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
print("SUMMARY")
print(f"{'='*70}")
print(f"Samples:      {len(results)}")
print(f"Total audio:  {total_duration:.2f}s")
print(f"Average WER:  {avg_wer:.2f}%")
print(f"Median WER:   {median_wer:.2f}%")
print(f"Min WER:      {min_wer:.2f}%")
print(f"Max WER:      {max_wer:.2f}%")
print(f"{'='*70}")
