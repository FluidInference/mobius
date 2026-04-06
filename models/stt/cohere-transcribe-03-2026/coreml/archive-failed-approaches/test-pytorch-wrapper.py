#!/usr/bin/env python3
"""Test if the wrapper has repetition issues in PyTorch before CoreML conversion."""

import torch
from transformers import AutoModelForSpeechSeq2Seq
from datasets import load_dataset
import sentencepiece as spm
import numpy as np
import sys

# Import the wrapper and CoreML
sys.path.insert(0, str(__file__).replace("test-pytorch-wrapper.py", ""))
from cohere_mel_spectrogram import CohereMelSpectrogram
import importlib.util
import coremltools as ct

spec = importlib.util.spec_from_file_location("export_decoder", "export-decoder-cached.py")
export_decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_decoder)
MaskedCachedDecoderWrapper = export_decoder.MaskedCachedDecoderWrapper

print("="*70)
print("PyTorch Wrapper Test - DynamicCache")
print("="*70)

# Configuration
NUM_SAMPLES = 10
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 200

# Load model
print("\n[1/5] Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()

# Wrap decoder
print("\n[2/5] Wrapping decoder...")
wrapped_decoder = MaskedCachedDecoderWrapper(model, max_seq_len=108)
wrapped_decoder.eval()

# Load CoreML encoder (we know this works)
print("   Loading CoreML encoder...")
coreml_encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)

# Load tokenizer
print("\n[3/5] Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")

# Load LibriSpeech
print(f"\n[4/5] Loading {NUM_SAMPLES} samples from LibriSpeech test-clean...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []
for i, sample in enumerate(dataset):
    if i >= NUM_SAMPLES:
        break
    samples.append(sample)
print(f"   ✓ Loaded {len(samples)} samples")

# Process samples
print(f"\n[5/5] Testing {NUM_SAMPLES} samples...")
mel_processor = CohereMelSpectrogram()

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

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
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + 1)

    distance = d[len(ref_words)][len(hyp_words)]
    wer = (distance / len(ref_words) * 100) if len(ref_words) > 0 else 0.0
    return wer

results = []

for sample_idx, sample in enumerate(samples):
    print(f"\n   Sample {sample_idx + 1}/{NUM_SAMPLES}:")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"     Duration: {duration:.2f}s")
    print(f"     Ground truth: \"{ground_truth}\"")

    # Compute mel spectrogram
    mel = mel_processor(audio)  # Returns (1, 128, T)
    mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant', constant_values=0)

    # Encode with CoreML encoder
    encoder_output = coreml_encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([mel.shape[2]], dtype=np.int32)
    })

    # Extract encoder hidden states and convert to PyTorch
    encoder_hidden = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden = torch.from_numpy(value)
            break

    if sample_idx == 0:
        print(f"     Encoder hidden shape: {encoder_hidden.shape}")

    # Initialize cache
    cache_k = torch.zeros(8, 8, 108, 128)
    cache_v = torch.zeros(8, 8, 108, 128)
    cross_attention_mask = torch.ones(1, 1, 1, encoder_hidden.shape[1])

    # Process prompt tokens
    tokens = list(PROMPT_IDS)
    for step, token_id in enumerate(PROMPT_IDS):
        input_id = torch.tensor([[token_id]], dtype=torch.long)
        step_tensor = torch.tensor([step], dtype=torch.int32)

        with torch.no_grad():
            logits, cache_k, cache_v = wrapped_decoder(
                input_id, encoder_hidden, cache_k, cache_v, step_tensor, cross_attention_mask
            )

    # Generate new tokens
    for step in range(len(PROMPT_IDS), len(PROMPT_IDS) + MAX_NEW_TOKENS):
        input_id = torch.tensor([[tokens[-1]]], dtype=torch.long)
        step_tensor = torch.tensor([step], dtype=torch.int32)

        with torch.no_grad():
            logits, cache_k, cache_v = wrapped_decoder(
                input_id, encoder_hidden, cache_k, cache_v, step_tensor, cross_attention_mask
            )

        next_token = int(torch.argmax(logits[0]).item())
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

    wer = calculate_wer(ground_truth, hypothesis)

    print(f"     Hypothesis:   \"{hypothesis}\"")
    print(f"     WER:          {wer:.2f}%")
    print(f"     Tokens:       {len(tokens) - len(PROMPT_IDS)}")

    results.append({
        'sample_idx': sample_idx,
        'duration': duration,
        'ground_truth': ground_truth,
        'hypothesis': hypothesis,
        'wer': wer,
        'tokens': len(tokens) - len(PROMPT_IDS)
    })

# Summary
print("\n" + "="*70)
print("RESULTS - PyTorch Wrapper (Before CoreML)")
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

avg_wer = sum(r['wer'] for r in results) / len(results)
median_wer = sorted(r['wer'] for r in results)[len(results) // 2]
min_wer = min(r['wer'] for r in results)
max_wer = max(r['wer'] for r in results)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Samples:       {len(results)}")
print(f"Total audio:   {total_duration:.2f}s")
print(f"Average WER:   {avg_wer:.2f}%")
print(f"Median WER:    {median_wer:.2f}%")
print(f"Min WER:       {min_wer:.2f}%")
print(f"Max WER:       {max_wer:.2f}%")
print(f"{'='*70}")

if avg_wer < 10:
    print("\n✅ PyTorch wrapper works perfectly!")
elif avg_wer < 50:
    print(f"\n⚠️  PyTorch wrapper has issues: {avg_wer:.2f}% WER")
else:
    print(f"\n❌ PyTorch wrapper is broken: {avg_wer:.2f}% WER")
print("    (CoreML WER was 174.03%)")
