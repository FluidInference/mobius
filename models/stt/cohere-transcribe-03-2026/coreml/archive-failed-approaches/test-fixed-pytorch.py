#!/usr/bin/env python3
"""Test the fixed wrapper in PyTorch."""

import torch
from transformers import AutoModelForSpeechSeq2Seq
from datasets import load_dataset
import sentencepiece as spm
import numpy as np
import sys
import coremltools as ct

# Import the FIXED wrapper
sys.path.insert(0, str(__file__).replace("test-fixed-pytorch.py", ""))
from cohere_mel_spectrogram import CohereMelSpectrogram
import importlib.util
spec = importlib.util.spec_from_file_location("export_decoder_fixed", "export-decoder-fixed.py")
export_decoder_fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_decoder_fixed)
FixedCachedDecoderWrapper = export_decoder_fixed.FixedCachedDecoderWrapper

print("="*70)
print("Test FIXED PyTorch Wrapper")
print("="*70)

# Configuration
NUM_SAMPLES = 3  # Just 3 samples for quick test
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

# Wrap decoder with FIXED version
print("\n[2/5] Wrapping decoder (FIXED)...")
wrapped_decoder = FixedCachedDecoderWrapper(model, max_seq_len=108)
wrapped_decoder.eval()

# Load CoreML encoder
print("   Loading CoreML encoder...")
coreml_encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)

# Load tokenizer
print("\n[3/5] Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")

# Load samples
print(f"\n[4/5] Loading {NUM_SAMPLES} samples from LibriSpeech...")
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

for sample_idx, sample in enumerate(samples):
    print(f"\n   Sample {sample_idx + 1}/{NUM_SAMPLES}:")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"     Duration: {duration:.2f}s")
    print(f"     Ground truth: \"{ground_truth}\"")

    # Compute mel and encode
    mel = mel_processor(audio)
    mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant', constant_values=0)

    encoder_output = coreml_encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([mel.shape[2]], dtype=np.int32)
    })

    encoder_hidden = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden = torch.from_numpy(value)
            break

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

    print(f"     Hypothesis:   \"{hypothesis}\"")
    print(f"     Tokens:       {len(tokens) - len(PROMPT_IDS)}")

print("\n" + "="*70)
print("QUICK TEST COMPLETE")
print("="*70)
print("Compare with broken version (174% WER with repetitions)")
