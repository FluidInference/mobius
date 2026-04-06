#!/usr/bin/env python3
"""Detailed test on 10-second samples to see actual outputs."""

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
print("Detailed Test: 10-second Audio Samples")
print("="*70)

PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 200
MAX_SEQ_LEN = 256

# Load models
print("\n[1/4] Loading models...")
encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
stateful_decoder = ct.models.MLModel("build/cohere_decoder_stateful_256.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")
print("   ✓ Models loaded")

# Find 10-second samples
print("\n[2/4] Finding 3 samples around 10 seconds...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []

for sample in dataset:
    duration = len(sample['audio']['array']) / 16000.0
    if 9.5 <= duration <= 10.5:
        samples.append(sample)
        if len(samples) >= 3:
            break

print(f"   ✓ Found {len(samples)} samples")

# Process
mel_processor = CohereMelSpectrogram()

for sample_idx, sample in enumerate(samples):
    print(f"\n{'='*70}")
    print(f"Sample {sample_idx + 1}/3")
    print(f"{'='*70}")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"Duration: {duration:.2f}s")

    # Encode
    mel = mel_processor(audio)
    mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])), mode='constant', constant_values=0)

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

    for step in range(min(MAX_NEW_TOKENS + len(PROMPT_IDS), MAX_SEQ_LEN)):
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
        next_token = int(np.argmax(decoder_output["logits"][0]))
        last_token = next_token

        if step >= len(PROMPT_IDS) - 1:
            tokens.append(next_token)
            if next_token == EOS_TOKEN_ID:
                break

    # Decode
    all_tokens = list(PROMPT_IDS) + tokens
    hypothesis = sp.DecodeIds(all_tokens)

    # Clean
    for special in ['<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
                    '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
                    '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
                    '<|endoftext|>', '<|en|>']:
        hypothesis = hypothesis.replace(special, '')
    hypothesis = hypothesis.strip().lower()

    # Compare
    gt_clean = re.sub(r'[^\w\s]', '', ground_truth).strip()
    hyp_clean = re.sub(r'[^\w\s]', '', hypothesis).strip()

    print(f"\nGround Truth ({len(ground_truth)} chars):")
    print(f'  "{ground_truth}"')
    print(f"\nHypothesis ({len(hypothesis)} chars):")
    print(f'  "{hypothesis}"')
    print(f"\nGround Truth (clean, {len(gt_clean)} chars):")
    print(f'  "{gt_clean}"')
    print(f"\nHypothesis (clean, {len(hyp_clean)} chars):")
    print(f'  "{hyp_clean}"')

    if gt_clean == hyp_clean:
        print(f"\n✅ PERFECT MATCH (ignoring punctuation)")
    else:
        print(f"\n❌ DIFFERENT")
        # Show first difference
        gt_words = gt_clean.split()
        hyp_words = hyp_clean.split()
        print(f"\nGT words ({len(gt_words)}):  {' '.join(gt_words[:20])}...")
        print(f"Hyp words ({len(hyp_words)}): {' '.join(hyp_words[:20])}...")

        # Find first diff
        for i, (gw, hw) in enumerate(zip(gt_words, hyp_words)):
            if gw != hw:
                print(f"\nFirst difference at word {i}: '{gw}' vs '{hw}'")
                break
        if len(gt_words) != len(hyp_words):
            print(f"Length difference: {len(gt_words)} vs {len(hyp_words)} words")

print(f"\n{'='*70}")
