#!/usr/bin/env python3
"""Test PyTorch Cohere model end-to-end on long audio - simple version."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datasets import load_dataset
from cohere_mel_spectrogram import CohereMelSpectrogram
import torch
import sentencepiece as spm

print("="*70)
print("Test: PyTorch Cohere Model on Long Audio")
print("="*70)

# Load model and tokenizer
print("\n[1/3] Loading model...")
try:
    from transformers import AutoModelForSpeechSeq2Seq
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.Load("../tokenizer.model")

    mel_processor = CohereMelSpectrogram()
    print("   ✓ Model loaded")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Prompt for English transcription
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3

# Get long audio samples
print("\n[2/3] Finding long audio samples (20-23s)...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
samples = []

for sample in dataset:
    duration = len(sample['audio']['array']) / 16000.0
    if 20.0 <= duration <= 23.5:
        samples.append(sample)
        print(f"   Found sample {len(samples)}: {duration:.2f}s")
        if len(samples) >= 3:
            break

# Test each sample
print(f"\n[3/3] Testing {len(samples)} samples...")

for idx, sample in enumerate(samples):
    print(f"\n{'='*70}")
    print(f"Sample {idx + 1}/{len(samples)}")
    print(f"{'='*70}")

    audio = sample['audio']['array'].astype(np.float32)
    ground_truth = sample['text'].lower()
    duration = len(audio) / 16000.0

    print(f"\nDuration: {duration:.2f}s")
    print(f"Ground truth: \"{ground_truth[:80]}...\"")

    # Process mel
    mel = mel_processor(audio)
    if mel.shape[2] > 3001:
        mel_padded = mel[:, :, :3001]
        actual_frames = 3001
    else:
        mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant', constant_values=0)
        actual_frames = mel.shape[2]

    # Transcribe with PyTorch (manual generation)
    with torch.no_grad():
        mel_torch = torch.from_numpy(mel_padded).float()
        feature_length_torch = torch.tensor([actual_frames], dtype=torch.long)

        # Encode
        encoder_output = model.encoder(
            input_features=mel_torch,
            feature_length=feature_length_torch
        )
        if hasattr(encoder_output, 'last_hidden_state'):
            encoder_raw = encoder_output.last_hidden_state
        else:
            encoder_raw = encoder_output[0]

        # Apply projection to check quality
        encoder_hidden = model.encoder_decoder_proj(encoder_raw)
        print(f"Encoder: mean={encoder_hidden.mean():.6f}, std={encoder_hidden.std():.6f}, max={encoder_hidden.max():.6f}")

        # Check if encoder output is weak
        if encoder_hidden.std() < 0.4:
            print(f"⚠️  Encoder output is WEAK (std < 0.4)")

        # Manual decoding using decoder directly
        decoder_input_ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
        positions = torch.arange(len(PROMPT_IDS), dtype=torch.long).unsqueeze(0)
        tokens = list(PROMPT_IDS)

        for step in range(200):
            # Forward through decoder (use projected encoder output)
            decoder_output = model.transf_decoder(
                input_ids=decoder_input_ids,
                positions=positions,
                encoder_hidden_states=encoder_hidden,  # Already projected
            )

            # Get hidden states and apply LM head
            hidden_states = decoder_output[0]
            logits = model.log_softmax.mlp.layer0(hidden_states)  # Apply LM head

            # Get next token
            next_token_logits = logits[0, -1, :]
            next_token = int(torch.argmax(next_token_logits))

            tokens.append(next_token)
            if next_token == EOS_TOKEN_ID:
                break

            # Append for next iteration
            decoder_input_ids = torch.cat([
                decoder_input_ids,
                torch.tensor([[next_token]], dtype=torch.long)
            ], dim=1)
            positions = torch.arange(decoder_input_ids.shape[1], dtype=torch.long).unsqueeze(0)

        generated = [torch.tensor(tokens)]

        # Decode
        hypothesis = sp.DecodeIds(generated[0].tolist())
        for special in ['<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
                        '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
                        '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
                        '<|endoftext|>', '<|en|>']:
            hypothesis = hypothesis.replace(special, '')
        hypothesis = hypothesis.strip().lower()

    print(f"\nPyTorch output ({len(generated[0])} tokens): \"{hypothesis[:100]}...\"")

    # Check quality
    gt_start = ground_truth[:50].replace(".", "").replace(",", "").strip()
    hyp_start = hypothesis[:50].replace(".", "").replace(",", "").strip()

    matches = gt_start in hyp_start or hyp_start in gt_start

    if matches:
        print(f"\n✅ CORRECT transcription")
    else:
        print(f"\n❌ INCORRECT transcription")
        print(f"   Expected: \"{gt_start}...\"")
        print(f"   Got: \"{hyp_start}...\"")

print(f"\n{'='*70}")
