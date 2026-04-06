#!/usr/bin/env python3
"""Debug the PyTorch wrapper to see cache and attention mask behavior."""

import torch
from transformers import AutoModelForSpeechSeq2Seq
from datasets import load_dataset
import sentencepiece as spm
import numpy as np
import sys
import coremltools as ct

# Import the wrapper
sys.path.insert(0, str(__file__).replace("debug-pytorch-wrapper.py", ""))
from cohere_mel_spectrogram import CohereMelSpectrogram
import importlib.util
spec = importlib.util.spec_from_file_location("export_decoder", "export-decoder-cached.py")
export_decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_decoder)
MaskedCachedDecoderWrapper = export_decoder.MaskedCachedDecoderWrapper

print("="*70)
print("Debug PyTorch Wrapper - Cache & Attention Mask Analysis")
print("="*70)

# Configuration
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3
MAX_NEW_TOKENS = 30  # Just 30 tokens to see the pattern

# Load model
print("\n[1/4] Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()

# Wrap decoder
print("\n[2/4] Wrapping decoder...")
wrapped_decoder = MaskedCachedDecoderWrapper(model, max_seq_len=108)
wrapped_decoder.eval()

# Load CoreML encoder
print("   Loading CoreML encoder...")
coreml_encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)

# Load tokenizer
print("\n[3/4] Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")

# Load one sample
print("\n[4/4] Loading sample from LibriSpeech...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
sample = next(iter(dataset))
audio = sample['audio']['array'].astype(np.float32)
ground_truth = sample['text'].lower()

print(f"\n   Sample: \"{ground_truth}\"")
print(f"   Duration: {len(audio) / 16000.0:.2f}s")

# Compute mel and encode
mel_processor = CohereMelSpectrogram()
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

print(f"   Encoder hidden: {encoder_hidden.shape}")

# Initialize cache
cache_k = torch.zeros(8, 8, 108, 128)
cache_v = torch.zeros(8, 8, 108, 128)
cross_attention_mask = torch.ones(1, 1, 1, encoder_hidden.shape[1])

print("\n" + "="*70)
print("DECODING WITH DEBUG INFO")
print("="*70)

tokens = list(PROMPT_IDS)

# Process prompt tokens (show first 3)
for step, token_id in enumerate(PROMPT_IDS[:3]):
    print(f"\n--- Prompt Step {step} (token={token_id}) ---")

    input_id = torch.tensor([[token_id]], dtype=torch.long)
    step_tensor = torch.tensor([step], dtype=torch.int32)

    # Check cache before
    cache_k_norm = torch.sqrt(torch.sum(cache_k**2, dim=(1, 2, 3)))  # (8,) - norm per layer
    num_nonzero = torch.sum(cache_k_norm > 1e-6).item()
    print(f"  Cache before: {num_nonzero}/8 layers non-zero")

    with torch.no_grad():
        logits, cache_k, cache_v = wrapped_decoder(
            input_id, encoder_hidden, cache_k, cache_v, step_tensor, cross_attention_mask
        )

    # Check cache after
    cache_k_norm = torch.sqrt(torch.sum(cache_k**2, dim=(1, 2, 3)))
    num_nonzero = torch.sum(cache_k_norm > 1e-6).item()
    cache_k_pos_norms = torch.sqrt(torch.sum(cache_k**2, dim=(0, 1, 3)))  # (108,) - norm per position
    nonzero_positions = torch.where(cache_k_pos_norms > 1e-6)[0].tolist()

    print(f"  Cache after:  {num_nonzero}/8 layers non-zero")
    print(f"  Non-zero positions: {nonzero_positions[:10]}{'...' if len(nonzero_positions) > 10 else ''}")

    next_token = int(torch.argmax(logits[0]).item())
    next_token_str = sp.IdToPiece(next_token)
    print(f"  Next token: {next_token} ({next_token_str})")

# Generate tokens (show first 15)
print(f"\n{'='*70}")
print("GENERATING NEW TOKENS")
print(f"{'='*70}")

for gen_step in range(15):
    step = len(PROMPT_IDS) + gen_step
    print(f"\n--- Generation Step {gen_step} (overall step={step}) ---")

    input_id = torch.tensor([[tokens[-1]]], dtype=torch.long)
    step_tensor = torch.tensor([step], dtype=torch.int32)

    # Check cache growth
    cache_k_pos_norms = torch.sqrt(torch.sum(cache_k**2, dim=(0, 1, 3)))
    nonzero_positions = torch.where(cache_k_pos_norms > 1e-6)[0].tolist()
    print(f"  Cache positions: {len(nonzero_positions)} non-zero: {nonzero_positions[:15]}{'...' if len(nonzero_positions) > 15 else ''}")

    with torch.no_grad():
        logits, cache_k, cache_v = wrapped_decoder(
            input_id, encoder_hidden, cache_k, cache_v, step_tensor, cross_attention_mask
        )

    # Check if cache grew
    cache_k_pos_norms_after = torch.sqrt(torch.sum(cache_k**2, dim=(0, 1, 3)))
    nonzero_positions_after = torch.where(cache_k_pos_norms_after > 1e-6)[0].tolist()

    if len(nonzero_positions_after) != len(nonzero_positions) + 1:
        print(f"  ⚠️  CACHE DID NOT GROW! Still {len(nonzero_positions_after)} positions")
        print(f"      Expected {len(nonzero_positions) + 1}, got {len(nonzero_positions_after)}")
    else:
        print(f"  ✓ Cache grew to {len(nonzero_positions_after)} positions")

    next_token = int(torch.argmax(logits[0]).item())
    next_token_str = sp.IdToPiece(next_token)
    tokens.append(next_token)

    print(f"  Generated: {next_token} ({next_token_str})")

    # Decode current sequence
    partial_text = sp.DecodeIds(tokens)
    for special in ['<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>',
                    '<|it|>', '<|pnc|>', '<|nopnc|>', '<|itn|>', '<|noitn|>',
                    '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
                    '<|endoftext|>', '<|en|>']:
        partial_text = partial_text.replace(special, '')
    partial_text = partial_text.strip()
    print(f"  Current text: \"{partial_text}\"")

    if next_token == EOS_TOKEN_ID:
        print("\n  🛑 EOS token generated")
        break

print("\n" + "="*70)
print("FINAL RESULT")
print("="*70)
print(f"Ground truth: \"{ground_truth}\"")
print(f"Hypothesis:   \"{partial_text}\"")
print(f"Tokens:       {len(tokens) - len(PROMPT_IDS)}")
