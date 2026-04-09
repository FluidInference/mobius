#!/usr/bin/env python3
"""Trace a single forward pass through the Cohere model to understand architecture."""

import torch
from transformers import AutoModelForSpeechSeq2Seq
import numpy as np

print("="*80)
print("EXPERIMENT 1: Trace PyTorch Forward Pass")
print("="*80)

# Load model
print("\n[1/6] Loading PyTorch model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()
print("✓ Model loaded")

# Print model structure
print("\n[2/6] Model architecture:")
print(model)

# Dummy input
print("\n[3/6] Creating test inputs...")
mel = torch.randn(1, 128, 100)  # [batch, n_mels, frames]
length = torch.tensor([100])
print(f"Mel spectrogram shape: {mel.shape}")
print(f"Length: {length}")

# Step 1: Encoder
print("\n[4/6] Running encoder...")
with torch.no_grad():
    encoder_out = model.encoder(mel, length)

# Encoder returns tuple: (hidden_states, length)
if isinstance(encoder_out, tuple):
    encoder_hidden, encoder_length = encoder_out
    print(f"Encoder output type: tuple (hidden_states, length)")
    print(f"Encoder hidden states shape: {encoder_hidden.shape}")
    print(f"Encoder output length: {encoder_length}")
else:
    encoder_hidden = encoder_out.last_hidden_state
    print(f"Encoder output type: {type(encoder_out)}")
    print(f"Encoder hidden states shape: {encoder_hidden.shape}")

print(f"Encoder hidden states sample (first 2 tokens, first 5 dims):")
print(encoder_hidden[0, :2, :5])

# Check encoder-decoder projection
print("\n[5/6] Checking encoder-decoder projection...")
if model.encoder_decoder_proj is not None:
    print(f"✓ Encoder-decoder projection exists")
    print(f"  Type: {type(model.encoder_decoder_proj)}")
    with torch.no_grad():
        projected = model.encoder_decoder_proj(encoder_hidden)
    print(f"  Projected shape: {projected.shape}")
    print(f"  Input dim: 1280, Output dim: 1024")
else:
    print("✗ No encoder-decoder projection")
    projected = encoder_hidden

# Step 2: Language token embeddings
print("\n[6/6] Analyzing language token embeddings...")

# Language tokens to test
languages = {
    "English": 62,
    "French": 69,
    "Spanish": 169,
    "Chinese": 50,
}

embeddings = {}
for lang_name, token_id in languages.items():
    with torch.no_grad():
        emb = model.transf_decoder._embedding.token_embedding(
            torch.tensor([[token_id]])
        )
    embeddings[lang_name] = emb[0, 0]
    print(f"\n{lang_name} (token {token_id}):")
    print(f"  Embedding shape: {emb.shape}")
    print(f"  First 5 dims: {emb[0, 0, :5]}")
    print(f"  Norm: {torch.norm(emb[0, 0]).item():.4f}")

# Compute pairwise similarities
print("\n" + "="*80)
print("Language Embedding Similarities")
print("="*80)

from torch.nn.functional import cosine_similarity

lang_list = list(languages.keys())
print("\nCosine similarity matrix:")
print(f"{'':10s}", end="")
for lang in lang_list:
    print(f"{lang:10s}", end="")
print()

for i, lang1 in enumerate(lang_list):
    print(f"{lang1:10s}", end="")
    for j, lang2 in enumerate(lang_list):
        if i == j:
            print(f"{'1.0000':>10s}", end="")
        else:
            sim = cosine_similarity(
                embeddings[lang1].unsqueeze(0),
                embeddings[lang2].unsqueeze(0),
                dim=1
            ).item()
            print(f"{sim:>10.4f}", end="")
    print()

# Compare to non-language tokens
print("\n" + "="*80)
print("Language vs Non-Language Token Embeddings")
print("="*80)

non_lang_tokens = {
    "START": 4,
    "END": 5,
    "word_boundary": 13764,
    "start_of_context": 7,
}

for name, token_id in non_lang_tokens.items():
    with torch.no_grad():
        emb = model.transf_decoder._embedding.token_embedding(
            torch.tensor([[token_id]])
        )

    print(f"\n{name} (token {token_id}):")
    print(f"  First 5 dims: {emb[0, 0, :5]}")
    print(f"  Norm: {torch.norm(emb[0, 0]).item():.4f}")

    # Compare to English
    sim = cosine_similarity(
        emb[0, 0].unsqueeze(0),
        embeddings["English"].unsqueeze(0),
        dim=1
    ).item()
    print(f"  Similarity to English token: {sim:.4f}")

# Test full prompt sequence
print("\n" + "="*80)
print("Testing Full Prompt Sequence")
print("="*80)

language_token = 62  # English
prompt = [13764, 7, 4, 16, language_token, language_token, 5, 9, 11, 13]
print(f"\nPrompt tokens: {prompt}")

with torch.no_grad():
    decoder_input_ids = torch.tensor([prompt])
    embeddings_seq = model.transf_decoder._embedding.token_embedding(decoder_input_ids)

print(f"Embeddings shape: {embeddings_seq.shape}")
print(f"\nEmbeddings per token (first 3 dims):")
for i, token in enumerate(prompt):
    print(f"  Position {i}, Token {token}: {embeddings_seq[0, i, :3]}")

# Check if language tokens (positions 4 and 5) have high similarity
with torch.no_grad():
    pos_4_emb = embeddings_seq[0, 4]
    pos_5_emb = embeddings_seq[0, 5]

    sim = cosine_similarity(
        pos_4_emb.unsqueeze(0),
        pos_5_emb.unsqueeze(0),
        dim=1
    ).item()

print(f"\nSimilarity between duplicate language tokens (pos 4 vs 5): {sim:.4f}")

print("\n" + "="*80)
print("EXPERIMENT 1 COMPLETE")
print("="*80)
