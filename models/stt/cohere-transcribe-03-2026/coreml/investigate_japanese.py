#!/usr/bin/env python3
"""Investigate Japanese tokenization issue."""
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

model_id = "CohereLabs/cohere-transcribe-03-2026"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print("=" * 80)
print("JAPANESE TOKENIZATION INVESTIGATION")
print("=" * 80)

# Check tokenizer vocabulary
tokenizer = processor.tokenizer

# Test Japanese text
test_texts = [
    "こんにちは",  # Hello
    "ありがとう",  # Thank you
    "日本語",      # Japanese language
    "hello world",  # English for comparison
]

print("\nTokenizer type:", type(tokenizer))
print("Vocab size:", tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else "Unknown")

print("\nTest tokenization:")
for text in test_texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"\nOriginal: {text}")
    print(f"Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")  # First 20 tokens
    print(f"Decoded: {decoded}")
    print(f"Match: {text.lower().strip() == decoded.lower().strip()}")

# Check model generation config
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    trust_remote_code=True
)

print("\n" + "=" * 80)
print("MODEL GENERATION CONFIG")
print("=" * 80)
print(f"Decoder start token ID: {model.generation_config.decoder_start_token_id}")
print(f"Forced decoder IDs: {model.generation_config.forced_decoder_ids}")

# Check if there's a language token
if hasattr(tokenizer, 'lang_code_to_id'):
    print("\nLanguage code to ID mapping:")
    for lang, idx in tokenizer.lang_code_to_id.items():
        if lang in ['ja', 'ko', 'zh', 'en', 'fr']:
            print(f"  {lang}: {idx}")
