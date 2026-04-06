#!/usr/bin/env python3
"""Extract vocabulary from Cohere tokenizer for Swift usage."""

import json
from pathlib import Path
from transformers import AutoTokenizer

def extract_vocabulary():
    """Extract vocabulary from Cohere tokenizer and save as simple JSON."""

    print("Loading Cohere tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True)

    # Get vocabulary as dict: {token_string: token_id}
    vocab = tokenizer.get_vocab()

    # Invert to {token_id: token_string} for easy lookup in Swift
    vocab_inverted = {v: k for k, v in vocab.items()}

    # Sort by token ID
    vocab_sorted = dict(sorted(vocab_inverted.items()))

    # Save to JSON
    output_path = Path("hf-upload/vocab.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_sorted, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved vocabulary with {len(vocab_sorted)} tokens to {output_path}")
    print(f"  Vocab size: {len(vocab_sorted)}")
    print(f"  Sample tokens:")
    for i in range(min(10, len(vocab_sorted))):
        token = vocab_sorted[i]
        # Escape special chars for display
        display = repr(token)[1:-1]  # Remove quotes from repr
        print(f"    {i}: {display}")

    # Verify special tokens
    print("\n  Special tokens:")
    print(f"    BOS (start): {tokenizer.bos_token} = {tokenizer.bos_token_id}")
    print(f"    EOS (end): {tokenizer.eos_token} = {tokenizer.eos_token_id}")
    print(f"    PAD: {tokenizer.pad_token} = {tokenizer.pad_token_id}")
    print(f"    UNK: {tokenizer.unk_token} = {tokenizer.unk_token_id}")

if __name__ == "__main__":
    extract_vocabulary()
