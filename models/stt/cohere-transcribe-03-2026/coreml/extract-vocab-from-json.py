#!/usr/bin/env python3
"""Extract vocabulary from tokenizer.json file."""

import json
from pathlib import Path
from huggingface_hub import hf_hub_download

def extract_vocabulary():
    """Download tokenizer.json and extract vocabulary."""

    print("Downloading tokenizer.json from HuggingFace...")
    tokenizer_path = hf_hub_download(
        repo_id="CohereLabs/cohere-transcribe-03-2026",
        filename="tokenizer.json",
        force_download=False
    )

    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # Extract vocabulary from tokenizer.json
    # The vocab is typically in tokenizer_data["model"]["vocab"]
    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
        vocab = tokenizer_data["model"]["vocab"]
    else:
        raise ValueError("Could not find vocab in tokenizer.json")

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
        if i in vocab_sorted:
            token = vocab_sorted[i]
            # Escape special chars for display
            display = repr(token)[1:-1]  # Remove quotes from repr
            print(f"    {i}: {display}")

    # Look for special tokens in tokenizer data
    print("\n  Special tokens:")
    if "added_tokens" in tokenizer_data:
        for token_info in tokenizer_data["added_tokens"]:
            if isinstance(token_info, dict):
                content = token_info.get("content", "")
                token_id = token_info.get("id", "?")
                special = token_info.get("special", False)
                if special:
                    print(f"    {content} = {token_id}")

if __name__ == "__main__":
    extract_vocabulary()
