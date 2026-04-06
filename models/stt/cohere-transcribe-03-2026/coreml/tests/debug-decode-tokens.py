#!/usr/bin/env python3
"""Debug: Decode individual tokens to understand what's being generated."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sentencepiece as spm

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.Load("../tokenizer.model")

# Test tokens from the stateful decoder output (Sample 1)
print("Sample 1 token sequence:")
tokens = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13, 2155, 13777, 853, 7051, 546, 1250, 1800, 934, 579, 604, 527, 3]
print(f"Tokens: {tokens}")

# Decode full sequence
full_text = sp.DecodeIds(tokens)
print(f"\nFull decode: \"{full_text}\"")

# Decode just the prompt
prompt_tokens = tokens[:10]
prompt_text = sp.DecodeIds(prompt_tokens)
print(f"\nPrompt (tokens 0-9): {prompt_tokens}")
print(f"Prompt decode: \"{prompt_text}\"")

# Decode generated tokens
gen_tokens = tokens[10:]
gen_text = sp.DecodeIds(gen_tokens)
print(f"\nGenerated (tokens 10+): {gen_tokens}")
print(f"Generated decode: \"{gen_text}\"")

# Decode individual generated tokens to see what they are
print(f"\nIndividual token decodes:")
for i, tok in enumerate(gen_tokens[:10]):
    decoded = sp.DecodeIds([tok])
    print(f"  Token {tok:5d}: \"{decoded}\"")

# Check what "concord" should tokenize to
print(f"\nExpected text: \"concord returned to its place amidst the tents\"")
expected_tokens = sp.EncodeAsIds("concord returned to its place amidst the tents")
print(f"Expected tokens: {expected_tokens[:15]}...")
expected_text = sp.DecodeIds(expected_tokens)
print(f"Expected decode: \"{expected_text}\"")
