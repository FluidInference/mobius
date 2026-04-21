"""Export a Qwen2 tokenizer parity fixture for the FluidAudio Swift port.

Dumps:
  build/frontend/tokenizer_fixture.json
      {
        "special_tokens": {"<|endoftext|>": 151643, ...},
        "cases": [
            {"text": "...", "ids": [..]},
            ...
        ]
      }

The Swift Qwen2BpeTokenizer parity test loads this file and asserts
byte-for-byte equality on every case.

Usage:
    uv run python verify/export_tokenizer_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "CosyVoice"))
sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))

from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer  # noqa: E402


TOKEN_DIR = ROOT / "cosyvoice3_dl" / "CosyVoice-BlankEN"
OUTPUT = ROOT / "build" / "frontend" / "tokenizer_fixture.json"


TEST_CASES = [
    # Pure ASCII
    "Hello, world!",
    "You are a helpful assistant.",
    "The quick brown fox jumps over the lazy dog.",
    "I'm happy, you're great, we've been.",
    "123 456.789 $42",
    # Pure Mandarin
    "希望你以后能够做的比我还好呦。",
    "你好，世界！",
    "今天天气真不错。",
    "一二三四五六七八九十",
    # Mixed
    "The 中文 and English 混合 text.",
    "Qwen2 模型 is great!",
    # With prompt delimiter
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    "You are a helpful assistant.<|endofprompt|>Hello",
    # With phoneme tags
    "[breath] hello",
    "text [laughter] more text",
    # Whitespace edge cases
    "  leading spaces",
    "trailing spaces  ",
    "multiple    spaces",
    "line1\nline2",
    # Punctuation heavy
    "?????!!!!!",
    "...ok",
    # Empty-ish
    " ",
    "",
    # Long tokens / rare combos
    "aaaaaaaabbbbbbbb",
    "Unicode emoji: 🎉🔥🚀",  # 4-byte UTF-8 chars
]


def main() -> None:
    tok = get_qwen_tokenizer(str(TOKEN_DIR), skip_special_tokens=False, version="cosyvoice3")

    # Collect the full special-token map (content -> id).
    # get_added_vocab() returns {str: int} for every added token, including
    # permanent ones from tokenizer_config.json and runtime add_special_tokens.
    special_map = tok.tokenizer.get_added_vocab()
    # Keep only entries that are actually special (all added ones are, here).
    special_map = {k: int(v) for k, v in special_map.items()}

    cases = []
    for text in TEST_CASES:
        ids = tok.encode(text)
        cases.append({"text": text, "ids": [int(x) for x in ids]})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "special_tokens": special_map,
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT}")
    print(f"  special tokens: {len(special_map)}")
    print(f"  cases         : {len(cases)}")


if __name__ == "__main__":
    main()
