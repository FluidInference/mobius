"""Create alias files expected by the Swift EnglishTokenizer.

The Swift EnglishTokenizer loads constants with short names:
  - english_phoneme_dict.json   (phoneme dictionary)
  - english_token2id.json       (token→ID mapping)
  - english_heteronyms.json     (heteronyms list)

But export_tokenizers.py generates them with the full tokenizer name prefix:
  - english_phoneme_phoneme_dict.json
  - english_phoneme_token2id.json
  - english_phoneme_heteronyms.json

This script copies the canonical files to the short-name aliases that Swift expects.
Run this after export_tokenizers.py.

Usage:
    python export_tokenizer_aliases.py [--output-dir constants]
"""
import argparse
import os
import shutil

ALIASES = [
    ("english_phoneme_phoneme_dict.json", "english_phoneme_dict.json"),
    ("english_phoneme_token2id.json", "english_token2id.json"),
    ("english_phoneme_heteronyms.json", "english_heteronyms.json"),
]


def export_aliases(output_dir="constants"):
    for src_name, dst_name in ALIASES:
        src = os.path.join(output_dir, src_name)
        dst = os.path.join(output_dir, dst_name)
        if not os.path.exists(src):
            print(f"  WARNING: {src_name} not found, skipping")
            continue
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"  {src_name} -> {dst_name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="constants")
    args = parser.parse_args()
    export_aliases(args.output_dir)
