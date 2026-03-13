"""Export pypinyin character→pinyin dictionary and phrase dict for Swift.

Extracts from pypinyin and jieba packages:
- mandarin_pypinyin_char_dict.json — character → pinyin list
- mandarin_pypinyin_phrase_dict.json — phrase → pinyin list (polyphone disambiguation)
- mandarin_jieba_dict.json — word → frequency (for jieba word segmentation)

These are used by MandarinTokenizer.swift for on-device Mandarin G2P.

Usage:
    python extras/export_pypinyin.py [--output-dir constants]

Requires: pypinyin, jieba (install via: uv pip install pypinyin jieba)
"""
import argparse
import json
import os


def export_pypinyin(output_dir="constants"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Character → pinyin dict (from pypinyin.PINYIN_DICT)
    from pypinyin.core import PINYIN_DICT

    char_pinyin = {}
    for codepoint, pinyins_str in PINYIN_DICT.items():
        char = chr(codepoint)
        pinyins = pinyins_str.split(',')
        char_pinyin[char] = pinyins

    path = os.path.join(output_dir, "mandarin_pypinyin_char_dict.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(char_pinyin, f, ensure_ascii=False, separators=(',', ':'))
    size_kb = os.path.getsize(path) / 1024
    print(f"char_pinyin: {len(char_pinyin)} entries, {size_kb:.1f} KB")

    # 2. Phrase → pinyin dict (for multi-char polyphone disambiguation)
    from pypinyin.phrases_dict import phrases_dict

    # Convert from list-of-lists to list-of-strings for simpler Swift parsing
    phrase_pinyin = {}
    for phrase, pinyins_list in phrases_dict.items():
        # pinyins_list is like [['yī'], ['dīng'], ['bù'], ['shí']]
        # Flatten to ['yī', 'dīng', 'bù', 'shí']
        flat = [p[0] if isinstance(p, list) else p for p in pinyins_list]
        phrase_pinyin[phrase] = flat

    path = os.path.join(output_dir, "mandarin_pypinyin_phrase_dict.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(phrase_pinyin, f, ensure_ascii=False, separators=(',', ':'))
    size_kb = os.path.getsize(path) / 1024
    print(f"phrase_pinyin: {len(phrase_pinyin)} entries, {size_kb:.1f} KB")

    # 3. Export jieba dictionary (word, frequency, POS)
    import jieba
    jieba_dict_path = os.path.join(os.path.dirname(jieba.__file__), 'dict.txt')
    # Convert to JSON: word → freq (POS not needed for cut)
    jieba_data = {}
    with open(jieba_dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                freq = int(parts[1])
                jieba_data[word] = freq

    path = os.path.join(output_dir, "mandarin_jieba_dict.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jieba_data, f, ensure_ascii=False, separators=(',', ':'))
    size_kb = os.path.getsize(path) / 1024
    print(f"jieba_dict: {len(jieba_data)} entries, {size_kb:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="constants")
    args = parser.parse_args()
    export_pypinyin(args.output_dir)
