#!/usr/bin/env python3
"""
Real-world polyphone disambiguation test.

Feeds actual Mandarin sentences containing 行 in two different contexts
and asserts the CoreML model picks xíng (verb: walk/do) vs háng (noun:
row/profession). This is the production contract — if argmax picks the
wrong label, downstream TTS produces the wrong audio.

Sentences:
    - 我们走在马路上 → 行 absent (control)
    - 他在银行工作   → 行 should be ㄏㄤ2 (háng, "bank")
    - 我们一起去旅行 → 行 should be ㄒㄧㄥ2 (xíng, "travel")

Requires:
    - bert-base-chinese tokenizer (auto-downloaded by transformers)
    - build/g2pw/g2pw.mlpackage + POLYPHONIC_CHARS.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_chars_and_labels(polyphonic_path: Path):
    """Reproduce upstream `get_phoneme_labels` ordering."""
    pairs = []
    with polyphonic_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            pairs.append((parts[0], parts[1]))
    # `chars`: unique chars in order of first appearance
    chars = []
    seen = set()
    for c, _ in pairs:
        if c not in seen:
            seen.add(c)
            chars.append(c)
    # `labels`: sorted unique bopomofo strings
    labels = sorted({p for _, p in pairs})
    char2phonemes = {}
    for c, p in pairs:
        char2phonemes.setdefault(c, []).append(labels.index(p))
    return chars, labels, char2phonemes


def _build_inputs(
    sentence: str,
    target_char: str,
    target_idx: int,
    chars: list,
    labels: list,
    char2phonemes: dict,
    tokenizer,
    seq_len: int = 512,
):
    """Reproduce upstream `__getitem__` for a single sentence."""
    raw_tokens = list(sentence)
    tokens = ["[CLS]"] + raw_tokens + ["[SEP]"]
    if len(tokens) > seq_len:
        raise ValueError(f"sentence too long: {len(tokens)} > {seq_len}")

    pad_n = seq_len - len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens) + [0] * pad_n
    attention_mask = [1] * len(tokens) + [0] * pad_n
    token_type_ids = [0] * seq_len

    valid = set(char2phonemes[target_char])
    phoneme_mask = [1.0 if i in valid else 0.0 for i in range(len(labels))]

    char_id = chars.index(target_char)
    position_id = target_idx + 1  # +1 for [CLS]

    return {
        "input_ids": np.array([input_ids], dtype=np.int32),
        "token_type_ids": np.array([token_type_ids], dtype=np.int32),
        "attention_mask": np.array([attention_mask], dtype=np.int32),
        "phoneme_mask": np.array([phoneme_mask], dtype=np.float32),
        "char_ids": np.array([char_id], dtype=np.int32),
        "position_ids": np.array([position_id], dtype=np.int32),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coreml-dir", type=Path, default=Path("./build/g2pw"))
    args = p.parse_args()

    mlpkg = args.coreml_dir / "g2pw.mlpackage"
    poly = args.coreml_dir / "POLYPHONIC_CHARS.txt"
    if not mlpkg.exists() or not poly.exists():
        print(f"missing artefacts in {args.coreml_dir}", file=sys.stderr)
        return 2

    import coremltools as ct
    from transformers import AutoTokenizer

    print("[init] loading bert-base-chinese tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    print(f"[init] loading {mlpkg}")
    mlmodel = ct.models.MLModel(str(mlpkg), compute_units=ct.ComputeUnit.CPU_ONLY)

    chars, labels, char2phonemes = _load_chars_and_labels(poly)
    print(f"[init] {len(chars)} polyphonic chars, {len(labels)} labels")

    # g2pW v2 was trained on traditional Chinese — sentences below use
    # traditional glyphs. Characters identical in simplified+traditional
    # (行, 重, 都) work either way.
    cases = [
        # 行: háng (row/profession) vs xíng (walk/can/do)
        ("他在銀行工作", "行", 3, "ㄏㄤ2"),
        ("我們一起去旅行", "行", 6, "ㄒㄧㄥ2"),
        ("銀行的行長很忙", "行", 1, "ㄏㄤ2"),
        ("銀行的行長很忙", "行", 3, "ㄏㄤ2"),
        ("我行你也行", "行", 1, "ㄒㄧㄥ2"),
        ("我行你也行", "行", 4, "ㄒㄧㄥ2"),
        # 長: cháng (long) vs zhǎng (grow/elder/chief)
        ("這條河很長", "長", 4, "ㄔㄤ2"),
        ("他長得很高", "長", 1, "ㄓㄤ3"),
        ("他是公司的董事長", "長", 7, "ㄓㄤ3"),
        # 重: zhòng (heavy) vs chóng (again/repeat)
        ("這個箱子很重", "重", 5, "ㄓㄨㄥ4"),
        ("請你重新再說一遍", "重", 2, "ㄔㄨㄥ2"),
        # 都: dōu (all) vs dū (capital city)
        ("我們都喜歡吃水果", "都", 2, "ㄉㄡ1"),
        ("北京是中國的首都", "都", 7, "ㄉㄨ1"),
        # 覺: jué (feel/sense) vs jiào (sleep)
        ("我覺得很好", "覺", 1, "ㄐㄩㄝ2"),
        ("他睡了一個好覺", "覺", 6, "ㄐㄧㄠ4"),
    ]

    failures = 0
    for sentence, char, idx, expected in cases:
        feed = _build_inputs(
            sentence, char, idx, chars, labels, char2phonemes, tokenizer
        )
        out = mlmodel.predict(feed)
        probs = next(iter(out.values()))[0]
        top1 = int(np.argmax(probs))
        got = labels[top1]
        ok = "OK" if got == expected else "FAIL"
        if got != expected:
            failures += 1
        # show top-3 for context
        top3 = np.argsort(-probs)[:3]
        top3_str = ", ".join(f"{labels[i]}({probs[i]:.3f})" for i in top3)
        print(
            f"[{ok}] {sentence!r:18}  pos={idx}  expected={expected:8}  "
            f"got={got:8}  top3=[{top3_str}]"
        )

    if failures:
        print(f"\n{failures}/{len(cases)} failed", file=sys.stderr)
        return 1
    print(f"\nall {len(cases)} polyphone cases correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
