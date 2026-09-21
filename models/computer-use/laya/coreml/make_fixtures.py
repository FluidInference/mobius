"""Generate tokenizer and sequence fixtures used by the Swift parity tests."""

from __future__ import annotations

import json

import laya
from tokenizers import Tokenizer

from assets import ROOT, checkpoint_dir, verify_assets
from preprocessing import Shape, encode

TEXTS = [
    "hello world",
    " hello world",
    "Hello, World!",
    "The piece leaves one hole under it and makes a small bump on top.",
    "noul question: Is this placement clean (no new holes, flat surface)?",
    " false: no, the statement does not hold",
    "客户说：我的订单已经三周了还没有发货。",
    "ユーザー：昨日アップデートしてからアプリがすぐに落ちます。",
    "Die Lieferung kam beschädigt an",
    "Ça va? Très bien — merci!",
    "emoji 🙂🚀 test",
    "tabs\tand\nnewlines  double  spaces",
    "12345 67.89 $100",
    "<mask> literal and <start_of_turn> token",
    "→←↑↓ ∑∫ ≠",
    "ᚠᚢᚦ runes",
    "Mixed 中文 and English",
    "fraktur \U0001d518\U0001d52b\U0001d526 letters",
    "private use  char",
    "new emoji \U0001fae0 melting",
    "<unused9> inside text",
    "text<mask>glued and <mask> spaced",
    "hello▁world literal marker",
    "combining é accent",
    "surrogate pair \U0001f600 grin",
    "trailing space ",
    "",
    "   ",
    "▁literal underscore marker",
    "a b nbsp",
]


def main() -> None:
    verify_assets("multilingual")
    tok = Tokenizer.from_file(str(checkpoint_dir("multilingual") / "tokenizer" / "tokenizer.json"))
    cases = [{"text": text, "ids": tok.encode(text, add_special_tokens=False).ids} for text in TEXTS]
    (ROOT / "fixtures" / "tokenizer-cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")

    agent = laya.load(str(checkpoint_dir("multilingual")), device="cpu")
    prompts = json.loads((ROOT / "fixtures" / "cases.json").read_text())
    sequences = []
    for case in prompts:
        for qid, question in case["questions"].items():
            for length in (128, 256):
                try:
                    ids, markers, qtype = encode(
                        agent.tok, case["state"], question, Shape(length), agent.cfg["head_max_len"]
                    )
                except ValueError:
                    continue
                criteria = question.get("criteria")
                if isinstance(criteria, dict):
                    options = [[label, description] for label, description in criteria.items()]
                elif isinstance(criteria, list):
                    options = [[label, None] for label in criteria]
                else:
                    options = []
                sequences.append(
                    {
                        "case": case["name"],
                        "question": qid,
                        "length": length,
                        "state": case["state"],
                        "type": question["type"],
                        "instructions": question["instructions"],
                        "options": options,
                        "ids": ids,
                        "markers": markers,
                        "qtype": qtype,
                    }
                )
    (ROOT / "fixtures" / "sequence-cases.json").write_text(json.dumps(sequences, ensure_ascii=False, indent=1) + "\n")
    print(f"tokenizer cases: {len(cases)}, sequence cases: {len(sequences)}")


if __name__ == "__main__":
    main()
