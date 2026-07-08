"""Approach B baseline: Misaki (Kokoro G2P) output + symbol mapping to the
espeak token set, scored against the espeak oracle.

Runs in a venv with `misaki` installed (spacy en_core_web_sm, num2words):
    /tmp/misaki-venv/bin/python coreml/g2p/misaki_mapping.py \
        --corpus coreml/g2p/corpus_en_1000.txt --out /tmp/misaki_dump.jsonl

The dump is then scored with `coreml.g2p.validate score`.

Mapping rules (best-effort espeak-parity):
  ligatures/diphthong shorthands: ʤ->dʒ ʧ->tʃ A->eɪ I->aɪ W->aʊ O->oʊ Y->ɔɪ
  rhotics: əɹ->ɚ  ɜɹ->ɜː(+ɹ before vowel is espeak-side; approximated ɜːɹ)
  length: ɑ->ɑː ɔ->ɔː(not before n) i->iː(stressed nucleus) u->uː ɜ->ɜː
"""

import argparse
import json
import re
from pathlib import Path


def map_to_espeak(phonemes: str) -> str:
    s = phonemes
    # multi-char first
    s = s.replace("ʤ", "dʒ").replace("ʧ", "tʃ")
    s = s.replace("əɹ", "ɚ").replace("ɜɹ", "ɜːɹ")
    for src, dst in [
        ("A", "eɪ"), ("I", "aɪ"), ("W", "aʊ"), ("O", "oʊ"), ("Y", "ɔɪ"),
        ("ᵊ", "ə"),
    ]:
        s = s.replace(src, dst)
    # length marks: espeak en-us long vowels
    s = re.sub(r"ɑ(?!ː)", "ɑː", s)
    s = re.sub(r"ɔ(?![ːɪn])", "ɔː", s)
    s = re.sub(r"ɜ(?!ː)", "ɜː", s)
    s = re.sub(r"u(?!ː)", "uː", s)
    # 'i' as a stressed nucleus is long in espeak (sˈiː), word-final happy-i
    # stays short; approximate: i after a stress mark cluster gets ː
    s = re.sub(r"([ˈˌ][^aeiouɑɔɜʊɪʌæɛə]*)i(?![ː])", r"\1iː", s)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--tokens", required=True, help="tokens.txt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from misaki import en

    g2p = en.G2P(trf=False, british=False)

    token2id = {}
    for line in Path(args.tokens).read_text(encoding="utf-8").splitlines():
        tab = line.index("\t")
        token2id[line[:tab]] = int(line[tab + 1:])

    sentences = [
        s.strip()
        for s in Path(args.corpus).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        for text in sentences:
            result = g2p(text)
            phonemes = result[0] if isinstance(result, tuple) else result
            mapped = map_to_espeak(phonemes)
            ids = [token2id[c] for c in mapped if c in token2id]
            f.write(
                json.dumps(
                    {"text": text, "phonemes": mapped, "ids": ids},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(sentences)} entries to {args.out}")


if __name__ == "__main__":
    main()
