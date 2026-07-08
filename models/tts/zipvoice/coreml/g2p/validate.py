"""LuxTTS phase-2 G2P validation harness.

Subcommands:
  dump-oracle   corpus.txt -> oracle_tokens.jsonl (EmiliaTokenizer ground truth:
                map_punctuations + EnglishTextNormalizer + espeak en-us)
  word-gap      measure whole-sentence vs word-by-word-joined espeak output
                (quantifies sentence-level effects a lexicon G2P must model)
  score         score a Swift token dump (jsonl: {"text":..., "ids":[...]})
                against the oracle: sentence exact-match % + token edit %

Gates (LuxTTS phase 2): sentence exact >= 90%, overall token edit <= 2%.

Usage:
    .venv/bin/python -m coreml.g2p.validate dump-oracle \
        --corpus coreml/g2p/corpus_en_1000.txt --out coreml/g2p/oracle_tokens.jsonl
    .venv/bin/python -m coreml.g2p.validate word-gap --corpus ... --limit 500
    .venv/bin/python -m coreml.g2p.validate score \
        --oracle coreml/g2p/oracle_tokens.jsonl --swift swift_dump.jsonl
"""

import argparse
import json
import re
from collections import Counter
from functools import lru_cache, reduce
from pathlib import Path

from piper_phonemize import phonemize_espeak

from zipvoice.tokenizer.tokenizer import EmiliaTokenizer

STAGING_TOKENS = Path(__file__).resolve().parents[2] / "build/hf-staging/tokens.txt"


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_corpus(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_oracle(args):
    tokenizer = EmiliaTokenizer(token_file=str(STAGING_TOKENS))
    sentences = load_corpus(args.corpus)
    with open(args.out, "w", encoding="utf-8") as f:
        for text in sentences:
            tokens = tokenizer.texts_to_tokens([text])[0]
            ids = tokenizer.tokens_to_token_ids([tokens])[0]
            f.write(
                json.dumps({"text": text, "tokens": tokens, "ids": ids}, ensure_ascii=False)
                + "\n"
            )
    print(f"wrote {len(sentences)} oracle entries to {args.out}")


@lru_cache(maxsize=None)
def word_phonemes(word):
    result = phonemize_espeak(word, "en-us")
    return "".join("".join(clause) for clause in result)


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’]*|[,.!?;:]")


def word_gap(args):
    """Phonemize each sentence whole vs word-by-word-joined; report the gap."""
    tokenizer = EmiliaTokenizer(token_file=str(STAGING_TOKENS))
    sentences = load_corpus(args.corpus)[: args.limit]

    exact = 0
    total_tokens = 0
    total_edit = 0
    diff_words = Counter()

    for text in sentences:
        # Same normalization path the tokenizer applies before espeak.
        normalized = tokenizer.english_normalizer.normalize(
            tokenizer.preprocess_text(text)
        )
        whole = tokenizer.texts_to_tokens([text])[0]

        pieces = WORD_RE.findall(normalized)
        joined = ""
        for piece in pieces:
            if piece in ",.!?;:":
                joined = joined.rstrip() + piece + " "
            else:
                joined += word_phonemes(piece) + " "
        approx = list(joined.rstrip())

        dist = edit_distance(whole, approx)
        total_tokens += len(whole)
        total_edit += dist
        if dist == 0:
            exact += 1
        elif args.verbose:
            print(f"[{dist}] {text}")
            print("  whole :", "".join(whole))
            print("  joined:", "".join(approx))

        # attribute diffs to words: re-phonemize sentence with each word
        # replaced by its isolated pronunciation is expensive; instead do a
        # cheap per-word alignment on space-separated chunks.
        whole_chunks = "".join(whole).split(" ")
        approx_chunks = "".join(approx).split(" ")
        if len(whole_chunks) == len(approx_chunks) == len(
            [p for p in pieces if p not in ",.!?;:"]
        ):
            words = [p for p in pieces if p not in ",.!?;:"]
            for w, wc, ac in zip(words, whole_chunks, approx_chunks):
                if wc != ac:
                    diff_words[w.lower()] += 1

    n = len(sentences)
    print(f"sentences: {n}")
    print(f"exact-match (word-joined == whole): {exact}/{n} = {100 * exact / n:.1f}%")
    print(f"token edit rate: {total_edit}/{total_tokens} = {100 * total_edit / total_tokens:.2f}%")
    print("top context-sensitive words:")
    for word, count in diff_words.most_common(40):
        print(f"  {count:5d}  {word}")


def score(args):
    oracle = [json.loads(line) for line in open(args.oracle, encoding="utf-8")]
    swift = [json.loads(line) for line in open(args.swift, encoding="utf-8")]
    assert len(oracle) == len(swift), (len(oracle), len(swift))

    exact = 0
    total_tokens = 0
    total_edit = 0
    worst = []
    for ref, hyp in zip(oracle, swift):
        assert ref["text"] == hyp.get("text", ref["text"]), ref["text"]
        dist = edit_distance(ref["ids"], hyp["ids"])
        total_tokens += len(ref["ids"])
        total_edit += dist
        if dist == 0:
            exact += 1
        else:
            worst.append((dist, ref["text"]))

    n = len(oracle)
    sent_pct = 100 * exact / n
    edit_pct = 100 * total_edit / total_tokens
    print(f"sentence exact match: {exact}/{n} = {sent_pct:.1f}%  (gate >= 90%)")
    print(f"token edit rate:      {total_edit}/{total_tokens} = {edit_pct:.2f}%  (gate <= 2%)")
    worst.sort(reverse=True)
    if args.verbose:
        for dist, text in worst[:30]:
            print(f"  [{dist}] {text}")
    print("PASS" if sent_pct >= 90 and edit_pct <= 2 else "FAIL")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dump-oracle")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=dump_oracle)

    p = sub.add_parser("word-gap")
    p.add_argument("--corpus", required=True)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=word_gap)

    p = sub.add_parser("score")
    p.add_argument("--oracle", required=True)
    p.add_argument("--swift", required=True)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
