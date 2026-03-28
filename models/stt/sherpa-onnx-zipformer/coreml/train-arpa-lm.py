#!/usr/bin/env python3
"""Train an ARPA n-gram language model from text data.

Builds bigram (or trigram) ARPA LM from a text corpus using simple
maximum likelihood estimation with add-k smoothing.

Usage:
    python train-arpa-lm.py --text sentences.txt --output atc.arpa --order 2
    python train-arpa-lm.py --text sentences.txt --output atc.arpa --order 3 --min-count 2
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train ARPA n-gram LM from text")
    parser.add_argument("--text", required=True, help="Training text (one sentence per line)")
    parser.add_argument("--output", required=True, help="Output ARPA file path")
    parser.add_argument("--order", type=int, default=2, choices=[2, 3], help="N-gram order (default: 2)")
    parser.add_argument("--min-count", type=int, default=1, help="Min count to include n-gram (default: 1)")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Add-k smoothing (default: 0.1)")
    args = parser.parse_args()

    print(f"Loading text: {args.text}")
    sentences = []
    with open(args.text) as f:
        for line in f:
            words = line.strip().lower().split()
            if words:
                sentences.append(words)
    print(f"  {len(sentences)} sentences")

    # Count n-grams
    unigram_counts: Counter = Counter()
    bigram_counts: dict[str, Counter] = defaultdict(Counter)
    trigram_counts: dict[tuple, Counter] = defaultdict(Counter)

    for words in sentences:
        tokens = ["<s>"] + words + ["</s>"]
        for w in tokens:
            unigram_counts[w] += 1
        for i in range(len(tokens) - 1):
            bigram_counts[tokens[i]][tokens[i + 1]] += 1
        if args.order >= 3:
            for i in range(len(tokens) - 2):
                trigram_counts[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1

    vocab = sorted(unigram_counts.keys())
    V = len(vocab)
    total = sum(unigram_counts.values())
    k = args.smoothing
    min_count = args.min_count

    print(f"  vocab={V}, total_tokens={total}")

    # Compute probabilities
    # Unigrams: P(w) = (count(w) + k) / (total + k*V)
    unigrams = {}
    for w in vocab:
        p = (unigram_counts[w] + k) / (total + k * V)
        unigrams[w] = math.log10(p)

    # Bigrams: P(w2|w1) = (count(w1,w2) + k) / (count(w1) + k*V)
    bigrams = {}
    for w1 in bigram_counts:
        c1 = unigram_counts[w1]
        for w2, c12 in bigram_counts[w1].items():
            if c12 >= min_count:
                p = (c12 + k) / (c1 + k * V)
                bigrams[(w1, w2)] = math.log10(p)

    # Backoff weights: bow(w1) = log10(1 - sum(P_bi(w2|w1) for seen w2)) / (1 - sum(P_uni(w2) for seen w2))
    backoffs = {}
    for w1 in bigram_counts:
        seen_w2 = set(bigram_counts[w1].keys())
        c1 = unigram_counts[w1]
        sum_bi = sum((bigram_counts[w1][w2] + k) / (c1 + k * V) for w2 in seen_w2)
        sum_uni = sum(10 ** unigrams[w2] for w2 in seen_w2 if w2 in unigrams)
        if sum_bi < 1.0 and sum_uni < 1.0:
            bow = math.log10((1.0 - sum_bi) / max(1.0 - sum_uni, 1e-10))
        else:
            bow = 0.0
        backoffs[w1] = bow

    # Trigrams
    trigrams_out = {}
    if args.order >= 3:
        for (w1, w2), counts in trigram_counts.items():
            c12 = bigram_counts[w1][w2]
            for w3, c123 in counts.items():
                if c123 >= min_count:
                    p = (c123 + k) / (c12 + k * V)
                    trigrams_out[(w1, w2, w3)] = math.log10(p)

    # Write ARPA file
    n_uni = len(unigrams)
    n_bi = len(bigrams)
    n_tri = len(trigrams_out)

    print(f"  unigrams={n_uni}, bigrams={n_bi}, trigrams={n_tri}")
    print(f"Writing: {args.output}")

    with open(args.output, "w") as f:
        f.write("\\data\\\n")
        f.write(f"ngram 1={n_uni}\n")
        f.write(f"ngram 2={n_bi}\n")
        if n_tri > 0:
            f.write(f"ngram 3={n_tri}\n")
        f.write("\n\\1-grams:\n")
        for w in sorted(unigrams):
            bow = backoffs.get(w, 0.0)
            if bow != 0.0:
                f.write(f"{unigrams[w]:.6f}\t{w}\t{bow:.6f}\n")
            else:
                f.write(f"{unigrams[w]:.6f}\t{w}\n")

        f.write("\n\\2-grams:\n")
        for (w1, w2) in sorted(bigrams):
            f.write(f"{bigrams[(w1, w2)]:.6f}\t{w1}\t{w2}\n")

        if n_tri > 0:
            f.write("\n\\3-grams:\n")
            for (w1, w2, w3) in sorted(trigrams_out):
                f.write(f"{trigrams_out[(w1, w2, w3)]:.6f}\t{w1}\t{w2}\t{w3}\n")

        f.write("\n\\end\\\n")

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"Done. ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
