#!/usr/bin/env python3
"""Corpus-level WER for FluidAudio `asr-benchmark` JSON files (total edit distance / total reference words).

The benchmark's own `averageWER` is a per-file mean, which one 2-word utterance can move by a point; published
leaderboards report the corpus ratio, so compare that. Same normalizer as the benchmark: lowercase, strip punctuation
except apostrophes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def norm(t: str) -> list:
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return t.split()


def edits(ref: list, hyp: list) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)]


def score(path: Path) -> dict:
    j = json.loads(path.read_text())
    e = n = 0
    for r in j["results"]:
        ref, hyp = norm(r["reference"]), norm(r["hypothesis"])
        e += edits(ref, hyp)
        n += len(ref)
    s = j["summary"]
    return {
        "file": path.name, "files": len(j["results"]), "corpus_wer": 100 * e / max(1, n),
        "per_file_mean_wer": 100 * s["averageWER"], "overall_rtfx": s["overallRTFx"],
        "audio_s": s["totalAudioDuration"], "proc_s": s["totalProcessingTime"],
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = score(Path(p))
        print(f"{r['file']:40s} n={r['files']:5d}  corpus WER {r['corpus_wer']:.2f}%  (per-file mean {r['per_file_mean_wer']:.2f}%)  "
              f"RTFx {r['overall_rtfx']:.1f}x  ({r['audio_s']:.0f}s / {r['proc_s']:.1f}s)")
