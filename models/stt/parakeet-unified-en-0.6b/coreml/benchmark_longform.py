#!/usr/bin/env python3
"""Long-form (hour-scale) WER benchmark for the parakeet-unified CoreML chain.

LibriSpeech test-clean utterances are concatenated (with short silence gaps)
into a single long composite with an exact reference transcript. The composite
is transcribed with the offline overlapping-batch mode and the streaming mode,
and compared against the same utterances scored individually — the delta
isolates what hour-scale processing adds (window seams, decoder state, merge
errors) from per-utterance model quality.

Usage:
    uv run --no-sync python benchmark_longform.py --minutes 60
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import jiwer
import numpy as np
import soundfile as sf

from benchmark_wer import DEFAULT_LIBRISPEECH, collect_files, load_tokenizer, normalize_text
from coreml_rnnt import (
    OFFLINE_WINDOW_SAMPLES,
    SAMPLE_RATE,
    CoreMLRnnt,
    batch_transcribe,
    offline_transcribe,
    splice_safe_token_ids,
    stream_transcribe,
)


def build_composite(
    files: List[Tuple[Path, str]], minutes: float, gap_seconds: float
) -> Tuple[np.ndarray, str, List[Tuple[Path, str]]]:
    gap = np.zeros(int(gap_seconds * SAMPLE_RATE), dtype=np.float32)
    pieces: List[np.ndarray] = []
    refs: List[str] = []
    used: List[Tuple[Path, str]] = []
    total = 0
    target = int(minutes * 60 * SAMPLE_RATE)
    for flac, ref in files:
        audio, sr = sf.read(str(flac), dtype="float32")
        assert sr == SAMPLE_RATE
        pieces.append(audio)
        pieces.append(gap)
        refs.append(ref)
        used.append((flac, ref))
        total += audio.size + gap.size
        if total >= target:
            break
    return np.concatenate(pieces), " ".join(refs), used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-dir", type=Path, default=DEFAULT_LIBRISPEECH)
    parser.add_argument("--coreml-dir", type=Path, default=Path("build/parakeet_unified_coreml"))
    parser.add_argument("--nemo-path", type=Path, default=Path("parakeet-unified-en-0.6b.nemo"))
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--gap-seconds", type=float, default=0.4)
    parser.add_argument("--streaming-context", type=str, default="70,13,13")
    args = parser.parse_args()

    sp = load_tokenizer(args.nemo_path)
    splice_safe = splice_safe_token_ids(sp)
    files = collect_files(args.librispeech_dir, 0)
    audio, reference, used = build_composite(files, args.minutes, args.gap_seconds)
    ref_norm = normalize_text(reference)
    print(
        f"composite: {audio.size / SAMPLE_RATE / 60:.1f} min from {len(used)} utterances "
        f"({len(ref_norm.split())} ref words)"
    )

    # Baseline: the same utterances scored individually (single offline windows;
    # utterances > 15 s use the batch path exactly like benchmark_wer.py).
    cm_offline = CoreMLRnnt(args.coreml_dir)
    hyps: List[str] = []
    refs: List[str] = []
    t0 = time.time()
    for flac, ref in used:
        utt, _ = sf.read(str(flac), dtype="float32")
        if utt.size > OFFLINE_WINDOW_SAMPLES:
            tokens = batch_transcribe(cm_offline, utt, splice_safe)
        else:
            tokens = offline_transcribe(cm_offline, utt)
        hyps.append(normalize_text(sp.decode([int(t) for t in tokens])))
        refs.append(normalize_text(ref))
    base_wer = jiwer.wer(refs, hyps)
    print(f"\nper-utterance baseline : WER {base_wer * 100:.2f}%  ({time.time() - t0:.0f}s)")

    # Offline overlapping batch on the full composite.
    t0 = time.time()
    tokens = batch_transcribe(cm_offline, audio, splice_safe)
    batch_wer = jiwer.wer(ref_norm, normalize_text(sp.decode([int(t) for t in tokens])))
    dt = time.time() - t0
    print(
        f"offline batch (1 file) : WER {batch_wer * 100:.2f}%  "
        f"(Δ {(batch_wer - base_wer) * 100:+.2f}, RTFx {audio.size / SAMPLE_RATE / dt:.0f})"
    )

    # Streaming on the full composite (one continuous session).
    context = tuple(int(x) for x in args.streaming_context.split(","))
    cm_stream = CoreMLRnnt(args.coreml_dir, streaming_suffix="_".join(map(str, context)))
    t0 = time.time()
    tokens = stream_transcribe(cm_stream, audio, context)
    stream_wer = jiwer.wer(ref_norm, normalize_text(sp.decode([int(t) for t in tokens])))
    dt = time.time() - t0
    print(
        f"streaming (1 session)  : WER {stream_wer * 100:.2f}%  "
        f"(RTFx {audio.size / SAMPLE_RATE / dt:.0f})"
    )


if __name__ == "__main__":
    main()
