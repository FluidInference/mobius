#!/usr/bin/env python3
"""LibriSpeech WER benchmark for the parakeet-unified CoreML chain.

Decodes LibriSpeech test-clean with the CoreML components (greedy RNNT) in
offline and/or streaming mode and reports WER + RTFx. Text is normalized the
same way as the nemotron benchmark: strip punctuation, lowercase.

The tokenizer is read directly from the .nemo archive (SentencePiece) so the
benchmark does not need to load the full NeMo model.

Usage:
    uv run --no-sync python benchmark_wer.py --mode both --max-files 200
    uv run --no-sync python benchmark_wer.py --mode streaming --streaming-context 70,13,13
"""
from __future__ import annotations

import argparse
import re
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import jiwer
import numpy as np
import sentencepiece as spm
import soundfile as sf

from coreml_rnnt import (
    OFFLINE_WINDOW_SAMPLES,
    SAMPLE_RATE,
    CoreMLRnnt,
    offline_transcribe,
    stream_transcribe,
)

DEFAULT_LIBRISPEECH = Path.home() / "Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean"


def load_tokenizer(nemo_path: Path) -> spm.SentencePieceProcessor:
    with tarfile.open(nemo_path, "r") as tar:
        names = [n for n in tar.getnames() if n.endswith("_tokenizer.model") or n.endswith("/tokenizer.model")]
        assert names, f"no tokenizer.model found in {nemo_path}"
        with tempfile.NamedTemporaryFile(suffix=".model") as tmp:
            tmp.write(tar.extractfile(names[0]).read())
            tmp.flush()
            sp = spm.SentencePieceProcessor()
            sp.load(tmp.name)
    return sp


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s']", "", text)
    text = text.replace("'", "'")
    return " ".join(text.lower().split())


def collect_files(root: Path, max_files: int) -> List[Tuple[Path, str]]:
    """Returns (flac_path, reference_transcript), sorted for reproducibility."""
    refs: Dict[str, str] = {}
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            file_id, _, text = line.partition(" ")
            refs[file_id] = text.strip()
    files = []
    for flac in sorted(root.rglob("*.flac")):
        if flac.stem in refs:
            files.append((flac, refs[flac.stem]))
    return files[:max_files] if max_files > 0 else files


def run_mode(
    mode: str,
    files: List[Tuple[Path, str]],
    coreml_dir: Path,
    sp: spm.SentencePieceProcessor,
    context: Tuple[int, int, int],
) -> None:
    if mode == "offline":
        cm = CoreMLRnnt(coreml_dir)
    else:
        cm = CoreMLRnnt(coreml_dir, streaming_suffix=f"{context[0]}_{context[1]}_{context[2]}")

    hyps: List[str] = []
    refs: List[str] = []
    skipped = 0
    audio_seconds = 0.0
    decode_seconds = 0.0

    for i, (flac, ref) in enumerate(files):
        audio, sr = sf.read(str(flac), dtype="float32")
        assert sr == SAMPLE_RATE
        if mode == "offline" and audio.size > OFFLINE_WINDOW_SAMPLES:
            skipped += 1
            continue
        t0 = time.time()
        if mode == "offline":
            tokens = offline_transcribe(cm, audio)
        else:
            tokens = stream_transcribe(cm, audio, context)
        decode_seconds += time.time() - t0
        audio_seconds += audio.size / SAMPLE_RATE
        hyps.append(normalize_text(sp.decode([int(t) for t in tokens])))
        refs.append(normalize_text(ref))
        if (i + 1) % 50 == 0:
            print(f"  [{mode}] {i + 1}/{len(files)} wer={jiwer.wer(refs, hyps) * 100:.2f}%")

    wer = jiwer.wer(refs, hyps)
    rtfx = audio_seconds / decode_seconds if decode_seconds else float("inf")
    print(f"\n=== {mode.upper()} ===")
    if mode == "streaming":
        print(f"context [L,C,R] = {list(context)} (latency {(context[1] + context[2]) * 0.08:.2f} s)")
    print(f"files: {len(hyps)} (skipped {skipped} > 15 s)" if mode == "offline" else f"files: {len(hyps)}")
    print(f"WER: {wer * 100:.2f}%")
    print(f"audio: {audio_seconds / 60:.1f} min, decode: {decode_seconds / 60:.1f} min, RTFx: {rtfx:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-dir", type=Path, default=DEFAULT_LIBRISPEECH)
    parser.add_argument("--coreml-dir", type=Path, default=Path("build/parakeet_unified_coreml"))
    parser.add_argument("--nemo-path", type=Path, default=Path("parakeet-unified-en-0.6b.nemo"))
    parser.add_argument("--mode", choices=["offline", "streaming", "both"], default="both")
    parser.add_argument("--streaming-context", type=str, default="70,13,13")
    parser.add_argument("--max-files", type=int, default=0, help="0 = all 2620 files")
    args = parser.parse_args()

    sp = load_tokenizer(args.nemo_path)
    files = collect_files(args.librispeech_dir, args.max_files)
    print(f"LibriSpeech test-clean: {len(files)} files from {args.librispeech_dir}")

    context = tuple(int(x) for x in args.streaming_context.split(","))
    if args.mode in ("offline", "both"):
        run_mode("offline", files, args.coreml_dir, sp, context)
    if args.mode in ("streaming", "both"):
        run_mode("streaming", files, args.coreml_dir, sp, context)


if __name__ == "__main__":
    main()
