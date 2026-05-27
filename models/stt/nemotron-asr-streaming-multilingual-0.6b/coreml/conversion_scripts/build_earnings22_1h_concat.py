"""Build the Earnings22-1h continuous benchmark WAV.

Concatenates the top-4 calls' chunks (by total duration) from the
`earnings22-kws` test dataset into one 3600s (1h) WAV. This is the
**canonical conversational long-form bench input** — used by
`nemotron-multilingual-transcribe` to measure RTFx and WER.

Top-4 calls by total duration (sum = exactly 3600s):
  4483857 → 1095s (73 chunks)
  4485206 → 1080s (72 chunks)
  4473238 →  780s (52 chunks)
  4482641 →  645s (43 chunks)

Total: 3600s @ 16kHz mono PCM_16 → ~110 MB output WAV.

Usage:
    .venv/bin/python conversion_scripts/build_earnings22_1h_concat.py

Source: ~/Library/Application Support/FluidAudio/earnings22-kws/test-dataset/
Output: ~/Library/Application Support/FluidAudio/earnings22-1h/earnings22_top4_1h.wav

Bench command:
    fluidaudio nemotron-multilingual-transcribe \
        --input ~/Library/Application\ Support/FluidAudio/earnings22-1h/earnings22_top4_1h.wav \
        --model-dir <build-dir> \
        --language en-US

RTFx = 3600s / wall_clock_time. Time `time` the command and divide.
"""
from __future__ import annotations

import os
import re

import numpy as np
import soundfile as sf

# Top-4 Earnings22 calls by total chunked duration. Sum = 3600.0s.
TOP_CALLS = ["4483857", "4485206", "4473238", "4482641"]

SOURCE_DIR = os.path.expanduser(
    "~/Library/Application Support/FluidAudio/earnings22-kws/test-dataset"
)
DEST_DIR = os.path.expanduser(
    "~/Library/Application Support/FluidAudio/earnings22-1h"
)
DEST_PATH = os.path.join(DEST_DIR, "earnings22_top4_1h.wav")


def _chunk_num(filename: str, call_id: str) -> int:
    """Extract numeric chunk index from `<call_id>_chunk_<N>.wav` or
    `<call_id>_chunk<N>.wav` (both naming conventions appear in the dataset).
    """
    rest = filename[len(call_id):].replace(".wav", "")
    m = re.search(r"(\d+)", rest)
    return int(m.group(1)) if m else -1


def main() -> None:
    if not os.path.isdir(SOURCE_DIR):
        raise SystemExit(
            f"earnings22-kws dataset not found at {SOURCE_DIR}.\n"
            f"Run: fluidaudio download --dataset earnings22-kws"
        )
    os.makedirs(DEST_DIR, exist_ok=True)

    all_pcm: list[np.ndarray] = []
    sr_seen: int | None = None
    total_dur = 0.0
    for call in TOP_CALLS:
        files = [
            f for f in os.listdir(SOURCE_DIR)
            if f.startswith(call + "_chunk") and f.endswith(".wav")
        ]
        files.sort(key=lambda f: _chunk_num(f, call))
        if not files:
            raise SystemExit(f"No chunks found for call {call} in {SOURCE_DIR}")
        print(f"Call {call}: {len(files)} chunks")
        for f in files:
            pcm, sr = sf.read(os.path.join(SOURCE_DIR, f))
            if sr_seen is None:
                sr_seen = sr
            elif sr != sr_seen:
                raise SystemExit(f"Sample-rate mismatch: {sr} vs {sr_seen} (chunk {f})")
            if pcm.ndim > 1:
                pcm = pcm[:, 0]
            all_pcm.append(pcm.astype(np.float32))
            total_dur += len(pcm) / sr

    merged = np.concatenate(all_pcm)
    assert sr_seen is not None
    sf.write(DEST_PATH, merged, sr_seen, subtype="PCM_16")
    actual_dur = len(merged) / sr_seen
    size_mb = os.path.getsize(DEST_PATH) / (1024 * 1024)
    print()
    print(f"Wrote {DEST_PATH}")
    print(f"  Duration: {actual_dur:.1f}s @ {sr_seen} Hz, mono PCM_16")
    print(f"  Size:     {size_mb:.1f} MB")
    if abs(actual_dur - 3600.0) > 1.0:
        print(f"  WARNING: expected ~3600s, got {actual_dur:.1f}s")


if __name__ == "__main__":
    main()
