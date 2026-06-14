#!/usr/bin/env python3
"""WER gate for the MIL-fused EOU decode graph.

The MIL fusion (fuse_decoder_joint_decision.py) shifts fp16 logits by up to
~3.5 absolute (4e-3 relative), which flips argmax on low-margin frames and
changes transcripts on ~25% of LibriSpeech utterances. Neither output is
"more correct" — both are fp16 roundings of the same fp32 math — so the ship
gate is WER against ground truth: if fused WER == ref WER within noise, the
fusion is quality-neutral; if worse, no-ship.

Uses the LibriSpeech *.trans.txt files next to each flac.

Usage:
    python wer_ref_vs_fused.py \
        --model-dir "$HOME/Library/Application Support/FluidAudio/Models/parakeet-eou-streaming/160ms" \
        --fused /tmp/eou_fused/decoder_joint_decision_fused.mlpackage \
        --librispeech-root "$HOME/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean" \
        --num-files 50 --seconds 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from parity_fused_decode import (
    EOU_ID,
    FusedDecoder,
    RefDecoder,
    encoder_steps,
    run_loop,
)


def edit_distance(ref: list, hyp: list) -> int:
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--librispeech-root", type=Path, required=True)
    ap.add_argument("--num-files", type=int, default=50)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    vocab = {int(k): v for k, v in json.loads((args.model_dir / "vocab.json").read_text()).items()}

    def detok(ids):
        return "".join(vocab.get(i, "") for i in ids if i != EOU_ID).replace("▁", " ").strip()

    flacs = sorted(args.librispeech_root.rglob("*.flac"))
    step = max(1, len(flacs) // args.num_files)
    flacs = flacs[::step][: args.num_files]

    ref = RefDecoder(args.model_dir)
    fused = FusedDecoder(args.fused)

    ref_err = fused_err = total_words = 0
    n_text_diff = 0
    for flac in flacs:
        trans_file = flac.parent / f"{flac.parent.parent.name}-{flac.parent.name}.trans.txt"
        truth = None
        for line in trans_file.read_text().splitlines():
            key, _, text = line.partition(" ")
            if key == flac.stem:
                truth = text.lower().split()
                break
        if truth is None:
            continue

        audio, sr = sf.read(str(flac), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        full = len(audio) <= int(args.seconds * sr)
        audio = audio[: int(args.seconds * sr)]
        frames = encoder_steps(args.model_dir, audio)

        ref.reset()
        fused.reset()
        ref_tokens, _ = run_loop(ref, frames)
        fused_tokens, _ = run_loop(fused, frames)
        ref_text = detok(ref_tokens).split()
        fused_text = detok(fused_tokens).split()

        if not full:
            # Audio truncated: align truth to the shorter hypothesis length+5
            truth = truth[: max(len(ref_text), len(fused_text)) + 5]

        ref_err += edit_distance(truth, ref_text)
        fused_err += edit_distance(truth, fused_text)
        total_words += len(truth)
        diff = ref_text != fused_text
        n_text_diff += int(diff)
        print(f"{flac.stem}: words={len(truth)} refE={edit_distance(truth, ref_text)} "
              f"fusedE={edit_distance(truth, fused_text)}{' DIFF' if diff else ''}")

    print()
    print(f"files                  : {len(flacs)} ({n_text_diff} with differing transcripts)")
    print(f"total reference words  : {total_words}")
    print(f"WER two-model reference: {100.0 * ref_err / total_words:.2f}%")
    print(f"WER MIL-fused          : {100.0 * fused_err / total_words:.2f}%")


if __name__ == "__main__":
    main()
