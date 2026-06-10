#!/usr/bin/env python3
"""Three-way WER gate: shipped reference pair vs traced fused vs MIL-lean fused.

Head-to-head deciding run for the EOU decoder+joint fusion. Decodes the SAME
LibriSpeech files with three decode paths, feeding all three IDENTICAL encoder
outputs (encoder frames are computed once per file and cached to .npy, so the
comparison isolates the decode graph):

  ref    shipped decoder.mlmodelc + joint_decision.mlmodelc (2 dispatches/step)
  traced torch.jit re-export fused decoder_joint_decision_fp16.mlpackage
         (fuse_decoder_joint.py, feat/parakeet-decode-fusion)
  lean   MIL-builder fused decoder_joint_decision_fused.mlpackage
         (fuse_decoder_joint_decision.py, this branch)

Per file: WER vs LibriSpeech ground truth for each variant + whether each
fused variant's emitted token sequence differs from the reference's.
Results are appended to a JSONL so the run is resumable; a summary pass
aggregates WER, diff counts, and the per-file blowup check for the ship gate.

Usage:
    python wer_three_way.py \
        --model-dir "$HOME/Library/Application Support/FluidAudio/Models/parakeet-eou-streaming/160ms" \
        --traced .../decoder_joint_decision_fp16.mlpackage \
        --lean /tmp/eou_fused/decoder_joint_decision_fused.mlpackage \
        --librispeech-root ".../LibriSpeech/test-clean" \
        --cache-dir /tmp/eou_enc_cache --results /tmp/eou_three_way.jsonl \
        --num-files 0            # 0 = all files
    python wer_three_way.py --summary --results /tmp/eou_three_way.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from parity_fused_decode import EOU_ID, FusedDecoder, RefDecoder, encoder_steps, run_loop
from wer_ref_vs_fused import edit_distance


def load_truth(flac: Path) -> list[str] | None:
    trans_file = flac.parent / f"{flac.parent.parent.name}-{flac.parent.name}.trans.txt"
    for line in trans_file.read_text().splitlines():
        key, _, text = line.partition(" ")
        if key == flac.stem:
            return text.lower().split()
    return None


def cached_encoder_steps(model_dir: Path, flac: Path, cache_dir: Path) -> np.ndarray:
    """Encoder frames [N, 512] for the full file, cached to .npy."""
    cache = cache_dir / f"{flac.stem}.npy"
    if cache.exists():
        return np.load(cache)
    audio, sr = sf.read(str(flac), dtype="float32")
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio[:, 0]
    frames = encoder_steps(model_dir, audio)
    np.save(cache, frames)
    return frames


def summarize(results_path: Path) -> None:
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    n = len(rows)
    total_words = sum(r["words"] for r in rows)
    print(f"files: {n}, reference words: {total_words}")
    for v in ("ref", "traced", "lean"):
        errs = sum(r[f"{v}_err"] for r in rows)
        print(f"WER {v:<6}: {100.0 * errs / total_words:.3f}%  ({errs} errors)")
    for v in ("traced", "lean"):
        diffs = sum(r[f"{v}_seq_diff"] for r in rows)
        print(f"token-seq diff vs ref, {v:<6}: {diffs}/{n} files")
    # Pathological per-file blowup check (ship gate): any file where a fused
    # variant's WER exceeds the reference's by > 20 pp absolute.
    for v in ("traced", "lean"):
        blowups = [
            (r["file"], 100.0 * (r[f"{v}_err"] - r["ref_err"]) / r["words"])
            for r in rows
            if r["words"] > 0 and (r[f"{v}_err"] - r["ref_err"]) / r["words"] > 0.20
        ]
        print(f"per-file blowups (> +20 pp WER vs ref), {v}: {len(blowups)}")
        for f, d in blowups[:10]:
            print(f"  {f}: +{d:.1f} pp")
    worst = sorted(
        ((r["file"], 100.0 * (r["lean_err"] - r["ref_err"]) / max(r["words"], 1)) for r in rows),
        key=lambda x: -x[1],
    )[:5]
    print("worst lean-vs-ref per-file deltas (pp):")
    for f, d in worst:
        print(f"  {f}: {d:+.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--traced", type=Path)
    ap.add_argument("--lean", type=Path)
    ap.add_argument("--librispeech-root", type=Path)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/eou_enc_cache"))
    ap.add_argument("--results", type=Path, default=Path("/tmp/eou_three_way.jsonl"))
    ap.add_argument("--num-files", type=int, default=0, help="0 = all")
    ap.add_argument("--summary", action="store_true", help="only aggregate an existing results file")
    args = ap.parse_args()

    if args.summary:
        summarize(args.results)
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    vocab = {int(k): v for k, v in json.loads((args.model_dir / "vocab.json").read_text()).items()}

    def detok(ids):
        return "".join(vocab.get(i, "") for i in ids if i != EOU_ID).replace("▁", " ").strip()

    flacs = sorted(args.librispeech_root.rglob("*.flac"))
    if args.num_files:
        flacs = flacs[: args.num_files]

    done = set()
    if args.results.exists():
        done = {json.loads(l)["file"] for l in args.results.read_text().splitlines() if l.strip()}

    ref = RefDecoder(args.model_dir)
    traced = FusedDecoder(args.traced)
    lean = FusedDecoder(args.lean)

    t_start = time.perf_counter()
    n_done_this_run = 0
    with args.results.open("a") as out:
        for idx, flac in enumerate(flacs):
            if flac.stem in done:
                continue
            truth = load_truth(flac)
            if truth is None:
                continue
            frames = cached_encoder_steps(args.model_dir, flac, args.cache_dir)

            texts, errs, seqs = {}, {}, {}
            for name, dec in (("ref", ref), ("traced", traced), ("lean", lean)):
                dec.reset()
                tokens, _ = run_loop(dec, frames)
                seqs[name] = tokens
                texts[name] = detok(tokens).split()
                errs[name] = edit_distance(truth, texts[name])

            row = {
                "file": flac.stem,
                "words": len(truth),
                "frames": int(frames.shape[0]),
                "ref_err": errs["ref"],
                "traced_err": errs["traced"],
                "lean_err": errs["lean"],
                "traced_seq_diff": int(seqs["traced"] != seqs["ref"]),
                "lean_seq_diff": int(seqs["lean"] != seqs["ref"]),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            n_done_this_run += 1
            if n_done_this_run % 25 == 0:
                el = time.perf_counter() - t_start
                print(
                    f"[{idx + 1}/{len(flacs)}] {n_done_this_run} files this run, "
                    f"{el:.0f}s elapsed, {el / n_done_this_run:.2f} s/file",
                    flush=True,
                )

    print(f"run complete: {n_done_this_run} new files in {time.perf_counter() - t_start:.0f}s")
    summarize(args.results)


if __name__ == "__main__":
    main()
