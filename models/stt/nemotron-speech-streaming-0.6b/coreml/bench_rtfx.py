#!/usr/bin/env python3
"""WER + RTFx benchmark for converted Nemotron streaming CoreML tiers.

Reuses the streaming decode loop from test_coreml_streaming.py but loads the
models on CPU_AND_NE (the ship config) and times wall-clock to report RTFx.
"""
import argparse
import glob
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

from test_coreml_streaming import NemotronCoreMLStreaming, load_ground_truth, compute_wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-files", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    inf = NemotronCoreMLStreaming(args.model_dir)
    # Reload on CPU_AND_NE (ship config); the harness defaults to ALL.
    md = Path(args.model_dir)
    cu = ct.ComputeUnit.CPU_AND_NE
    inf.preprocessor = ct.models.MLModel(str(md / "preprocessor.mlpackage"), compute_units=cu)
    inf.encoder = ct.models.MLModel(str(md / "encoder.mlpackage"), compute_units=cu)
    inf.decoder = ct.models.MLModel(str(md / "decoder.mlpackage"), compute_units=cu)
    inf.joint = ct.models.MLModel(str(md / "joint.mlpackage"), compute_units=cu)

    gt = load_ground_truth(args.dataset)
    files = sorted(glob.glob(f"{args.dataset}/**/*.flac", recursive=True))[: args.num_files + args.warmup]

    total_err = total_words = 0
    audio_s = 0.0
    compute_s = 0.0
    n = 0
    for i, path in enumerate(files):
        fid = Path(path).stem
        audio, sr = sf.read(path, dtype="float32")
        t0 = time.perf_counter()
        hyp = inf.transcribe_streaming(audio)
        dt = time.perf_counter() - t0
        if i < args.warmup:
            continue
        n += 1
        audio_s += len(audio) / sr
        compute_s += dt
        if fid in gt:
            e, w = compute_wer(gt[fid], hyp)
            total_err += e
            total_words += w
        if n % 20 == 0:
            print(f"  {n} files | WER {100*total_err/max(total_words,1):.2f}% | RTFx {audio_s/compute_s:.1f}")

    wer = 100 * total_err / total_words if total_words else 0.0
    print("=" * 60)
    print(f"model-dir : {args.model_dir}")
    print(f"chunk_mel : {inf.chunk_mel_frames} ({inf.chunk_mel_frames*10}ms)")
    print(f"files     : {n}")
    print(f"WER       : {wer:.2f}%")
    print(f"audio_s   : {audio_s:.1f}  compute_s : {compute_s:.1f}")
    print(f"RTFx      : {audio_s/compute_s:.1f}")


if __name__ == "__main__":
    main()
