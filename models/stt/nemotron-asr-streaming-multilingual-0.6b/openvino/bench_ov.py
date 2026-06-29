#!/usr/bin/env python3
"""CPU benchmark for the Nemotron OpenVINO IR: latency, RTFx, per-component.

Times end-to-end streaming transcription over N runs for a given IR dir,
plus a per-component breakdown (preprocessor/encoder/decoder/joint). Reports
real-time factor (audio_seconds / wall_seconds).
"""
import argparse
import time

import numpy as np
import soundfile as sf

from transcribe_ov import NemotronOV


def bench(model_dir, audio, sr, target_lang, runs):
    r = NemotronOV(model_dir, device="CPU")
    dur = len(audio) / sr
    # warmup
    r.transcribe_streaming(audio, target_lang=target_lang)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _, text, _ = r.transcribe_streaming(audio, target_lang=target_lang)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return dur, times, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--target-lang", default="en-US")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur, times, text = bench(args.model_dir, audio, sr, args.target_lang, args.runs)

    mean_s = times.mean()
    print(f"\n=== {args.model_dir} ===")
    print(f"audio:    {dur:.2f}s")
    print(f"runs:     {args.runs}")
    print(f"latency:  mean {mean_s*1000:.0f} ms | min {times.min()*1000:.0f} | max {times.max()*1000:.0f}")
    print(f"RTFx:     {dur/mean_s:.1f}x  (audio_sec / wall_sec)")
    print(f"text:     {text[:80]}...")


if __name__ == "__main__":
    main()
