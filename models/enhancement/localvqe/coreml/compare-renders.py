#!/usr/bin/env python
"""Aligned A/B of two blind-set render dirs (e.g. Core ML vs upstream GGML).

The upstream GGML CLI emits its output one hop (256 samples) late, its
whole-clip mode can truncate the tail and zero-fills the last partial hop,
and it writes 16-bit PCM, so raw score tables of the two engines are not
comparable clip by clip. This script drops --shift-b samples from the B
render, truncates both to the common whole-hop length, optionally rounds A
to 16-bit (--quantize-a) and scores AECMOS + blind ERLE on identical sample
regions, reporting per-scenario means for both and the per-clip delta
distribution.

    uv run python compare-renders.py --blind-dir blind --a renders/coreml-v1.3 \
        --b renders/ggml-v1.3 --shift-b 256 --aecmos-dir aecmos --label-a coreml --label-b ggml
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np

import score_blind as sb


def one(job):
    scenario, stem, mic_p, lpb_p, a_p, b_p, shift_b, quantize_a = job
    talk = sb.SCENARIOS[scenario]
    mic, lpb, a, b = sb.load(mic_p), sb.load(lpb_p), sb.load(a_p), sb.load(b_p)
    if quantize_a:  # emulate a 16-bit PCM WAV round trip (what the GGML CLI writes)
        a = np.clip(np.round(a * 32767.0), -32768, 32767) / 32767.0
    b = b[shift_b:]
    # Compare whole hops only: the GGML whole-clip mode zero-fills the trailing
    # partial hop instead of rendering it, which would otherwise dominate the
    # residual on near-silent outputs.
    n = (min(len(mic), len(lpb), len(a), len(b)) // sb.HOP) * sb.HOP
    mic, lpb, a, b = mic[:n], lpb[:n], a[:n], b[:n]
    start = {"st": n // 2, "dt": max(0, n - int(((n / sb.SR) - 15) / 2 * sb.SR)), "nst": 0}[talk]
    ea, da = sb.aecmos(talk, lpb[start:], mic[start:], a[start:])
    eb, db = sb.aecmos(talk, lpb[start:], mic[start:], b[start:])
    pm = np.mean(mic[start:] ** 2) + 1e-12
    erle_a = 10 * np.log10(pm / (np.mean(a[start:] ** 2) + 1e-12))
    erle_b = 10 * np.log10(pm / (np.mean(b[start:] ** 2) + 1e-12))
    e = a - b
    snr = 10 * np.log10((np.sum(b**2) + 1e-20) / (np.sum(e**2) + 1e-20))
    return {"scenario": scenario, "stem": stem, "echo_a": ea, "deg_a": da, "echo_b": eb, "deg_b": db,
            "erle_a": float(erle_a), "erle_b": float(erle_b), "snr_ab": float(snr), "maxdiff": float(np.abs(e).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-dir", required=True, type=Path)
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--shift-b", type=int, default=0)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--aecmos-dir", required=True)
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--quantize-a", action="store_true", help="round A to 16-bit PCM before scoring")
    ap.add_argument("--worst", type=int, default=0, help="list the N clips with the lowest A/B waveform SNR")
    args = ap.parse_args()

    jobs = []
    for mic in sorted(args.blind_dir.glob("*/*_mic.flac")):
        stem = mic.name[: -len("_mic.flac")]
        scenario = stem.split("_", 1)[1]
        if scenario not in sb.SCENARIOS:
            continue
        rel = mic.parent.name
        jobs.append((scenario, stem, mic, mic.with_name(stem + "_lpb.flac"),
                     args.a / rel / f"{stem}_enh.wav", args.b / rel / f"{stem}_enh.wav", args.shift_b, args.quantize_a))
    with mp.Pool(args.jobs, initializer=sb._init, initargs=(args.aecmos_dir,)) as pool:
        rows = pool.map(one, jobs, chunksize=4)

    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)
    la, lb = args.label_a, args.label_b
    print(f"{'scenario':34s} {'n':>4s} {'echo '+la:>12s} {'echo '+lb:>12s} {'deg '+la:>10s} {'deg '+lb:>10s} {'ERLE '+la:>10s} {'ERLE '+lb:>10s}")
    for s in sb.SCENARIOS:
        rs = by[s]
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"{s:34s} {len(rs):4d} {m('echo_a'):12.2f} {m('echo_b'):12.2f} {m('deg_a'):10.2f} {m('deg_b'):10.2f} {m('erle_a'):10.1f} {m('erle_b'):10.1f}")
    de = np.array([r["echo_a"] - r["echo_b"] for r in rows])
    dd = np.array([r["deg_a"] - r["deg_b"] for r in rows])
    dr = np.array([r["erle_a"] - r["erle_b"] for r in rows])
    snr = np.array([r["snr_ab"] for r in rows])
    md = np.array([r["maxdiff"] for r in rows])
    print(f"per-clip delta ({la} - {lb}) echo: mean {de.mean():+.4f}  p95|d| {np.percentile(np.abs(de),95):.4f}  max|d| {np.abs(de).max():.4f}")
    print(f"per-clip delta ({la} - {lb}) deg : mean {dd.mean():+.4f}  p95|d| {np.percentile(np.abs(dd),95):.4f}  max|d| {np.abs(dd).max():.4f}")
    print(f"per-clip delta ({la} - {lb}) ERLE: mean {dr.mean():+.4f} dB  p95|d| {np.percentile(np.abs(dr),95):.3f}  max|d| {np.abs(dr).max():.3f}")
    print(f"waveform {la} vs {lb} (aligned): median SNR {np.median(snr):.1f} dB, min {snr.min():.1f} dB, max abs diff {md.max():.2e}")
    for r in sorted(rows, key=lambda r: r["snr_ab"])[: args.worst]:
        print(f"  worst: {r['stem']} snr={r['snr_ab']:.1f} dB maxdiff={r['maxdiff']:.2e} echo {la}={r['echo_a']:.2f} {lb}={r['echo_b']:.2f}")


if __name__ == "__main__":
    main()
