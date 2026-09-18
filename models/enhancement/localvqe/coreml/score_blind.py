#!/usr/bin/env python
"""Score enhanced renders of the ICASSP 2022 AEC-Challenge blind test set.

AECMOS echo / degradation MOS, blind ERLE (two definitions: the plain energy
ratio 10*log10(E[mic^2] / E[enh^2]) over the rated segment, and the LocalVQE
technical-report gated version pooled over 512/256 frames whose loopback RMS
is above its 75th percentile and whose mic RMS is below 2x the loopback RMS)
and DNSMOS P.835 OVRL on the enhanced clip (over the rated segment by default,
--dnsmos-region whole for the entire recording). Two AECMOS protocols:

  --protocol challenge (default): Run_1663915512_Stage_0.onnx (scenario
      marker) with the AECMOS README trimming rules — far-end single talk ->
      last half, double talk -> last (len-15)/2 s, near-end -> whole clip.
  --protocol upstream: what the LocalVQE README / HF model-card table was
      produced with — the legacy Run_1663829550_Stage_0.onnx (no scenario
      marker) over the whole clip, i.e. its first 20 s. Reproduces the
      published unprocessed baseline exactly and the model rows to within
      ~0.05 echo MOS when fed the GGML CLI's raw output.

    uv run --no-project --python 3.12 --with librosa --with onnxruntime --with soundfile --with scipy \
        python score_blind.py --blind-dir blind --enh-dir renders/coreml-v1.3 \
        --aecmos-dir aecmos --output scores-coreml-v1.3.json [--protocol upstream] [--no-dnsmos]

--enh-dir mirrors the blind layout: <scenario>/<stem>_enh.wav. Pass
--enh-dir unprocessed to score the raw mic (the upstream "unprocessed" row).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

SR = 16000
HOP = 256
SCENARIOS = {
    "doubletalk": "dt",
    "doubletalk-with-movement": "dt",
    "farend-singletalk": "st",
    "farend-singletalk-with-movement": "st",
    "nearend-singletalk": "nst",
}

_aecmos = None
_dnsmos = None
_p808 = None
_protocol = "challenge"
_run_dnsmos = True
_dnsmos_region = "rated"


def _init(aecmos_dir: str, protocol: str = "challenge", run_dnsmos: bool = True, dnsmos_region: str = "rated"):
    global _aecmos, _dnsmos, _p808, _protocol, _run_dnsmos, _dnsmos_region
    _protocol, _run_dnsmos, _dnsmos_region = protocol, run_dnsmos, dnsmos_region
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    model = "Run_1663829550_Stage_0.onnx" if protocol == "upstream" else "Run_1663915512_Stage_0.onnx"
    _aecmos = ort.InferenceSession(f"{aecmos_dir}/{model}", opts)
    if run_dnsmos:
        _dnsmos = ort.InferenceSession(f"{aecmos_dir}/sig_bak_ovr.onnx", opts)
        _p808 = ort.InferenceSession(f"{aecmos_dir}/model_v8.onnx", opts)


def _mel(sample: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=sample, sr=SR, n_fft=513, hop_length=256, n_mels=160)
    return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T


def aecmos(talk_type: str, lpb: np.ndarray, mic: np.ndarray, enh: np.ndarray) -> tuple[float, float]:
    n = min(len(lpb), len(mic), len(enh), 20 * SR)
    lpb, mic, enh = _mel(lpb[:n]), _mel(mic[:n]), _mel(enh[:n])
    if _protocol != "upstream":  # the legacy model takes no scenario marker
        ne_st, fe_st = {"nst": (1, 0), "st": (0, 1), "dt": (0, 0)}[talk_type]
        w = mic.shape[1]
        mic = np.concatenate((mic, np.ones((20, w)) * (1 - fe_st), np.zeros((20, w))), axis=0)
        lpb = np.concatenate((lpb, np.ones((20, w)) * (1 - ne_st), np.zeros((20, w))), axis=0)
        enh = np.concatenate((enh, np.ones((20, w)), np.zeros((20, w))), axis=0)
    feats = np.expand_dims(np.stack((lpb, mic, enh)).astype(np.float32), 0)
    h0 = np.zeros((4, 1, 64), dtype=np.float32)
    out = _aecmos.run([], {_aecmos.get_inputs()[0].name: feats, "h0": h0})[0]
    return float(out[0]), float(out[1])


def _dns_mel(audio, n_mels=120, frame_size=320, hop_length=160):
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels)
    return (librosa.power_to_db(mel, ref=np.max) + 40) / 40


def _polyfit(sig, bak, ovr):
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    return p_sig(sig), p_bak(bak), p_ovr(ovr)


def dnsmos(audio: np.ndarray) -> tuple[float, float, float, float]:
    """Mirrors microsoft/DNS-Challenge dnsmos_local.py (non-personalized)."""
    input_length = 9.01
    len_samples = int(input_length * SR)
    while len(audio) < len_samples:
        audio = np.append(audio, audio)
    num_hops = int(np.floor(len(audio) / SR) - input_length) + 1
    hop = SR
    sig, bak, ovr, p808 = [], [], [], []
    for idx in range(num_hops):
        seg = audio[int(idx * hop): int((idx + input_length) * hop)]
        if len(seg) < len_samples:
            continue
        oi = {"input_1": np.array(seg, dtype=np.float32)[np.newaxis, :]}
        p808_oi = {"input_1": np.array(_dns_mel(seg[:-160]).T, dtype=np.float32)[np.newaxis, :, :]}
        p808.append(_p808.run(None, p808_oi)[0][0][0])
        s, b, o = _dnsmos.run(None, oi)[0][0]
        s, b, o = _polyfit(s, b, o)
        sig.append(s)
        bak.append(b)
        ovr.append(o)
    return float(np.mean(sig)), float(np.mean(bak)), float(np.mean(ovr)), float(np.mean(p808))


def load(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def score_one(job):
    scenario, stem, mic_path, lpb_path, enh_path = job
    talk = SCENARIOS[scenario]
    mic, lpb = load(mic_path), load(lpb_path)
    enh = mic if enh_path is None else load(enh_path)
    n = min(len(mic), len(lpb), len(enh))
    mic, lpb, enh = mic[:n], lpb[:n], enh[:n]

    # Challenge trimming rules (AECMOS README): far-end single talk -> last
    # half; double talk -> last (len - 15) / 2 seconds; near-end -> whole clip.
    # The upstream-table protocol scores the whole clip (first 20 s).
    if _protocol == "upstream":
        start = 0
    elif talk == "st":
        start = n // 2
    elif talk == "dt":
        start = max(0, n - int(((n / SR) - 15) / 2 * SR))
    else:
        start = 0
    echo, deg = aecmos(talk, lpb[start:], mic[start:], enh[start:])
    seg_mic, seg_lpb, seg_enh = mic[start:], lpb[start:], enh[start:]
    erle = 10 * np.log10((np.mean(seg_mic**2) + 1e-12) / (np.mean(seg_enh**2) + 1e-12))
    erle_gated = gated_erle(seg_mic, seg_lpb, seg_enh)
    dns_input = enh[start:] if _dnsmos_region == "rated" else enh
    sig, bak, ovr, p808 = dnsmos(dns_input) if _run_dnsmos else (float("nan"),) * 4
    return {
        "scenario": scenario, "stem": stem, "echo": echo, "deg": deg, "erle": float(erle),
        "erle_gated": erle_gated, "sig": sig, "bak": bak, "ovrl": ovr, "p808": p808, "seconds": n / SR,
    }


def gated_erle(mic: np.ndarray, lpb: np.ndarray, enh: np.ndarray, frame: int = 512, hop: int = HOP) -> float:
    """LocalVQE technical report (sec. 4) blind ERLE: pool mic/enh energy over
    frames where loopback RMS exceeds its 75th percentile and mic RMS is
    below 2x loopback RMS (far-end-dominated frames). NaN if no frame qualifies."""
    n = min(len(mic), len(lpb), len(enh))
    if n < frame:
        return float("nan")
    idx = np.arange(0, n - frame + 1, hop)
    def rms(x):
        return np.sqrt(np.array([np.mean(x[i:i + frame] ** 2) for i in idx]) + 1e-20)
    r_lpb, r_mic = rms(lpb), rms(mic)
    gate = (r_lpb > np.percentile(r_lpb, 75)) & (r_mic < 2 * r_lpb)
    if not gate.any():
        return float("nan")
    e_mic = sum(np.sum(mic[i:i + frame] ** 2) for i in idx[gate])
    e_enh = sum(np.sum(enh[i:i + frame] ** 2) for i in idx[gate])
    return float(10 * np.log10((e_mic + 1e-12) / (e_enh + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-dir", required=True, type=Path)
    ap.add_argument("--enh-dir", required=True, help="render dir mirroring the blind layout, or 'unprocessed'")
    ap.add_argument("--aecmos-dir", required=True, type=str)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--protocol", choices=["challenge", "upstream"], default="challenge")
    ap.add_argument("--no-dnsmos", action="store_true", help="skip DNSMOS (much faster)")
    ap.add_argument("--dnsmos-region", choices=["rated", "whole"], default="rated",
                    help="score DNSMOS on the AECMOS-rated segment (default) or the whole recording")
    args = ap.parse_args()

    jobs = []
    # Blind layout: <dir>/<guid>_<scenario>_{mic,lpb}.flac; the scenario is
    # the filename suffix (with-movement variants share a directory).
    for mic in sorted(args.blind_dir.glob("*/*_mic.flac")):
        stem = mic.name[: -len("_mic.flac")]
        scenario = stem.split("_", 1)[1]
        if scenario not in SCENARIOS:
            continue
        lpb = mic.with_name(stem + "_lpb.flac")
        if not lpb.exists():
            continue
        rel = mic.parent.name
        enh = None if args.enh_dir == "unprocessed" else Path(args.enh_dir) / rel / f"{stem}_enh.wav"
        if enh is not None and not enh.exists():
            raise SystemExit(f"missing render: {enh}")
        jobs.append((scenario, stem, mic, lpb, enh))
    if args.limit:
        per = defaultdict(list)
        for j in jobs:
            per[j[0]].append(j)
        jobs = [j for s in SCENARIOS for j in per[s][: args.limit]]
    print(f"{len(jobs)} clips, {args.jobs} workers, enh={args.enh_dir}, protocol={args.protocol}")

    with mp.Pool(args.jobs, initializer=_init, initargs=(args.aecmos_dir, args.protocol, not args.no_dnsmos, args.dnsmos_region)) as pool:
        rows = pool.map(score_one, jobs, chunksize=4)

    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r)
    summary = {}
    print(f"{'scenario':34s} {'n':>4s} {'echo':>6s} {'deg':>6s} {'ERLE':>8s} {'gERLE':>8s} {'OVRL':>6s}")
    for scenario in SCENARIOS:
        rs = by_scenario[scenario]
        if not rs:
            continue
        m = {k: float(np.mean([r[k] for r in rs])) for k in ("echo", "deg", "erle", "ovrl", "sig", "bak", "p808")}
        gated = [r["erle_gated"] for r in rs if np.isfinite(r["erle_gated"])]
        m["erle_gated"] = float(np.mean(gated)) if gated else float("nan")
        m["erle_gated_n"] = len(gated)
        m["n"] = len(rs)
        summary[scenario] = m
        erle = f"{m['erle']:7.1f} dB" if SCENARIOS[scenario] == "st" else "       —"
        print(f"{scenario:34s} {m['n']:4d} {m['echo']:6.2f} {m['deg']:6.2f} {erle} {m['erle_gated']:6.1f}dB {m['ovrl']:6.2f}")
    if args.output:
        args.output.write_text(json.dumps(
            {"enh_dir": args.enh_dir, "protocol": args.protocol, "dnsmos_region": args.dnsmos_region,
             "summary": summary, "clips": rows}, indent=1))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
