#!/usr/bin/env python
"""Score enhanced renders of the ICASSP 2022 AEC-Challenge blind test set.

AECMOS echo / degradation MOS, blind ERLE (two definitions: the plain energy
ratio 10*log10(E[mic^2] / E[enh^2]) over the rated segment, and the LocalVQE
technical-report gated version pooled over 512/256 frames whose loopback RMS
is above its 75th percentile and whose mic RMS is below 2x the loopback RMS)
and DNSMOS P.835 OVRL on the enhanced clip (over the challenge-rated segment
by default, whichever AECMOS protocol is selected; --dnsmos-region whole for
the entire recording). Two AECMOS protocols:

  --protocol challenge (default): Run_1663915512_Stage_0.onnx (scenario
      marker) with the AECMOS README trimming rules — far-end single talk ->
      last half, double talk -> last (len-15)/2 s, near-end -> whole clip.
  --protocol upstream: what the LocalVQE README / HF model-card table was
      produced with — the legacy Run_1663829550_Stage_0.onnx (no scenario
      marker) over the whole clip, i.e. its first 20 s. Reproduces the
      published unprocessed baseline exactly. The v1.2 model table and
      the card's ERLE definition remain unresolved (see REPRODUCTION.md).

    uv run --no-project --python 3.12 --with librosa --with onnxruntime --with soundfile --with scipy \
        python score_blind.py --blind-dir blind --enh-dir renders/coreml-v1.3 \
        --aecmos-dir aecmos --output scores-coreml-v1.3.json [--protocol upstream] [--no-dnsmos]

--enh-dir mirrors the blind layout: <scenario>/<stem>_enh.wav. Pass
--enh-dir unprocessed to score the raw mic (the upstream "unprocessed" row).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from localvqe_coreml.benchmark_manifest import collect_jobs, require_coverage

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
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("DNSMOS requires nonempty, finite audio")
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
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    return audio


def score_one(job):
    scenario, stem, mic_path, lpb_path, enh_path = job
    talk = SCENARIOS[scenario]
    mic, lpb = load(mic_path), load(lpb_path)
    enh = mic if enh_path is None else load(enh_path)
    n = min(len(mic), len(lpb), len(enh))
    # Blind mic/loopback files legitimately have unequal tails. Score their
    # common overlap, allowing the renderer to discard fewer than one hop.
    # Reject enhanced files that truncate the common interval further.
    if min(len(mic), len(lpb)) - len(enh) >= HOP or len(enh) - len(mic) >= HOP:
        raise ValueError(f"Audio length mismatch for {stem}: {len(mic)}, {len(lpb)}, {len(enh)}")
    input_samples = {"mic": len(mic), "lpb": len(lpb), "enhanced": len(enh), "scored": n}
    mic, lpb, enh = mic[:n], lpb[:n], enh[:n]

    # Challenge-rated segment (AECMOS README): far-end single talk -> last
    # half; double talk -> last (len - 15) / 2 seconds; near-end -> whole clip.
    if talk == "st":
        rated_start = n // 2
    elif talk == "dt":
        rated_start = max(0, n - int(((n / SR) - 15) / 2 * SR))
    else:
        rated_start = 0
    # AECMOS/ERLE segment: the rated segment, or the whole clip (first 20 s
    # via the model's cap) under the upstream-table protocol. DNSMOS "rated"
    # always means the challenge-rated segment, independent of the protocol.
    start = 0 if _protocol == "upstream" else rated_start
    echo, deg = aecmos(talk, lpb[start:], mic[start:], enh[start:])
    seg_mic, seg_lpb, seg_enh = mic[start:], lpb[start:], enh[start:]
    erle = 10 * np.log10((np.mean(seg_mic**2) + 1e-12) / (np.mean(seg_enh**2) + 1e-12))
    erle_gated = gated_erle(seg_mic, seg_lpb, seg_enh)
    dns_input = enh[rated_start:] if _dnsmos_region == "rated" else enh
    sig, bak, ovr, p808 = dnsmos(dns_input) if _run_dnsmos else (float("nan"),) * 4
    return {
        "scenario": scenario, "stem": stem, "echo": echo, "deg": deg, "erle": float(erle),
        "erle_gated": erle_gated, "sig": sig, "bak": bak, "ovrl": ovr, "p808": p808, "seconds": n / SR,
        "input_samples": input_samples,
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
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).resolve().parent / "validation/blind-manifest.txt",
                    help="Exact input stem set (default: committed 800-clip manifest)")
    ap.add_argument("--protocol", choices=["challenge", "upstream"], default="challenge")
    ap.add_argument("--no-dnsmos", action="store_true", help="skip DNSMOS (much faster)")
    ap.add_argument("--dnsmos-region", choices=["rated", "whole"], default="rated",
                    help="score DNSMOS on the AECMOS-rated segment (default) or the whole recording")
    args = ap.parse_args()

    try:
        jobs, expected = collect_jobs(args.blind_dir, args.enh_dir, args.manifest, SCENARIOS, args.limit)
        if args.jobs <= 0:
            raise ValueError("--jobs must be positive")
    except ValueError as error:
        ap.error(str(error))
    print(f"{len(jobs)} clips, {args.jobs} workers, enh={args.enh_dir}, protocol={args.protocol}")

    model_names = ["Run_1663829550_Stage_0.onnx" if args.protocol == "upstream" else "Run_1663915512_Stage_0.onnx"]
    if not args.no_dnsmos:
        model_names += ["sig_bak_ovr.onnx", "model_v8.onnx"]
    model_hashes = {name: hashlib.sha256((Path(args.aecmos_dir) / name).read_bytes()).hexdigest()
                    for name in model_names}
    # Validate model loading in the parent. Pool initializer failures otherwise
    # keep respawning workers instead of returning a useful benchmark failure.
    _init(args.aecmos_dir, args.protocol, not args.no_dnsmos, args.dnsmos_region)
    with mp.Pool(args.jobs, initializer=_init, initargs=(args.aecmos_dir, args.protocol, not args.no_dnsmos, args.dnsmos_region)) as pool:
        rows = pool.map(score_one, jobs, chunksize=4)

    require_coverage([row["stem"] for row in rows], [job[1] for job in jobs])
    for row in rows:
        required = ["echo", "deg", "erle", "seconds"]
        if not args.no_dnsmos:
            required += ["sig", "bak", "ovrl", "p808"]
        for metric in required:
            if not np.isfinite(row[metric]):
                raise ValueError(f"Non-finite {metric} for {row['stem']}")

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
        # JSON null represents deliberately unscored metrics / ineligible ERLE,
        # never a non-standard NaN token in a supposedly valid JSON report.
        def finite_values(value):
            if isinstance(value, dict):
                return {key: finite_values(item) for key, item in value.items()}
            if isinstance(value, list):
                return [finite_values(item) for item in value]
            return None if isinstance(value, float) and not np.isfinite(value) else value

        args.output.write_text(json.dumps(
            finite_values({"enh_dir": args.enh_dir, "protocol": args.protocol, "dnsmos_region": args.dnsmos_region,
                           "dnsmos_scored": not args.no_dnsmos,
                           "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                           "manifest_count": len(expected), "selected_stems": [job[1] for job in jobs],
                           "complete_manifest": len(jobs) == len(expected),
                           "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "metric_models_sha256": model_hashes,
                           "environment": {"python": sys.version,
                                           "packages": {name: version(name) for name in
                                                        ("numpy", "scipy", "librosa", "soundfile", "onnxruntime")}},
                           "summary": summary, "clips": rows}), indent=1, allow_nan=False))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
