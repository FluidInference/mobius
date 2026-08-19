"""Energy/dB similarity between the CoreML-pipeline wav and the PyTorch oracle wav.

Reports level (dBFS), frame-energy envelope agreement, per-band energy deltas,
and spectral distances. Waveform-phase-insensitive by design, since the flow
decoder's fp16 noise shifts vocoder phase without perceptual effect.

Usage:
    .venv/bin/python -m coreml.audio_similarity \
        --ref build/oracle/reference_48k.wav --test build/coreml/parity_48k.wav
"""

import argparse

import numpy as np
import soundfile as sf

EPS = 1e-12


def db(x):
    return 10.0 * np.log10(np.maximum(x, EPS))


def frame_rms(x, frame, hop):
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt((x[idx] ** 2).mean(axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="build/oracle/reference_48k.wav")
    parser.add_argument("--test", default="build/coreml/parity_48k.wav")
    parser.add_argument("--sr", type=int, default=48000)
    args = parser.parse_args()

    ref, sr = sf.read(args.ref)
    test, sr2 = sf.read(args.test)
    assert sr == sr2 == args.sr
    n = min(len(ref), len(test))
    ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)
    dur = n / sr

    print(f"samples={n} ({dur:.3f}s @ {sr}Hz)")

    # ---- overall level ----
    rms_ref, rms_test = np.sqrt((ref**2).mean()), np.sqrt((test**2).mean())
    print("\n== level ==")
    print(f"RMS   ref {20*np.log10(rms_ref):+7.2f} dBFS | coreml {20*np.log10(rms_test):+7.2f} dBFS "
          f"| delta {20*np.log10(rms_test/rms_ref):+.3f} dB")
    print(f"peak  ref {20*np.log10(np.abs(ref).max()):+7.2f} dBFS | coreml {20*np.log10(np.abs(test).max()):+7.2f} dBFS "
          f"| delta {20*np.log10(np.abs(test).max()/np.abs(ref).max()):+.3f} dB")
    print(f"total energy ratio {db((test**2).sum()) - db((ref**2).sum()):+.3f} dB")

    # ---- short-time energy envelope (25 ms / 10 ms) ----
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    er, et = frame_rms(ref, frame, hop), frame_rms(test, frame, hop)
    env_corr = np.corrcoef(er, et)[0, 1]
    dr, dt = 20 * np.log10(np.maximum(er, EPS)), 20 * np.log10(np.maximum(et, EPS))
    active = dr > (dr.max() - 40)  # ignore silence tails below -40 dB rel
    diff_db = dt[active] - dr[active]
    print("\n== short-time energy envelope (25ms/10ms) ==")
    print(f"frames={len(er)} active={active.sum()}")
    print(f"envelope corr (linear RMS)     {env_corr:.5f}")
    print(f"per-frame level diff (active): mean {diff_db.mean():+.3f} dB | median {np.median(diff_db):+.3f} dB "
          f"| std {diff_db.std():.3f} dB | max|.| {np.abs(diff_db).max():.3f} dB")
    print(f"envelope SNR: {db((er**2).sum()) - db(((er-et)**2).sum()):.2f} dB")

    # ---- per-band energy (octave bands) ----
    spec_r = np.abs(np.fft.rfft(ref)) ** 2
    spec_t = np.abs(np.fft.rfft(test)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    print("\n== per-band energy delta (coreml - ref) ==")
    edges = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000, 24000]
    lo = 20
    for hi in edges:
        m = (freqs >= lo) & (freqs < hi)
        delta = db(spec_t[m].sum()) - db(spec_r[m].sum())
        print(f"  {lo:>5}-{hi:<5} Hz: {delta:+6.2f} dB")
        lo = hi

    # ---- spectral distances (STFT 1024/256, magnitude) ----
    win = np.hanning(1024)
    def stft_mag(x):
        nf = 1 + (len(x) - 1024) // 256
        idx = np.arange(1024)[None, :] + 256 * np.arange(nf)[:, None]
        return np.abs(np.fft.rfft(x[idx] * win, axis=1))
    mr, mt = stft_mag(ref), stft_mag(test)
    sc = np.linalg.norm(mr - mt) / np.linalg.norm(mr)
    lsd = np.sqrt(((db(mt**2) - db(mr**2)) ** 2).mean(axis=1)).mean()
    print("\n== spectral ==")
    print(f"spectral convergence {sc:.4f}  (lower=better; <0.15 is near-identical)")
    print(f"log-spectral distance {lsd:.3f} dB (mean over frames)")


if __name__ == "__main__":
    main()
