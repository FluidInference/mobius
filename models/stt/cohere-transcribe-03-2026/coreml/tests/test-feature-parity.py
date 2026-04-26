"""Parity test: numpy CohereMelSpectrogram vs HF CohereAsrFeatureExtractor.

Loads the real HF feature extractor (trust_remote_code) and runs it
alongside the numpy port on a handful of real audio samples across
languages and lengths. Reports max/mean absolute diff in the mel output
and exits non-zero if parity is not within tolerance.

The HF reference is fetched on demand from CohereLabs/cohere-transcribe-03-2026.
Override with --pytorch-dir to point at a local snapshot, e.g.

    huggingface-cli download CohereLabs/cohere-transcribe-03-2026 \\
        --local-dir ../cohere-pytorch
    uv run python tests/test-feature-parity.py --pytorch-dir ../cohere-pytorch

Usage:
    uv run python tests/test-feature-parity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from cohere_features_v2 import CohereMelSpectrogram as CohereMelV2  # noqa: E402

HF_REPO = "CohereLabs/cohere-transcribe-03-2026"


def load_hf_extractor(pytorch_dir: Path | None = None):
    """Load the official Cohere HF feature extractor.

    Uses pytorch_dir if it exists, otherwise pulls directly from HF.
    """
    from transformers import AutoFeatureExtractor

    if pytorch_dir is not None and pytorch_dir.is_dir():
        source = str(pytorch_dir)
    else:
        source = HF_REPO

    fe = AutoFeatureExtractor.from_pretrained(source, trust_remote_code=True)
    return fe


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # For parity we expect the sample files to already be 16kHz.
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def run_hf(fe, audio: np.ndarray):
    """Return (mel, length) from HF extractor as numpy float32."""
    out = fe(audio, sampling_rate=16000, return_tensors="pt")
    mel = out["input_features"].float().cpu().numpy()
    length = int(out["length"].cpu().numpy().item())
    return mel, length


def compare(label: str, a: np.ndarray, b: np.ndarray, tol_max: float, tol_mean: float):
    min_t = min(a.shape[-1], b.shape[-1])
    a_ = a[..., :min_t]
    b_ = b[..., :min_t]
    diff = np.abs(a_ - b_)
    dmax = float(diff.max())
    dmean = float(diff.mean())
    rmsa = float(np.sqrt((a_ ** 2).mean()))
    rmsb = float(np.sqrt((b_ ** 2).mean()))
    print(f"  {label:30s} shape_a={a.shape} shape_b={b.shape} "
          f"max_abs={dmax:.4e} mean_abs={dmean:.4e} rms_a={rmsa:.3f} rms_b={rmsb:.3f}")
    ok_max = dmax <= tol_max
    ok_mean = dmean <= tol_mean
    status = "PASS" if (ok_max and ok_mean) else "FAIL"
    print(f"    -> tol_max={tol_max:.1e} tol_mean={tol_mean:.1e} [{status}]")
    return ok_max and ok_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pytorch-dir",
        type=Path,
        default=None,
        help=f"Local snapshot of {HF_REPO}. Defaults to downloading from HF.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Feature Extractor Parity Test")
    print("=" * 72)

    src = str(args.pytorch_dir) if args.pytorch_dir and args.pytorch_dir.is_dir() else HF_REPO
    print(f"\n[1/3] Loading HF CohereAsrFeatureExtractor from {src}...")
    fe_hf = load_hf_extractor(args.pytorch_dir)
    print(f"  sampling_rate={fe_hf.sampling_rate}")
    print(f"  hop_length={fe_hf.hop_length}")

    print("\n[2/3] Loading CohereMelSpectrogram (numpy port v2)...")
    mel_v2 = CohereMelV2()
    print(f"  n_fft={mel_v2.n_fft} win_length={mel_v2.win_length} "
          f"hop={mel_v2.hop_length} n_mels={mel_v2.n_mels}")

    # Collect a few test samples across languages and durations.
    candidates = [
        ROOT / "fleurs_samples/en_us/sample_0000.wav",
        ROOT / "fleurs_samples/fr_fr/sample_0000.wav",
        ROOT / "fleurs_samples/es_419/sample_0000.wav",
        ROOT / "fleurs_samples/cmn_hans_cn/sample_0000.wav",
        ROOT / "librispeech_test_samples/sample_00.wav",
    ]
    samples = [p for p in candidates if p.exists()]
    if not samples:
        print("ERROR: no test audio samples found")
        sys.exit(1)

    print(f"\n[3/3] Running parity across {len(samples)} samples...")

    all_v2_ok = True
    for wav in samples:
        audio = load_audio(wav)
        duration = len(audio) / 16000
        print(f"\n  sample: {wav.name}  ({duration:.2f}s, {len(audio)} samples)")

        hf_mel, hf_len = run_hf(fe_hf, audio)
        v2_mel, v2_len = mel_v2(audio)

        print(f"    HF  length={hf_len:5d}  mel_shape={hf_mel.shape}")
        print(f"    v2  length={v2_len:5d}  mel_shape={v2_mel.shape}")

        ok = compare("HF vs v2 (full, unaligned)", hf_mel, v2_mel,
                     tol_max=1e-2, tol_mean=1e-3)
        all_v2_ok &= ok

        # Focus on valid region only (first hf_len frames).
        # HF applies pad_to=16 which can inflate the length; trim.
        ok_valid = compare(
            "HF vs v2 (valid frames only)",
            hf_mel[:, :, :hf_len],
            v2_mel[:, :, :hf_len],
            tol_max=5e-2,
            tol_mean=5e-3,
        )
        all_v2_ok &= ok_valid

    print("\n" + "=" * 72)
    if all_v2_ok:
        print("RESULT: v2 extractor matches HF reference within tolerance.")
        sys.exit(0)
    else:
        print("RESULT: v2 extractor DOES NOT match HF reference. Needs fixes.")
        sys.exit(2)


if __name__ == "__main__":
    main()
