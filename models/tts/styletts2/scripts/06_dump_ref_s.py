"""Dump 256-fp32 ref_s.bin voice blobs from reference WAV(s).

Output format (matches `StyleTTS2VoiceStyle.load`):
  256 little-endian float32, 1024 bytes total. Layout:
    [0:128]   predictor_encoder(mel)   — prosody / duration branch
    [128:256] style_encoder(mel)       — acoustic / decoder branch

Note: the FluidAudio Swift accessors `voice.acoustic` (first 128) and
`voice.prosody` (last 128) have historically inverted names: the first 128
floats are actually the **prosody** branch (fed to text_predictor +
f0n_energy), and the last 128 are the **acoustic** branch (fed to decoder).
The byte layout is the source of truth — the property names are legacy.

The `99_parity_check.compute_ref_s` helper uses the opposite concat order
(`[ref_s, ref_p]`) for in-Python parity sanity checks against upstream
PyTorch — that ordering is **not** the on-disk voice format.

Usage:
  # Single WAV
  python 06_dump_ref_s.py /path/to/voice.wav --output ref_s_voice.bin

  # Batch a directory (writes ref_s_<stem>.bin for every *.wav)
  python 06_dump_ref_s.py /path/to/wav_dir/ --output-dir voices/

  # Use a custom checkpoint (defaults to LibriTTS-trained 2nd-stage)
  python 06_dump_ref_s.py voice.wav --checkpoint /path/to/epoch_2nd.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _styletts2_lib import DEFAULT_CHECKPOINT, load_inference_modules


def _load_wav_24k(path: Path) -> np.ndarray:
    """Load WAV → mono float32 @ 24 kHz, with -30 dB silence trim.

    Falls back to soundfile + scipy.signal if librosa is not available
    (the `mobius` venv ships librosa; the system Python may not).
    """
    try:
        import librosa  # type: ignore

        wave, sr = librosa.load(str(path), sr=24000)
        wave, _ = librosa.effects.trim(wave, top_db=30)
        return wave.astype(np.float32)
    except ImportError:
        import soundfile as sf
        import scipy.signal as sps

        wave, sr = sf.read(str(path), dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if sr != 24000:
            g = np.gcd(sr, 24000)
            wave = sps.resample_poly(wave, 24000 // g, sr // g).astype(np.float32)
        # -30 dB peak-relative trim to mirror librosa.effects.trim(top_db=30).
        abs_w = np.abs(wave)
        if abs_w.size:
            thresh = abs_w.max() * 10 ** (-30 / 20)
            mask = abs_w > thresh
            if mask.any():
                lo = int(np.argmax(mask))
                hi = int(len(mask) - np.argmax(mask[::-1]))
                wave = wave[lo:hi]
        return wave


def _build_mel():
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_mels=80,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
    )


def extract_ref_s(modules, wav_path: Path, mel_transform) -> tuple[np.ndarray, float]:
    """Compute the 256-fp32 [ref_p, ref_s] blob for a single WAV.

    Returns (blob, duration_seconds).
    """
    wave = _load_wav_24k(wav_path)
    duration = wave.shape[0] / 24000.0

    mel = mel_transform(torch.from_numpy(wave).float())
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - (-4.0)) / 4.0  # (1, 80, T)

    with torch.no_grad():
        ref_s = modules["style_encoder"](mel.unsqueeze(1))      # (1, 128) acoustic
        ref_p = modules["predictor_encoder"](mel.unsqueeze(1))  # (1, 128) prosody

    # On-disk layout: prosody first, acoustic second.
    blob = torch.cat([ref_p, ref_s], dim=1).squeeze(0).numpy().astype(np.float32)
    if blob.shape != (256,):
        raise RuntimeError(f"unexpected blob shape {blob.shape} for {wav_path}")
    return blob, duration


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="WAV file or directory of WAVs")
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Output .bin path (single-WAV mode). Default: ref_s_<stem>.bin",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (batch mode). Required when input is a directory.",
    )
    ap.add_argument(
        "--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT),
        help=f"StyleTTS2 2nd-stage checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    args = ap.parse_args()

    modules, _cfg = load_inference_modules(args.checkpoint)
    mel_transform = _build_mel()

    if args.input.is_file():
        out = args.output or Path(f"ref_s_{args.input.stem}.bin")
        out.parent.mkdir(parents=True, exist_ok=True)
        blob, dur = extract_ref_s(modules, args.input, mel_transform)
        blob.tofile(str(out))
        rms = float(np.sqrt(np.mean(blob ** 2)))
        print(f"  {args.input.stem:<40} dur={dur:5.2f}s  rms={rms:.4f}  → {out}")
        return

    if not args.input.is_dir():
        ap.error(f"input not found: {args.input}")
    if args.output_dir is None:
        ap.error("--output-dir is required when input is a directory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(args.input.glob("*.wav"))
    if not wavs:
        ap.error(f"no .wav files in {args.input}")

    for wav in wavs:
        name = wav.stem.replace("/", "_")
        blob, dur = extract_ref_s(modules, wav, mel_transform)
        out = args.output_dir / f"ref_s_{name}.bin"
        blob.tofile(str(out))
        rms = float(np.sqrt(np.mean(blob ** 2)))
        print(f"  {name:<40} dur={dur:5.2f}s  rms={rms:.4f}  → {out.name}")


if __name__ == "__main__":
    main()
