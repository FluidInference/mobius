"""End-to-end audio A/B: fused 8-step VE vs the 8-call host loop.

Tensor parity on random inputs overstates fp16 divergence for a flow-matching
ODE (trials.md: trajectory divergence, not degradation). The honest yardstick
is the rendered audio. This renders the SAME utterance (same seeded noise,
same text_emb/duration) through:
  (a) single-step VE looped 8x (the shipped host path), and
  (b) the fused 8-step VE (1 call),
then vocodes both and reports waveform + mag-STFT SNR and writes wavs for
listening.

Usage:
    python3.11 -m coreml.e2e_fused_ab \
        --shared-dir .../build/_pipeline_shared \
        --single .../VectorEstimator_L128_int4.mlmodelc \
        --fused build/_mlpackage_ve_fused/VectorEstimator_L128_fused8_palette4.mlpackage \
        --tts-json ~/.cache/.../tts.json --unicode-indexer ~/.cache/.../unicode_indexer.json \
        --voice-style ~/.cache/.../voice_styles/M1.json \
        --text "..." --out-dir build/e2e_fused_ab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np

from .infer import UnicodeProcessor, load_voice_style, sample_noisy_latent

L_BUCKET = 128
T_FIXED = 128


def _load(path: Path, units=ct.ComputeUnit.CPU_AND_NE):
    if path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(path), compute_units=units)
    return ct.models.MLModel(str(path), compute_units=units)


def _pad_last(arr: np.ndarray, target: int) -> np.ndarray:
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (0, target - arr.shape[-1])
    return np.pad(arr, pad, constant_values=0.0)


def _stft_mag(x: np.ndarray, n_fft=1024, hop=256) -> np.ndarray:
    n_frames = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft)
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n_frames)])
    return np.abs(np.fft.rfft(frames, axis=-1))


def _snr_db(ref: np.ndarray, x: np.ndarray) -> float:
    n = min(len(ref), len(x))
    ref, x = ref[:n].astype(np.float64), x[:n].astype(np.float64)
    return 10 * np.log10((ref ** 2).mean() / max(((x - ref) ** 2).mean(), 1e-30))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shared-dir", type=Path, required=True)
    p.add_argument("--single", type=Path, required=True)
    p.add_argument("--fused", type=Path, required=True)
    p.add_argument("--tts-json", type=Path, required=True)
    p.add_argument("--unicode-indexer", type=Path, required=True)
    p.add_argument("--voice-style", type=Path, required=True)
    p.add_argument("--text", type=str,
                   default="The quick brown fox jumps over the lazy dog near the riverbank.")
    p.add_argument("--lang", type=str, default="en")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--speed", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("build/e2e_fused_ab"))
    args = p.parse_args()

    cfg = json.loads(args.tts_json.read_text())
    sample_rate = int(cfg["ae"]["sample_rate"])

    proc = UnicodeProcessor(args.unicode_indexer)
    style = load_voice_style([args.voice_style])

    te = _load(args.shared_dir / "TextEncoder.mlpackage")
    dp = _load(args.shared_dir / "DurationPredictor.mlpackage")
    voc = _load(args.shared_dir / "Vocoder.mlpackage")
    single = _load(args.single)
    fused = _load(args.fused)

    ids, mask = proc([args.text], [args.lang])
    T = ids.shape[1]
    if T > T_FIXED:
        raise SystemExit(f"text too long for one chunk: T={T}")
    ids = np.pad(ids, ((0, 0), (0, T_FIXED - T)), constant_values=0).astype(np.int32)
    mask = np.pad(mask, ((0, 0), (0, 0), (0, T_FIXED - T)), constant_values=0.0).astype(np.float32)

    dur = np.asarray(dp.predict({
        "text_ids": ids, "style_dp": style.dp.astype(np.float32), "text_mask": mask,
    })["duration"], dtype=np.float32) / args.speed
    text_emb = np.asarray(te.predict({
        "text_ids": ids, "style_ttl": style.ttl.astype(np.float32), "text_mask": mask,
    })["text_emb"], dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    noisy, latent_mask = sample_noisy_latent(
        dur,
        sample_rate=sample_rate,
        base_chunk_size=int(cfg["ae"]["base_chunk_size"]),
        chunk_compress_factor=int(cfg["ttl"]["chunk_compress_factor"]),
        latent_dim=int(cfg["ttl"]["latent_dim"]),
        rng=rng,
    )
    L_true = noisy.shape[-1]
    if L_true > L_BUCKET:
        raise SystemExit(f"latent too long for the L128 bucket: L={L_true}")
    noisy = _pad_last(noisy, L_BUCKET).astype(np.float32)
    latent_mask = _pad_last(latent_mask, L_BUCKET).astype(np.float32)
    feeds = {
        "text_emb": text_emb, "style_ttl": style.ttl.astype(np.float32),
        "latent_mask": latent_mask, "text_mask": mask,
    }
    print(f"T={T} tokens, L_true={L_true} (padded to {L_BUCKET}), duration={float(dur[0]):.2f}s")

    # (a) 8-call host loop.
    xt = noisy
    total = np.full((1,), float(args.steps), dtype=np.float32)
    for k in range(args.steps):
        xt = np.asarray(single.predict({
            **feeds, "noisy_latent": xt,
            "current_step": np.full((1,), float(k), dtype=np.float32),
            "total_step": total,
        })["denoised_latent"], dtype=np.float32)
    lat_loop = xt

    # (b) fused single call.
    lat_fused = np.asarray(
        fused.predict({**feeds, "noisy_latent": noisy})["denoised_latent"], dtype=np.float32)

    diff = (lat_fused - lat_loop).astype(np.float64)
    print(f"final latent: max_abs={np.abs(diff).max():.3e} mean_abs={np.abs(diff).mean():.3e} "
          f"latent_SNR={10 * np.log10((lat_loop.astype(np.float64) ** 2).mean() / (diff ** 2).mean()):.1f} dB")

    n_samples = (512 * int(cfg["ttl"]["chunk_compress_factor"])) * L_true
    wavs = {}
    for name, lat in (("loop", lat_loop), ("fused", lat_fused)):
        w = np.asarray(voc.predict({"latent": lat[..., :L_true]})["wav"], dtype=np.float32)
        wavs[name] = w[0, :n_samples]

    wav_snr = _snr_db(wavs["loop"], wavs["fused"])
    m_ref, m_x = _stft_mag(wavs["loop"]), _stft_mag(wavs["fused"])
    mag_snr = 10 * np.log10((m_ref.astype(np.float64) ** 2).mean()
                            / max(((m_x - m_ref) ** 2).mean(), 1e-30))
    # Log-spectral distance (dB RMS) — the perceptually-aligned metric.
    lsd = float(np.sqrt(((20 * np.log10(m_x + 1e-8) - 20 * np.log10(m_ref + 1e-8)) ** 2).mean()))
    print(f"audio: waveform_SNR={wav_snr:.1f} dB  magSTFT_SNR={mag_snr:.1f} dB  LSD={lsd:.2f} dB")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    for name, w in wavs.items():
        out = args.out_dir / f"ve_{name}.wav"
        sf.write(out, w, sample_rate)
        print(f"wrote {out} ({len(w) / sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
