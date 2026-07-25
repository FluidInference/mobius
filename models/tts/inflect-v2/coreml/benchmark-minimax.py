"""MiniMax-English benchmark for the Inflect v2 CoreML pipeline.

Mirrors FluidAudio's `tts-benchmark` metrics (one-shot backend): per-phrase
warm synth ms (TTFT == synth), p50/p95, aggregate RTFx, cold start, peak RSS.
Synthesis = espeak G2P + encoder (t512) + host expansion/noise + smallest
fitting synthesizer bucket. WAVs land as phrase_%03d.wav for WER scoring with
`fluidaudio tts-asr-verify --score-only`.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
HOP = 256
SAMPLE_RATE = 24000


def load_frontend(checkpoint_dir: Path):
    sys.path.insert(0, str(checkpoint_dir / "runtime"))
    sys.path.insert(0, str(checkpoint_dir))
    import commons
    from inflect_nano_v2_frontend import _configure_espeak, normalize_text
    from inflect_vits_frontend import _apply_phoneme_overrides
    from text import cleaned_text_to_sequence

    # The upstream frontend's phonemize() convenience constructs a fresh
    # espeak backend per call (~450 ms). A persistent backend matches what
    # an in-process port would do and produces identical phonemes.
    _configure_espeak()
    from phonemizer.backend import EspeakBackend

    backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)

    def tokenize(text: str) -> list[int]:
        phonemes = backend.phonemize([normalize_text(text)], strip=True, njobs=1)[0]
        phonemes = _apply_phoneme_overrides(phonemes)
        sequence = cleaned_text_to_sequence(phonemes)
        return commons.intersperse(sequence, 0)

    return tokenize


def edge_fade(waveform: np.ndarray, milliseconds: float = 5.0) -> np.ndarray:
    frames = min(round(SAMPLE_RATE * milliseconds / 1000.0), waveform.size // 2)
    if frames <= 0:
        return waveform
    ramp = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float32)
    waveform = waveform.copy()
    waveform[:frames] *= ramp
    waveform[-frames:] *= ramp[::-1]
    return waveform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["micro", "nano"], required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument(
        "--buckets", default="256,512,1024,2048",
        help="comma-separated synthesizer frame buckets (build dirs must exist)")
    parser.add_argument("--io-fp16", action="store_true", help="use -io16 fp16-I/O bundles")
    args = parser.parse_args()

    phrases = [
        l.strip()
        for l in args.corpus.read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    tokenize = load_frontend(ROOT / "checkpoints" / f"inflect-{args.variant}-v2")
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"build/inflect-{args.variant}-v2-fp16"
    suffix = "-io16" if args.io_fp16 else ""
    io_dtype = np.float16 if args.io_fp16 else np.float32
    bucket_list = sorted(int(b) for b in args.buckets.split(","))
    # f1024 predates the t512 conversions; the synthesizer is t_text-independent.
    t_for = lambda frames: 256 if (frames == 1024 and not args.io_fp16) else 512
    cold0 = time.perf_counter()
    encoder = ct.models.MLModel(
        f"{prefix}-t512-f256/encoder.mlpackage", compute_units=ct.ComputeUnit.ALL)
    synths = {
        frames: ct.models.MLModel(
            f"{prefix}-t{t_for(frames)}-f{frames}{suffix}/synthesizer.mlpackage",
            compute_units=ct.ComputeUnit.ALL)
        for frames in bucket_list
    }
    t_text = 512

    def synth(text: str, seed: int) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        tokens = tokenize(text)
        n_tok = len(tokens)
        assert n_tok <= t_text, f"{n_tok} tokens exceed t_text={t_text}: {text}"
        tokens_pad = np.zeros((1, t_text), dtype=np.int32)
        tokens_pad[0, :n_tok] = tokens
        x_mask = np.zeros((1, 1, t_text), dtype=np.float32)
        x_mask[0, 0, :n_tok] = 1.0
        enc = encoder.predict({"tokens": tokens_pad, "x_mask": x_mask})
        w_ceil = np.ceil(np.exp(enc["logw"][0, 0, :n_tok])).astype(np.int64)
        y_len = int(w_ceil.sum())
        buckets = sorted(synths)
        assert y_len <= buckets[-1], f"{y_len} frames exceed largest bucket: {text}"
        t_frames = next(b for b in buckets if y_len <= b)
        idx = np.repeat(np.arange(n_tok), w_ceil)
        m_exp = enc["m_p"][:, :, :n_tok][:, :, idx]
        logs_exp = enc["logs_p"][:, :, :n_tok][:, :, idx]
        noise = np.random.default_rng(seed).standard_normal(m_exp.shape, dtype=np.float32)
        z_p = np.zeros((1, m_exp.shape[1], t_frames), dtype=io_dtype)
        z_p[:, :, :y_len] = m_exp + noise * np.exp(logs_exp) * args.noise_scale
        y_mask = np.zeros((1, 1, t_frames), dtype=io_dtype)
        y_mask[0, 0, :y_len] = 1.0
        out = synths[t_frames].predict({"z_p": z_p, "y_mask": y_mask})
        audio = out["audio"][0, 0, : y_len * HOP].astype(np.float32)
        audio = np.clip(edge_fade(audio), -1.0, 1.0)
        return audio, (time.perf_counter() - t0) * 1000

    # Cold start = model loads; first synth = first end-to-end call (warm-up).
    cold_s = time.perf_counter() - cold0
    first0 = time.perf_counter()
    synth("Initialization warm-up.", seed=999)
    first_ms = (time.perf_counter() - first0) * 1000
    print(f"cold start {cold_s:.2f}s  first synth {first_ms:.0f} ms")

    per_phrase = []
    total_audio_s = 0.0
    total_synth_s = 0.0
    for i, text in enumerate(phrases):
        audio, ms = synth(text, seed=i)
        audio_s = len(audio) / SAMPLE_RATE
        sf.write(args.audio_dir / f"phrase_{i + 1:03d}.wav", audio, SAMPLE_RATE)
        per_phrase.append({"index": i + 1, "text": text, "synth_ms": ms, "audio_s": audio_s})
        total_audio_s += audio_s
        total_synth_s += ms / 1000
        print(f"[{i + 1:03d}/{len(phrases)}] {ms:6.1f} ms  {audio_s:5.2f}s  {text[:50]}")

    lat = sorted(p["synth_ms"] for p in per_phrase)
    p50 = lat[len(lat) // 2]
    p95 = lat[min(len(lat) - 1, round((len(lat) - 1) * 0.95))]
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    summary = {
        "backend": f"inflect-{args.variant}-v2-coreml",
        "corpus": args.corpus.name,
        "phrase_count": len(phrases),
        "cold_start_s": cold_s,
        "first_synth_ms": first_ms,
        "ttft_ms_p50": p50,
        "ttft_ms_p95": p95,
        "warm_synth_ms_p50": p50,
        "warm_synth_ms_p95": p95,
        "agg_rtfx": total_audio_s / total_synth_s,
        "peak_rss_mb": peak_rss_mb,
        "total_audio_s": total_audio_s,
    }
    args.output_json.write_text(json.dumps({"summary": summary, "phrases": per_phrase}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
