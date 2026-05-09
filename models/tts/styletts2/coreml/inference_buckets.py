"""Bucket-aware end-to-end CoreML inference.

Same pipeline as `coreml/inference.py` but loads the bucketed bert +
fused_diffusion_sampler packages built by `coreml/exporters/build_buckets.py`:

    bert_fp16_t{T}.mlpackage
    fused_diffusion_sampler_fp16_t{T}.mlpackage

The other six stages (text_encoder, ref_encoder, duration_predictor,
fused_f0n_har_source, decoder_pre, decoder_upsample) are identical to
the iteration_3 manifest and load from the standard package paths.

Usage
-----

    cd models/tts/styletts2
    uv run python coreml/inference_buckets.py --bucket 64  --text "..." \
        --output out_t64.wav
    uv run python coreml/inference_buckets.py --bucket 128 ...
    uv run python coreml/inference_buckets.py --bucket 256 ...

Or run all three with `--all`, which uses bucket-appropriate sample
prompts (short clause / sentence / paragraph) and writes
`out_t{64,128,256}.wav` next to `--output-dir`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import coremltools as ct  # noqa: E402

from coreml import inference as _inf  # noqa: E402  (reuse helpers)
from coreml._runtime import HERE, ensure_nltk  # noqa: E402

PACKAGES_DIR = HERE / "coreml" / "packages"

VALID_BUCKETS = (64, 128, 256)

# Sentences sized to phonemize within each bucket's token budget.
# Token counts measured empirically (espeak + TextCleaner gives ~1.5
# tokens / character for English LibriTTS-style prose).
DEFAULT_TEXTS: dict[int, str] = {
    64: "Hello there. How are you today?",
    128: "StyleTTS 2 is a text to speech model.",
    256: (
        "StyleTTS 2 is a text to speech model that produces clear, natural "
        "sounding speech in a variety of voices and speaking styles."
    ),
}


def _load_bucketed(stage: str, bucket_T: int, precision: str) -> ct.models.MLModel:
    """Load `<stage>_fp16_t{T}.mlpackage` with the manifest's compute placement."""
    if precision != "fp16":
        raise ValueError(f"bucketed packages are fp16 only, got {precision!r}")
    pkg = PACKAGES_DIR / f"{stage}_fp16_t{bucket_T}.mlpackage"
    if not pkg.exists():
        raise FileNotFoundError(
            f"missing {pkg} — run coreml/exporters/build_buckets.py --buckets {bucket_T} first"
        )
    cu = _inf._STAGE_COMPUTE[stage]
    return ct.models.MLModel(str(pkg), compute_units=cu)


def run_bucket(
    *,
    bucket_T: int,
    text: str,
    reference: str,
    output: str,
    alpha: float,
    beta: float,
    diffusion_steps: int,
    seed: int,
) -> dict:
    """Run the iteration_3 8-stage pipeline at the given bucket size.

    Returns a metadata dict (bucket_T, real_n, real_frames, output_path,
    duration_s, per-stage timings).
    """
    if bucket_T not in VALID_BUCKETS:
        raise ValueError(f"bucket must be one of {VALID_BUCKETS}, got {bucket_T}")

    import phonemizer  # noqa: E402

    import run_inference  # type: ignore  # noqa: E402
    from text_utils import TextCleaner  # type: ignore  # noqa: E402

    ensure_nltk()
    cleaner = TextCleaner()
    espeak = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )

    print(f"\n=== bucket T={bucket_T} ===")
    print(f"text: {text!r}")

    print("Loading eager StyleTTS2 (params lookup only)…")
    t0 = time.perf_counter()
    _, eager_params = run_inference.load_styletts2(
        Path(HERE / "checkpoints" / "LibriTTS"), "cpu"
    )
    print(f"  eager load: {time.perf_counter() - t0:.2f}s")

    print("Loading CoreML stages…")
    t0 = time.perf_counter()
    text_encoder = _inf._load_stage("text_encoder", precision="fp16")
    bert = _load_bucketed("bert", bucket_T, "fp16")
    ref_encoder = _inf._load_stage("ref_encoder", precision="fp16")
    fused_sampler = _load_bucketed("fused_diffusion_sampler", bucket_T, "fp16")
    duration_predictor = _inf._load_stage("duration_predictor", precision="fp16")
    fused_f0n_har = _inf._load_stage("fused_f0n_har_source", precision="fp32")
    decoder_pre = _inf._load_stage("decoder_pre", precision="fp16")
    decoder_upsample = _inf._load_stage("decoder_upsample", precision="fp16")
    print(f"  coreml load: {time.perf_counter() - t0:.2f}s")

    # ---- phonemize / tokenize ----
    from nltk.tokenize import word_tokenize

    HOP = 300

    ps = espeak.phonemize([text.strip()])
    ps = " ".join(word_tokenize(ps[0]))
    token_ids = cleaner(ps)
    token_ids.insert(0, 0)
    real_n = len(token_ids)
    if real_n > bucket_T:
        raise ValueError(
            f"text produced {real_n} tokens, exceeds bucket T={bucket_T}; "
            f"try a larger bucket or shorter text."
        )

    tokens_native = torch.LongTensor(token_ids).unsqueeze(0)
    input_lengths = torch.LongTensor([real_n])
    text_mask_native = run_inference.length_to_mask(input_lengths)

    pad_cols = bucket_T - real_n
    tokens_padded_ids = list(token_ids) + [0] * pad_cols
    tokens_padded = torch.LongTensor(tokens_padded_ids).unsqueeze(0)
    text_mask_padded = (
        torch.cat(
            [text_mask_native, torch.ones(1, pad_cols, dtype=text_mask_native.dtype)],
            dim=-1,
        )
        if pad_cols > 0
        else text_mask_native
    )
    print(f"tokens: real={real_n} padded={bucket_T}")

    timings: dict[str, float] = {}

    # ---- text_encoder ----
    t0 = time.perf_counter()
    feed = {
        "tokens": tokens_native.numpy().astype(np.int32),
        "input_lengths": input_lengths.numpy().astype(np.int32),
        "text_mask": text_mask_native.numpy().astype(np.float32),
    }
    (t_en_np,) = _inf._predict(text_encoder, feed)
    timings["text_encoder"] = time.perf_counter() - t0
    print(f"text_encoder:    {timings['text_encoder']:.3f}s  out={t_en_np.shape}")

    # ---- bert (bucketed) ----
    t0 = time.perf_counter()
    feed = {
        "tokens": tokens_padded.numpy().astype(np.int32),
        "attention_mask": (~text_mask_padded).int().numpy().astype(np.int32),
    }
    bert_dur_np, d_en_np_padded = _inf._predict(bert, feed)
    timings["bert"] = time.perf_counter() - t0
    print(
        f"bert:            {timings['bert']:.3f}s  "
        f"bert_dur={bert_dur_np.shape}  d_en={d_en_np_padded.shape}"
    )
    d_en_np = d_en_np_padded[:, :, :real_n].astype(np.float32)

    # ---- ref_encoder ----
    mel_4d = _inf._compute_mel_4d(reference)
    t0 = time.perf_counter()
    (ref_s_np,) = _inf._predict(ref_encoder, {"mel": mel_4d.numpy().astype(np.float32)})
    timings["ref_encoder"] = time.perf_counter() - t0
    print(f"ref_encoder:     {timings['ref_encoder']:.3f}s  ref_s={ref_s_np.shape}")
    ref_s = torch.from_numpy(ref_s_np).float()

    # ---- fused_diffusion_sampler (bucketed) ----
    run_inference.seed_everything(seed)
    t0 = time.perf_counter()
    noise_init = torch.randn(1, 256).unsqueeze(1).numpy().astype(np.float32)
    noises_aux = np.stack(
        [
            torch.randn(1, 1, 256).numpy().astype(np.float32)
            for _ in range(diffusion_steps - 1)
        ],
        axis=0,
    )
    feed = {
        "noise_init": noise_init,
        "noises_aux": noises_aux,
        "embedding": bert_dur_np.astype(np.float32),
        "features": ref_s_np.astype(np.float32),
    }
    (s_pred_np,) = _inf._predict(fused_sampler, feed)
    timings["fused_diffusion_sampler"] = time.perf_counter() - t0
    s_pred = torch.from_numpy(s_pred_np).squeeze(1)
    s_diff = s_pred[:, 128:]
    ref_diff = s_pred[:, :128]
    ref = alpha * ref_diff + (1.0 - alpha) * ref_s[:, :128]
    s = beta * s_diff + (1.0 - beta) * ref_s[:, 128:]
    print(
        f"sampler:         {timings['fused_diffusion_sampler']:.3f}s  "
        f"s_pred={tuple(s_pred.shape)}"
    )

    # ---- duration_predictor → alignment ----
    t0 = time.perf_counter()
    feed = {
        "d_en": d_en_np,
        "s": s.numpy().astype(np.float32),
        "text_mask": text_mask_native.float().numpy().astype(np.float32),
    }
    d_np, duration_logits_np = _inf._predict(duration_predictor, feed)
    duration = torch.sigmoid(torch.from_numpy(duration_logits_np)).sum(axis=-1)
    pred_dur = torch.round(duration.squeeze()).clamp(min=1)
    real_frames = int(pred_dur.sum().item())
    pred_aln_trg = _inf._build_pred_aln_trg(pred_dur, real_n)
    timings["duration_predictor"] = time.perf_counter() - t0
    print(
        f"duration_pred:   {timings['duration_predictor']:.3f}s  "
        f"frames={real_frames}"
    )

    # ---- en/asr build + fused_f0n_har_source ----
    d = torch.from_numpy(d_np).float()
    t_en = torch.from_numpy(t_en_np).float()
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
    asr = t_en @ pred_aln_trg.unsqueeze(0)
    if eager_params.decoder.type == "hifigan":
        en = _inf._hifigan_shift(en)
        asr = _inf._hifigan_shift(asr)
    t0 = time.perf_counter()
    feed = {
        "en": en.numpy().astype(np.float32),
        "s": s.numpy().astype(np.float32),
    }
    f0_pred_np, n_pred_np, har_np = _inf._predict(fused_f0n_har, feed)
    timings["fused_f0n_har_source"] = time.perf_counter() - t0
    print(
        f"f0n_har:         {timings['fused_f0n_har_source']:.3f}s  "
        f"f0={f0_pred_np.shape}  har={har_np.shape}"
    )

    # ---- decoder_pre ----
    ref_in = ref.squeeze().unsqueeze(0).numpy().astype(np.float32)
    t0 = time.perf_counter()
    feed = {
        "asr": asr.numpy().astype(np.float32),
        "f0_pred": f0_pred_np.astype(np.float32),
        "n_pred": n_pred_np.astype(np.float32),
        "ref": ref_in,
    }
    (x_pre_np,) = _inf._predict(decoder_pre, feed)
    timings["decoder_pre"] = time.perf_counter() - t0
    print(f"decoder_pre:     {timings['decoder_pre']:.3f}s  x_pre={x_pre_np.shape}")

    # ---- decoder_upsample ----
    t0 = time.perf_counter()
    feed = {
        "x_pre": x_pre_np.astype(np.float32),
        "ref": ref_in,
        "har_source": har_np.astype(np.float32),
    }
    (audio_np,) = _inf._predict(decoder_upsample, feed)
    timings["decoder_upsample"] = time.perf_counter() - t0
    print(f"decoder_upsample:{timings['decoder_upsample']:.3f}s  audio={audio_np.shape}")

    waveform = np.squeeze(audio_np)[..., :-50].astype(np.float32)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), waveform, 24000)
    duration_s = waveform.shape[-1] / 24000.0
    print(f"wrote {out_path} ({duration_s:.2f}s @ 24 kHz)")

    return {
        "bucket_T": bucket_T,
        "real_n": real_n,
        "real_frames": real_frames,
        "output_path": str(out_path),
        "duration_s": duration_s,
        "timings": timings,
        "total_s": sum(timings.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=int, choices=VALID_BUCKETS, default=None)
    parser.add_argument("--all", action="store_true", help="run all three buckets")
    parser.add_argument("--text", default=None, help="override default per-bucket text")
    parser.add_argument(
        "--reference",
        default=str(HERE / "reference_audio" / "696_92939_000016_000006.wav"),
    )
    parser.add_argument("--output", default=None, help="single-bucket output path")
    parser.add_argument(
        "--output-dir",
        default=str(HERE / "coreml"),
        help="directory for --all outputs (writes out_t{T}.wav)",
    )
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.all and args.bucket is not None:
        raise SystemExit("pass either --bucket or --all, not both")
    if not args.all and args.bucket is None:
        raise SystemExit("pass --bucket {64,128,256} or --all")

    targets = VALID_BUCKETS if args.all else (args.bucket,)
    out_dir = Path(args.output_dir)

    summaries = []
    for T in targets:
        text = args.text if args.text is not None else DEFAULT_TEXTS[T]
        if args.all:
            output = out_dir / f"out_t{T}.wav"
        else:
            output = (
                Path(args.output) if args.output is not None else out_dir / f"out_t{T}.wav"
            )
        meta = run_bucket(
            bucket_T=T,
            text=text,
            reference=args.reference,
            output=str(output),
            alpha=args.alpha,
            beta=args.beta,
            diffusion_steps=args.diffusion_steps,
            seed=args.seed,
        )
        summaries.append(meta)

    print("\n=== summary ===")
    for m in summaries:
        print(
            f"  T={m['bucket_T']:>3}  tokens={m['real_n']:>3}  frames={m['real_frames']:>4}  "
            f"audio={m['duration_s']:.2f}s  pipeline={m['total_s']*1000:.0f} ms  "
            f"-> {m['output_path']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
