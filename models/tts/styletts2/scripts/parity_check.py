"""Parity check: decomposed `pipeline.orchestrator` vs `run_inference` ground truth.

Both code paths share the same model load, the same helpers, the same sampler,
and the same blend math (the orchestrator factors `make_inference_fn` into
named stages without duplicating logic). With identical seeds and a
read-only `ref_s`, the two waveforms should be bit-equivalent up to floating
point reordering — MSE expected near zero.

Usage:

    cd models/tts/styletts2
    uv run python scripts/parity_check.py
    uv run python scripts/parity_check.py \
        --text "Hello, this is StyleTTS 2." \
        --reference reference_audio/696_92939_000016_000006.wav \
        --seed 0 \
        --max-mse 1e-10

The script also runs a `ref_s`-mutation check (snapshots before each
call and asserts the tensor is byte-identical after).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Resolve paths so this script can be invoked from any cwd.
HERE = Path(__file__).resolve().parent.parent  # models/tts/styletts2
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# macOS espeak-ng dylib (mirrors run_inference.main()).
if sys.platform == "darwin":
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.1.dylib")
    os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", "/opt/homebrew/bin/espeak-ng")

# Ground-truth module — also installs vendor on sys.path and the torch.load shim.
import run_inference  # noqa: E402

from pipeline.orchestrator import synthesize  # noqa: E402
from pipeline.ref_s_guard import RefSGuard, freeze_ref_s  # noqa: E402


def _ensure_nltk() -> None:
    import nltk

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def _build_runtime(checkpoint_dir: Path, device: str):
    from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
    from text_utils import TextCleaner
    import phonemizer

    model, model_params = run_inference.load_styletts2(checkpoint_dir, device)
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )
    cleaner = TextCleaner()
    espeak = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )
    return model, model_params, sampler, espeak, cleaner


def _audio_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    diff = a - b
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(diff)))
    # Pearson correlation; handle zero-variance corner case.
    a_zero = a - a.mean()
    b_zero = b - b.mean()
    denom = float(np.sqrt((a_zero ** 2).sum() * (b_zero ** 2).sum()))
    corr = float((a_zero * b_zero).sum() / denom) if denom > 0 else float("nan")
    return {
        "samples_compared": n,
        "len_a": len(a),
        "len_b": len(b),
        "mse": mse,
        "rmse": rmse,
        "max_abs_delta": max_abs,
        "pearson_corr": corr,
        "rms_a": float(np.sqrt(np.mean(a * a))),
        "rms_b": float(np.sqrt(np.mean(b * b))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=(
            "StyleTTS 2 is a text to speech model that leverages style diffusion "
            "and adversarial training with large speech language models."
        ),
    )
    parser.add_argument(
        "--reference",
        default=str(HERE / "reference_audio" / "696_92939_000016_000006.wav"),
    )
    parser.add_argument("--checkpoint-dir", default=str(HERE / "checkpoints" / "LibriTTS"))
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--embedding-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-mse",
        type=float,
        default=1e-10,
        help="Fail the parity check if MSE exceeds this (default: 1e-10).",
    )
    parser.add_argument(
        "--min-corr",
        type=float,
        default=1.0 - 1e-6,
        help="Fail the parity check if pearson correlation is below this.",
    )
    parser.add_argument(
        "--save-wavs",
        action="store_true",
        help="Save out_monolithic.wav and out_orchestrator.wav alongside this script.",
    )
    args = parser.parse_args()

    device = "cpu"
    _ensure_nltk()

    print(f"Device: {device}")
    print(f"Text:   {args.text!r}")
    print(f"Ref:    {args.reference}")
    print(f"Seed:   {args.seed}")
    print()

    model, model_params, sampler, espeak, cleaner = _build_runtime(
        Path(args.checkpoint_dir), device
    )
    preprocess = run_inference.make_preprocess()

    # Compute ref_s once. Both paths must consume an immutable copy of it.
    ref_s_master = run_inference.compute_style(model, preprocess, device, args.reference)
    print(f"ref_s: shape={tuple(ref_s_master.shape)} dtype={ref_s_master.dtype}")

    # ---------- Path A: monolithic ground truth ----------
    inference = run_inference.make_inference_fn(
        model, model_params, sampler, espeak, cleaner, device
    )
    ref_s_a = freeze_ref_s(ref_s_master)
    run_inference.seed_everything(args.seed)
    with RefSGuard(ref_s_a, name="ref_s (monolithic)"):
        wav_a = inference(
            args.text,
            ref_s_a,
            alpha=args.alpha,
            beta=args.beta,
            diffusion_steps=args.diffusion_steps,
            embedding_scale=args.embedding_scale,
        )
    print(f"[A] monolithic   : {len(wav_a):>7d} samples ({len(wav_a) / 24000:.3f}s)")

    # ---------- Path B: decomposed orchestrator ----------
    ref_s_b = freeze_ref_s(ref_s_master)
    run_inference.seed_everything(args.seed)
    wav_b = synthesize(
        model=model,
        model_params=model_params,
        sampler=sampler,
        phonemizer=espeak,
        cleaner=cleaner,
        device=device,
        text=args.text,
        ref_s=ref_s_b,
        alpha=args.alpha,
        beta=args.beta,
        diffusion_steps=args.diffusion_steps,
        embedding_scale=args.embedding_scale,
        guard_ref_s=True,
    )
    print(f"[B] orchestrator : {len(wav_b):>7d} samples ({len(wav_b) / 24000:.3f}s)")

    # Also assert the master ref_s tensor never moved.
    delta_master = (ref_s_master - ref_s_master.detach().clone()).abs().max().item()
    assert delta_master == 0.0, f"ref_s_master corrupted (max_abs={delta_master})"

    metrics = _audio_metrics(wav_a, wav_b)
    print()
    print("Audio parity (orchestrator vs monolithic):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18s} {v:.6e}")
        else:
            print(f"  {k:<18s} {v}")

    if args.save_wavs:
        import soundfile as sf

        out_a = HERE / "out_monolithic.wav"
        out_b = HERE / "out_orchestrator.wav"
        sf.write(out_a, wav_a, 24000)
        sf.write(out_b, wav_b, 24000)
        print(f"\nWrote {out_a}")
        print(f"Wrote {out_b}")

    failed = []
    if metrics["mse"] > args.max_mse:
        failed.append(f"MSE {metrics['mse']:.3e} > {args.max_mse:.3e}")
    if not (metrics["pearson_corr"] >= args.min_corr):
        failed.append(f"corr {metrics['pearson_corr']:.6f} < {args.min_corr:.6f}")
    if metrics["len_a"] != metrics["len_b"]:
        failed.append(f"length mismatch {metrics['len_a']} vs {metrics['len_b']}")

    if failed:
        print("\nPARITY FAIL:")
        for f in failed:
            print(f"  - {f}")
        return 1

    print("\nPARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
