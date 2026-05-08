"""Parity + ref_s-immutability tests.

These tests load the real model (~771 MB) and run two end-to-end
inferences, so they are slow (~30s on CPU). They are skipped automatically
if the checkpoint is not present.

Run:
    cd models/tts/styletts2
    uv run python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent.parent  # models/tts/styletts2
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CHECKPOINT = HERE / "checkpoints" / "LibriTTS" / "epochs_2nd_00020.pth"
REFERENCE = HERE / "reference_audio" / "696_92939_000016_000006.wav"

if sys.platform == "darwin":
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.1.dylib")
    os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", "/opt/homebrew/bin/espeak-ng")

requires_assets = pytest.mark.skipif(
    not CHECKPOINT.exists() or not REFERENCE.exists(),
    reason="StyleTTS2 LibriTTS checkpoint or reference audio not bootstrapped",
)


# ----- Fast unit tests (no model load) ---------------------------------------


def test_freeze_ref_s_returns_independent_copy() -> None:
    from pipeline.ref_s_guard import freeze_ref_s

    src = torch.randn(1, 256)
    frozen = freeze_ref_s(src)
    assert frozen.data_ptr() != src.data_ptr()
    assert torch.equal(frozen, src)
    frozen.add_(1.0)  # in-place mutate the copy
    assert not torch.equal(frozen, src), "src should not have been mutated"


def test_ref_s_guard_detects_mutation() -> None:
    from pipeline.ref_s_guard import RefSGuard

    t = torch.randn(1, 256)
    g = RefSGuard(t)
    t.add_(1e-7)  # tiny in-place mutation
    with pytest.raises(AssertionError, match="was mutated"):
        g.assert_unchanged()


def test_ref_s_guard_passes_when_unchanged() -> None:
    from pipeline.ref_s_guard import RefSGuard

    t = torch.randn(1, 256)
    with RefSGuard(t):
        # read-only ops should not trip the guard
        _ = t * 2.0
        _ = t[:, :128].mean()


# ----- Slow end-to-end parity -------------------------------------------------


@pytest.fixture(scope="module")
def runtime():
    """Build the StyleTTS2 model + sampler once for all parity tests."""
    import nltk

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    import run_inference  # noqa: WPS433
    import phonemizer  # noqa: WPS433
    from Modules.diffusion.sampler import (  # noqa: WPS433
        ADPM2Sampler,
        DiffusionSampler,
        KarrasSchedule,
    )
    from text_utils import TextCleaner  # noqa: WPS433

    device = "cpu"
    model, model_params = run_inference.load_styletts2(
        HERE / "checkpoints" / "LibriTTS", device
    )
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

    preprocess = run_inference.make_preprocess()
    ref_s = run_inference.compute_style(model, preprocess, device, str(REFERENCE))

    return {
        "device": device,
        "model": model,
        "model_params": model_params,
        "sampler": sampler,
        "cleaner": cleaner,
        "espeak": espeak,
        "ref_s": ref_s,
        "run_inference": run_inference,
    }


@requires_assets
def test_orchestrator_matches_monolithic(runtime) -> None:
    """Decomposed orchestrator must produce ~bit-equivalent audio to ground truth."""
    from pipeline.orchestrator import synthesize
    from pipeline.ref_s_guard import freeze_ref_s

    ri = runtime["run_inference"]
    text = "StyleTTS 2 is a text to speech model."

    # Path A — monolithic
    inference = ri.make_inference_fn(
        runtime["model"],
        runtime["model_params"],
        runtime["sampler"],
        runtime["espeak"],
        runtime["cleaner"],
        runtime["device"],
    )
    ref_a = freeze_ref_s(runtime["ref_s"])
    ri.seed_everything(0)
    wav_a = inference(text, ref_a, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1.0)

    # Path B — decomposed
    ref_b = freeze_ref_s(runtime["ref_s"])
    ri.seed_everything(0)
    wav_b = synthesize(
        model=runtime["model"],
        model_params=runtime["model_params"],
        sampler=runtime["sampler"],
        phonemizer=runtime["espeak"],
        cleaner=runtime["cleaner"],
        device=runtime["device"],
        text=text,
        ref_s=ref_b,
        alpha=0.3,
        beta=0.7,
        diffusion_steps=5,
        embedding_scale=1.0,
        guard_ref_s=True,
    )

    assert len(wav_a) == len(wav_b), f"length mismatch: {len(wav_a)} vs {len(wav_b)}"
    diff = wav_a.astype(np.float64) - wav_b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    max_abs = float(np.max(np.abs(diff)))
    assert mse < 1e-10, f"MSE {mse:.3e} above tolerance"
    assert max_abs < 1e-5, f"max abs delta {max_abs:.3e} above tolerance"


@requires_assets
def test_ref_s_not_mutated_by_orchestrator(runtime) -> None:
    """Running synthesize must leave the caller's ref_s tensor byte-identical."""
    from pipeline.orchestrator import synthesize

    ri = runtime["run_inference"]
    ref_s = runtime["ref_s"].clone()
    snapshot = ref_s.clone()

    ri.seed_everything(0)
    _ = synthesize(
        model=runtime["model"],
        model_params=runtime["model_params"],
        sampler=runtime["sampler"],
        phonemizer=runtime["espeak"],
        cleaner=runtime["cleaner"],
        device=runtime["device"],
        text="hello",
        ref_s=ref_s,
        diffusion_steps=5,
        guard_ref_s=True,
    )

    assert torch.equal(ref_s, snapshot), "synthesize() mutated caller's ref_s"
