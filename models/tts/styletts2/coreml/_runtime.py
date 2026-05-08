"""Shared runtime bring-up + representative-input capture for CoreML conversion.

Builds the StyleTTS2 model exactly as `run_inference.py` does (single
source of truth for the model graph) and captures every intermediate
tensor a CoreML stage might need by running `pipeline.orchestrator`
once. The captured tensors become both:

  * the *trace inputs* for each stage's `torch.jit.trace`, and
  * the *reference outputs* for per-stage parity checks.

There is no separate model-build path for CoreML conversion — the
weights, sampler, and helpers are exactly those used by ground truth.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent.parent  # models/tts/styletts2
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

if sys.platform == "darwin":
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.1.dylib"
    )
    os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", "/opt/homebrew/bin/espeak-ng")


def ensure_nltk() -> None:
    import nltk

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


@dataclass
class Runtime:
    device: str
    model: Any
    model_params: Any
    sampler: Any
    cleaner: Any
    espeak: Any
    ref_s: torch.Tensor      # [1, 256], frozen
    mel_4d: torch.Tensor     # [1, 1, 80, T_mel] — input to ref_encoder stage
    captures: Any            # pipeline.stages.StageOutputs from one full run


def _compute_ref_mel(reference_path: str) -> torch.Tensor:
    """Re-run the librosa load + trim that compute_style does, returning the
    4D mel tensor that the ref_encoder stage consumes."""
    import librosa

    import run_inference  # type: ignore

    wave, sr = librosa.load(reference_path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    preprocess = run_inference.make_preprocess()
    mel = preprocess(audio)  # [1, 80, T_mel]
    return mel.unsqueeze(1)   # [1, 1, 80, T_mel]


def build_runtime(
    *,
    text: str = "StyleTTS 2 is a text to speech model.",
    reference: str | None = None,
    checkpoint_dir: str | None = None,
    seed: int = 0,
) -> Runtime:
    """Load the model, sampler, helpers; capture all intermediates from one run."""
    import phonemizer

    import run_inference  # type: ignore
    from Modules.diffusion.sampler import (  # type: ignore
        ADPM2Sampler,
        DiffusionSampler,
        KarrasSchedule,
    )
    from text_utils import TextCleaner  # type: ignore

    from pipeline.orchestrator import run_pipeline
    from pipeline.stages import StageInputs

    if reference is None:
        reference = str(HERE / "reference_audio" / "696_92939_000016_000006.wav")
    if checkpoint_dir is None:
        checkpoint_dir = str(HERE / "checkpoints" / "LibriTTS")

    ensure_nltk()
    device = "cpu"

    model, model_params = run_inference.load_styletts2(Path(checkpoint_dir), device)
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
    ref_s = run_inference.compute_style(model, preprocess, device, reference)
    mel_4d = _compute_ref_mel(reference)

    # Capture every intermediate tensor by running the orchestrator once.
    run_inference.seed_everything(seed)
    captures = run_pipeline(
        model=model,
        model_params=model_params,
        sampler=sampler,
        phonemizer=espeak,
        cleaner=cleaner,
        device=device,
        inputs=StageInputs(text=text, ref_s=ref_s),
        guard_ref_s=False,
    )

    return Runtime(
        device=device,
        model=model,
        model_params=model_params,
        sampler=sampler,
        cleaner=cleaner,
        espeak=espeak,
        ref_s=ref_s.detach().clone().contiguous(),
        mel_4d=mel_4d,
        captures=captures,
    )


def stage_example_inputs(stage: str, rt: Runtime) -> tuple:
    """Return the example tensor tuple to feed the wrapper for tracing.

    Tensors are returned exactly as they appeared during the captured
    `run_pipeline` call (single source of truth).
    """
    c = rt.captures
    if stage == "text_encoder":
        return (c.tokens, c.input_lengths, c.text_mask)
    if stage == "bert":
        attn = (~c.text_mask).int()
        return (c.tokens, attn)
    if stage == "ref_encoder":
        return (rt.mel_4d,)
    if stage == "diffusion_unet":
        # Single representative denoise step. The 5-step ADPM2 schedule
        # lives in Python; CoreML only sees one step at a time.
        # Use a deterministic generator so convert.py and parity.py see
        # bit-identical x_noisy values.
        g = torch.Generator()
        g.manual_seed(42)
        x_noisy = torch.randn(1, 1, 256, generator=g)
        sigma = torch.tensor([1.0], dtype=torch.float32)
        embedding = c.bert_dur
        features = rt.ref_s
        return (x_noisy, sigma, embedding, features)
    if stage == "duration_predictor":
        # text_mask: bool -> fp32 so CoreML I/O stays fp32-only (matches
        # text_encoder convention; mask is consumed multiplicatively in
        # the wrapper).
        mask_f = c.text_mask.to(torch.float32)
        return (c.d_en, c.s, mask_f)
    if stage == "f0n_predictor":
        return (c.en, c.s)
    if stage == "har_source":
        return (c.f0_pred,)
    if stage == "decoder":
        from coreml.wrappers import precompute_har_source
        har = precompute_har_source(rt.model.decoder, c.f0_pred)
        return (c.asr, c.f0_pred, c.n_pred, c.ref, har)
    if stage == "decoder_pre":
        return (c.asr, c.f0_pred, c.n_pred, c.ref)
    if stage == "decoder_upsample":
        from coreml.wrappers import DecoderPreWrapper, precompute_har_source
        pre = DecoderPreWrapper(rt.model.decoder)
        with torch.no_grad():
            x_pre = pre(c.asr, c.f0_pred, c.n_pred, c.ref)
        har = precompute_har_source(rt.model.decoder, c.f0_pred)
        return (x_pre, c.ref, har)
    raise ValueError(f"unknown stage: {stage!r}")


def stage_reference_outputs(stage: str, rt: Runtime) -> tuple:
    """Return the *expected* output tensor(s) for parity checks."""
    c = rt.captures
    if stage == "text_encoder":
        return (c.t_en,)
    if stage == "bert":
        return (c.bert_dur, c.d_en)
    if stage == "ref_encoder":
        return (rt.ref_s,)
    if stage == "duration_predictor":
        # `d` is captured directly. Re-derive the duration_proj logits
        # from `d` here so per-stage parity covers the full wrapper
        # output. (StageOutputs only stores the post-sigmoid `pred_dur`
        # — the raw logits would round-trip cleanly anyway.)
        with torch.no_grad():
            try:
                rt.model.predictor.lstm.flatten_parameters()
            except Exception:  # noqa: BLE001
                pass
            x, _ = rt.model.predictor.lstm(c.d)
            duration = rt.model.predictor.duration_proj(x)
        return (c.d, duration)
    if stage == "f0n_predictor":
        return (c.f0_pred, c.n_pred)
    if stage == "har_source":
        from coreml.wrappers import precompute_har_source
        # Reference: the eager CPU-side precompute_har_source path.
        # HarSourceWrapper math is bit-equivalent (see docstring).
        har = precompute_har_source(rt.model.decoder, c.f0_pred)
        return (har,)
    if stage == "decoder":
        # The captured `c.waveform` was post-`squeeze() + [..., :-50]`
        # trimmed (see pipeline/stages.decode_audio). The CoreML wrapper
        # emits the raw `(1, 1, T_audio)` decoder output. To compare on a
        # consistent shape we re-run the wrapper eagerly. The wrapper
        # patches SineGen/SourceModuleHnNSF to zero out their three
        # `torch.rand`/`randn_like` calls, so eager and CoreML are
        # bit-identical by construction (no seeding required).
        from coreml.wrappers import build_wrapper as _bw  # avoid cyclic at module import
        wrapper = _bw("decoder", rt.model)
        inputs = stage_example_inputs("decoder", rt)
        with torch.no_grad():
            out = wrapper(*inputs)
        return (out,)
    if stage == "diffusion_unet":
        # Re-run the wrapper eagerly. The denoise step itself is
        # deterministic (no RNG inside KDiffusion.denoise_fn); only the
        # surrounding ADPM2 sampler draws randn for the stochastic step.
        from coreml.wrappers import build_wrapper as _bw
        wrapper = _bw("diffusion_unet", rt.model)
        inputs = stage_example_inputs("diffusion_unet", rt)
        with torch.no_grad():
            out = wrapper(*inputs)
        return (out,)
    if stage == "decoder_pre":
        from coreml.wrappers import build_wrapper as _bw
        wrapper = _bw("decoder_pre", rt.model)
        inputs = stage_example_inputs("decoder_pre", rt)
        with torch.no_grad():
            out = wrapper(*inputs)
        return (out,)
    if stage == "decoder_upsample":
        from coreml.wrappers import build_wrapper as _bw
        wrapper = _bw("decoder_upsample", rt.model)
        inputs = stage_example_inputs("decoder_upsample", rt)
        with torch.no_grad():
            out = wrapper(*inputs)
        return (out,)
    raise NotImplementedError(f"reference outputs for {stage!r} not yet defined")
