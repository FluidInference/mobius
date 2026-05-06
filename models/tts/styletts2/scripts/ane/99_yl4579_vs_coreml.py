"""Head-to-head: yl4579/StyleTTS2 reference PyTorch inference vs our 7-graph CoreML.

Loads the upstream model (`vendor/StyleTTS2/models.py::build_model`), runs the
notebook's `inference()` function verbatim, then runs our `synth_coreml_pipeline`
with the *same* tokens + ref_s + sampler seed, and compares the outputs.

Both paths share:
  - The stripped checkpoint (`checkpoints/styletts2_libritts_inference.pt`)
  - The same espeak-ng phonemizer + word_tokenize + TextCleaner
  - The same `compute_style(...)` recipe (mel(80,2048,1200,300), trim top_db=30)
  - The same noise seed (`torch.manual_seed(0); torch.randn((1,1,256))`)
  - The same alpha/beta (0.3, 0.7) and diffusion_steps (5)

Reports:
  - Per-stage shape sanity
  - Audio: RMS dB, peak dB, length samples
  - log-mel cosine similarity (the validator's existing metric)
  - Optional ASR readback (off by default; pass --asr to enable)

Usage:
    cd mobius/models/tts/styletts2
    uv run python scripts/ane/99_yl4579_vs_coreml.py \\
        --phrase "Hello world. This is a test of the StyleTTS 2 system." \\
        --ref vendor/StyleTTS2/Demo/reference_audio/1221-135767-0014.wav \\
        --coreml-dir coreml/build/ane \\
        --out-dir /tmp/yl4579-vs-coreml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Auto-detect espeak-ng on macOS (Homebrew). Phonemizer can't find it itself.
if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
    for _cand in (
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
    ):
        if Path(_cand).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _cand
            break

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent  # models/tts/styletts2
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT / "scripts"))
# Vendor path is set up by `_styletts2_lib`, but we import the upstream
# `models` and `utils` here directly so push it once more to be safe.
VENDOR = ROOT / "vendor" / "StyleTTS2"
sys.path.insert(0, str(VENDOR))

from _styletts2_lib import DEFAULT_CHECKPOINT  # noqa: E402

# Reuse helpers from the existing 7-graph validator.
from importlib import import_module

_validator = import_module("99_e2e_validate")
log_mel = _validator.log_mel
cosine_2d = _validator.cosine_2d
db = _validator.db
peak_db = _validator.peak_db
synth_coreml_pipeline = _validator.synth_coreml_pipeline
_write_wav = _validator._write_wav


# ---------------------------------------------------------------------------
# Upstream model loading and inference (notebook recipe verbatim)
# ---------------------------------------------------------------------------


def load_upstream_model(checkpoint: Path):
    """Build upstream `model` dict via `models.build_model` and load weights
    from our stripped checkpoint. The non-inference modules (text_aligner,
    pitch_extractor, plbert) are still required by `build_model`; they're
    loaded from the vendored `Utils/` assets.
    """
    import yaml
    from munch import Munch

    cfg_path = VENDOR / "Models" / "LibriTTS" / "config.yml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # --- Support models (these only appear in the upstream graph; not used
    # at inference time but build_model needs them).
    cwd = Path.cwd()
    import os

    os.chdir(VENDOR)  # JDC/PLBERT/ASR config paths are relative to vendor root
    # PyTorch 2.6 flipped torch.load's `weights_only` default to True. The
    # upstream JDC/ASR loaders (`models.py:_load_model`) don't pass the
    # kwarg and the .pth files contain non-tensor objects (config dicts),
    # so the safe loader rejects them. Wrap torch.load for the duration of
    # this build only — these are local files we just downloaded from HF.
    _orig_torch_load = torch.load

    def _torch_load_unsafe(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_torch_load(*a, **kw)

    torch.load = _torch_load_unsafe  # type: ignore[assignment]
    try:
        from Utils.PLBERT.util import load_plbert  # type: ignore
        # load_ASR_models / load_F0_models live in models.py; recursive_munch
        # lives in utils.py. utils.py imports `monotonic_align` (training-only,
        # Cython) at the top, so we re-implement recursive_munch here to skip
        # that dep entirely.
        from models import build_model, load_ASR_models, load_F0_models  # type: ignore
        from munch import Munch

        def recursive_munch(d):
            if isinstance(d, dict):
                return Munch((k, recursive_munch(v)) for k, v in d.items())
            if isinstance(d, list):
                return [recursive_munch(v) for v in d]
            return d

        text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
        pitch_extractor = load_F0_models(config["F0_path"])
        plbert = load_plbert(config["PLBERT_dir"])
        model_params = recursive_munch(config["model_params"])
        model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    finally:
        os.chdir(cwd)
        torch.load = _orig_torch_load  # type: ignore[assignment]

    # All modules to eval mode and CPU.
    for k in model:
        model[k].eval()
        model[k].to("cpu")

    # Load stripped weights.
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Stripped checkpoint {checkpoint} not found. "
            f"Run scripts/00_fetch_weights.py first."
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for name, sd in state.items():
        if name not in model:
            print(f"[upstream]   ! '{name}' in checkpoint but not in upstream model")
            continue
        missing, unexpected = model[name].load_state_dict(sd, strict=False)
        if missing:
            print(f"[upstream] {name}: {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"[upstream] {name}: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    return model, model_params


def make_sampler(model):
    """Build the upstream DiffusionSampler exactly as in the notebook."""
    from Modules.diffusion.sampler import (  # type: ignore
        ADPM2Sampler,
        DiffusionSampler,
        KarrasSchedule,
    )

    return DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def upstream_compute_style(model, path: Path) -> torch.Tensor:
    """yl4579 `compute_style` verbatim."""
    import librosa
    import torchaudio

    wave, sr = librosa.load(str(path), sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)

    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300
    )
    mean, std = -4.0, 4.0
    wave_tensor = torch.from_numpy(audio).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)


def upstream_phonemize_and_tokenize(text: str):
    """Notebook recipe: espeak-ng → word_tokenize → TextCleaner → BOS-prepend."""
    import phonemizer

    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
    except (ImportError, LookupError):
        try:
            import nltk

            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    from nltk.tokenize import word_tokenize

    from text_utils import TextCleaner  # type: ignore

    backend = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )
    cleaner = TextCleaner()
    ps = backend.phonemize([text.strip()])
    ps = word_tokenize(ps[0])
    ps = " ".join(ps)
    tokens = cleaner(ps)
    tokens.insert(0, 0)  # BOS
    return tokens, ps


def upstream_inference(
    model,
    model_params,
    sampler,
    text: str,
    ref_s: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
    seed: int = 0,
):
    """yl4579 notebook `inference()` verbatim."""
    tokens, ps = upstream_phonemize_and_tokenize(text)
    tok = torch.LongTensor(tokens).unsqueeze(0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tok.shape[-1]])
        text_mask = length_to_mask(input_lengths)

        t_en = model.text_encoder(tok, input_lengths, text_mask)
        bert_dur = model.bert(tok, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    audio = out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at end (notebook comment)
    return audio.astype(np.float32), {
        "tokens": np.array(tokens, dtype=np.int64),
        "ps": ps,
        "s_pred": s_pred.cpu().numpy(),
        "F0_pred": F0_pred.squeeze(0).cpu().numpy(),
        "N_pred": N_pred.squeeze(0).cpu().numpy(),
        "pred_dur": pred_dur.cpu().numpy(),
        "ref_s_mixed_acoustic": ref.squeeze().cpu().numpy(),
        "ref_s_mixed_pros": s.squeeze().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Wrapper: run our 7-graph CoreML with the same tokens + ref_s
# ---------------------------------------------------------------------------


def coreml_inference(
    coreml_dir: Path,
    tokens: np.ndarray,
    ref_s: np.ndarray,
    *,
    ref_s_pred_acoustic: np.ndarray,
    ref_s_pred_pros: np.ndarray,
    pred_dur: np.ndarray,
    bert_dur: np.ndarray,
    compute_units: str = "cpu_and_gpu",
):
    """Drive `synth_coreml_pipeline` with overrides matching upstream's
    diffusion + duration outputs. This isolates parity to the *graph*
    computations (PostBert/Alignment/Prosody/Vocoder) rather than letting
    sampler RNG drift dominate the comparison.
    """
    from _styletts2_ane_lib import load_modules_for_ane

    modules, cfg = load_modules_for_ane(checkpoint=DEFAULT_CHECKPOINT)
    # Concatenate the upstream-mixed halves into the [acou|pros] layout the
    # CoreML pipeline expects for the post-diffusion ref_s.
    style_pred = np.concatenate([ref_s_pred_acoustic, ref_s_pred_pros], axis=-1).reshape(-1)
    style_raw = ref_s.reshape(-1).astype(np.float32)
    kwargs = dict(
        ref_s=style_raw,
        tokens=tokens.astype(np.int64),
        coreml_dir=coreml_dir,
        ref_s_pred=style_pred.astype(np.float32),
        pred_dur_override=pred_dur.astype(np.float32),
        use_mlpackage=True,
        compute_units=compute_units,
    )
    if bert_dur is not None:
        kwargs["bert_dur_override"] = bert_dur.astype(np.float32)
    audio, F0, N = synth_coreml_pipeline(modules, cfg, **kwargs)
    return audio.astype(np.float32), F0.astype(np.float32), N.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phrase", type=str, default="Hello world. This is a test of the StyleTTS 2 system.")
    p.add_argument(
        "--ref",
        type=Path,
        default=ROOT / "vendor/StyleTTS2/Demo/reference_audio/1221-135767-0014.wav",
    )
    p.add_argument("--coreml-dir", type=Path, default=ROOT / "coreml" / "build" / "ane")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/yl4579-vs-coreml"))
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--diffusion-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--compute-units", default="cpu_and_gpu",
        choices=["cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"],
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.ref.exists():
        raise FileNotFoundError(f"Reference audio missing: {args.ref}")

    print(f"[head-to-head] phrase: {args.phrase!r}")
    print(f"[head-to-head] ref:    {args.ref}")
    print(f"[head-to-head] coreml: {args.coreml_dir}")

    # ---- Upstream PyTorch
    print("[head-to-head] loading upstream model …", flush=True)
    t0 = time.time()
    model, model_params = load_upstream_model(DEFAULT_CHECKPOINT)
    sampler = make_sampler(model)
    print(f"[head-to-head] upstream loaded in {time.time() - t0:.2f}s", flush=True)

    print("[head-to-head] computing ref_s from reference WAV …", flush=True)
    ref_s = upstream_compute_style(model, args.ref)
    print(f"[head-to-head]   ref_s shape: {tuple(ref_s.shape)}, "
          f"|ref_s|: {ref_s.norm().item():.3f}")

    print("[head-to-head] running upstream inference() …", flush=True)
    t0 = time.time()
    pt_audio, meta = upstream_inference(
        model,
        model_params,
        sampler,
        args.phrase,
        ref_s,
        alpha=args.alpha,
        beta=args.beta,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
    )
    print(f"[head-to-head] upstream inference: {time.time() - t0:.2f}s, "
          f"audio_len={pt_audio.shape[-1]} ({pt_audio.shape[-1] / 24000:.2f}s)")

    pt_path = args.out_dir / "upstream_pt.wav"
    _write_wav(pt_path, pt_audio, sr=24000)
    print(f"[head-to-head] wrote {pt_path}")

    # ---- CoreML 7-graph (using upstream-derived tokens + ref_s_pred + pred_dur)
    print("[head-to-head] running 7-graph CoreML pipeline …", flush=True)
    t0 = time.time()
    cm_audio, cm_F0, cm_N = coreml_inference(
        args.coreml_dir,
        meta["tokens"],
        ref_s.squeeze(0).cpu().numpy(),
        ref_s_pred_acoustic=meta["ref_s_mixed_acoustic"],
        ref_s_pred_pros=meta["ref_s_mixed_pros"],
        pred_dur=meta["pred_dur"],
        bert_dur=None,  # let CoreML's PLBert produce bert_dur (also worth checking)
        compute_units=args.compute_units,
    )
    print(f"[head-to-head] coreml inference: {time.time() - t0:.2f}s, "
          f"audio_len={cm_audio.shape[-1]} ({cm_audio.shape[-1] / 24000:.2f}s)")

    cm_path = args.out_dir / "coreml.wav"
    _write_wav(cm_path, cm_audio, sr=24000)
    print(f"[head-to-head] wrote {cm_path}")

    # ---- Compare
    print()
    print("=" * 70)
    print("HEAD-TO-HEAD: yl4579 upstream PyTorch vs our 7-graph CoreML")
    print("=" * 70)
    n = min(pt_audio.shape[-1], cm_audio.shape[-1])
    pt_n = pt_audio[..., :n]
    cm_n = cm_audio[..., :n]
    print(f"audio length:    pt={pt_audio.shape[-1]:>6}  cm={cm_audio.shape[-1]:>6}  common={n}")
    print(f"RMS dB:          pt={db(pt_n):+7.2f}  cm={db(cm_n):+7.2f}  Δ={db(cm_n) - db(pt_n):+6.2f}")
    print(f"peak dB:         pt={peak_db(pt_n):+7.2f}  cm={peak_db(cm_n):+7.2f}  Δ={peak_db(cm_n) - peak_db(pt_n):+6.2f}")

    pt_mel = log_mel(pt_n, sr=24000)
    cm_mel = log_mel(cm_n, sr=24000)
    cos = cosine_2d(pt_mel, cm_mel)
    print(f"log-mel cos:     {cos:.4f}   (≥ 0.99 = ship; ≥ 0.97 = useful; < 0.95 = drift)")

    pt_f0 = meta["F0_pred"][..., :cm_F0.shape[-1] if cm_F0.ndim > 0 else None]
    print(f"F0 cos (PT vs CM): "
          f"{cosine_2d(meta['F0_pred'][None, :], cm_F0[None, :]):.4f}")
    print(f"N  cos (PT vs CM): "
          f"{cosine_2d(meta['N_pred'][None, :], cm_N[None, :]):.4f}")

    print()
    print(f"WAVs written to {args.out_dir}/")
    print(f"  upstream:  {pt_path}")
    print(f"  coreml:    {cm_path}")


if __name__ == "__main__":
    main()
