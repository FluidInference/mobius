"""StyleTTS2 LibriTTS PyTorch ground-truth inference.

Drives the upstream `yl4579/StyleTTS2` LibriTTS model end-to-end on CPU and
writes a 24 kHz WAV. The output of this script is the parity reference for
all CoreML conversion work in this directory.

Layout this script expects (everything except the script itself is gitignored):

    models/tts/styletts2/
    ├── run_inference.py            ← this file
    ├── vendor/StyleTTS2/           ← `git clone https://github.com/yl4579/StyleTTS2`
    ├── checkpoints/LibriTTS/
    │   ├── config.yml              ← from yl4579/StyleTTS2-LibriTTS
    │   └── epochs_2nd_00020.pth    ← from yl4579/StyleTTS2-LibriTTS (~771 MB)
    └── reference_audio/            ← from reference_audio.zip (yl4579/StyleTTS2-LibriTTS)

See README.md for the bootstrap commands.

Usage:
    uv run python run_inference.py \
        --text "Hello, this is StyleTTS 2." \
        --reference reference_audio/696_92939_000016_000006.wav \
        --output out.wav \
        --seed 0
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

# StyleTTS2 ships .pth/.t7 checkpoints saved with the older torch convention.
# PyTorch 2.6+ defaults `weights_only=True`, which breaks loading. We trust
# these official author-released checkpoints, so force weights_only=False.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

import yaml  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

# Paths
HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "StyleTTS2"
DEFAULT_CHECKPOINT_DIR = HERE / "checkpoints" / "LibriTTS"
DEFAULT_REFERENCE = HERE / "reference_audio" / "696_92939_000016_000006.wav"

if not VENDOR.exists():
    sys.exit(
        f"vendor/StyleTTS2 missing at {VENDOR}.\n"
        "Bootstrap with:\n"
        f"    git clone https://github.com/yl4579/StyleTTS2.git {VENDOR}"
    )

# Vendor must be importable for `models`, `utils`, `text_utils`, `Modules`, `Utils` modules.
sys.path.insert(0, str(VENDOR))

from models import build_model, load_ASR_models, load_F0_models  # noqa: E402
from utils import recursive_munch  # noqa: E402
from text_utils import TextCleaner  # noqa: E402
from Modules.diffusion.sampler import (  # noqa: E402
    ADPM2Sampler,
    DiffusionSampler,
    KarrasSchedule,
)
from Utils.PLBERT.util import load_plbert  # noqa: E402


def seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def make_preprocess():
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300
    )
    mean, std = -4.0, 4.0

    def preprocess(wave: np.ndarray) -> torch.Tensor:
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        return (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std

    return preprocess


def compute_style(model, preprocess, device: str, path: str) -> torch.Tensor:
    wave, sr = librosa.load(path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    mel_tensor = preprocess(audio).to(device)
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)


def _resolve_config_paths(config: dict) -> dict:
    """Rewrite Utils/* paths in upstream config to point at vendor/StyleTTS2/Utils/*."""
    for key in ("ASR_path", "ASR_config", "F0_path", "PLBERT_dir"):
        v = config.get(key)
        if isinstance(v, str) and v.startswith(("Utils/", "./Utils/")):
            config[key] = str(VENDOR / v.lstrip("./"))
    return config


def load_styletts2(checkpoint_dir: Path, device: str):
    config_path = checkpoint_dir / "config.yml"
    ckpt_path = checkpoint_dir / "epochs_2nd_00020.pth"
    config = yaml.safe_load(open(config_path))
    config = _resolve_config_paths(config)

    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    for k in model:
        model[k].eval()
        model[k].to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    params_whole = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = params_whole["net"]
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
                print(f"  {key} loaded")
            except RuntimeError:
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k[7:]] = v  # strip 'module.'
                model[key].load_state_dict(new_state_dict, strict=False)
                print(f"  {key} loaded (stripped module.)")
    for k in model:
        model[k].eval()

    return model, model_params


def make_inference_fn(model, model_params, sampler, phonemizer, cleaner, device):
    def inference(
        text: str,
        ref_s: torch.Tensor,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0,
    ) -> np.ndarray:
        text = text.strip()
        ps = phonemizer.phonemize([text])
        ps = " ".join(word_tokenize(ps[0]))
        tokens = cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(device),
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

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().item()))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].item())] = 1
                c_frame += int(pred_dur[i].item())

            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            f0_pred, n_pred = model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))

        # Repo notes: trim weird tail pulse.
        return out.squeeze().cpu().numpy()[..., :-50]

    return inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=(
            "StyleTTS 2 is a text to speech model that leverages style diffusion "
            "and adversarial training with large speech language models."
        ),
    )
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--output", default="out.wav")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--embedding-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)

    # macOS: the diffusion sampler has historically had issues on MPS.
    device = "cpu"
    print(f"Device: {device}")

    # phonemizer needs espeak-ng; on macOS the dylib isn't auto-discovered.
    if sys.platform == "darwin":
        os.environ.setdefault(
            "PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.1.dylib"
        )
        os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", "/opt/homebrew/bin/espeak-ng")

    # nltk punkt for word_tokenize
    import nltk  # noqa: E402

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            print(f"Downloading nltk: {pkg}")
            nltk.download(pkg, quiet=True)

    import phonemizer  # noqa: E402

    global_phonemizer = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )

    model, model_params = load_styletts2(Path(args.checkpoint_dir), device)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    cleaner = TextCleaner()
    inference = make_inference_fn(
        model, model_params, sampler, global_phonemizer, cleaner, device
    )

    preprocess = make_preprocess()
    print(f"Computing style from: {args.reference}")
    ref_s = compute_style(model, preprocess, device, args.reference)

    print("Synthesizing...")
    t0 = time.time()
    wav = inference(
        args.text,
        ref_s,
        alpha=args.alpha,
        beta=args.beta,
        diffusion_steps=args.diffusion_steps,
        embedding_scale=args.embedding_scale,
    )
    elapsed = time.time() - t0
    duration = len(wav) / 24000.0
    print(f"Generated {duration:.2f}s of audio in {elapsed:.2f}s (RTF={elapsed/duration:.3f})")

    sf.write(args.output, wav, 24000)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
