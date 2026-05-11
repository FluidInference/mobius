#!/usr/bin/env python3
"""Export a voice as a v2 KV-cache safetensors file using pocket-tts >= 2.0.0.

Why this exists
---------------
The legacy ``export_voice_coreml.py`` runs a CoreML ``mimi_encoder.mlmodelc``
that was traced against a pocket-tts version older than the Apr 27 cond_step /
flowlm_step / flow_decoder mlpackages now deployed under ``build/<lang>/``.
That mismatch produces a ``*_audio_prompt.bin`` whose conditioning latent space
disagrees with the new cond_step weights; feeding it through the v1 prefill
path causes the flow LM to emit EOS within a handful of steps (silent or
garbled output — FluidAudio issue #592).

This script side-steps the broken CoreML encoder by running the *exact* PyTorch
pipeline that produced the deployed v2 caches: pocket-tts 2.0.0's
``TTSModel.get_state_for_audio_prompt`` (encode mimi -> project -> bos prefix
-> cond_step prefill -> capture KV state). The output is a v2 KV-cache
safetensors with shape ``[2, 1, prompt_len, 16, 64]`` per layer plus
``[prompt_len]`` offsets, drop-in compatible with
``_locate_voice_v2`` in ``generate_coreml_v4.py`` and with FluidAudio's Swift
runtime once a v2 loader is in place.

Requirements
------------
    pip install "pocket-tts==2.0.0"

(or 2.1.0; do **not** use the local 1.0.3-era ``pocket_tts/`` directory in
this repo — it pins a different HF revision and predates the deployed
mlpackage trace.)

Usage
-----
    python export_voice_v2.py speaker.wav -o build/english/constants_bin/speaker.safetensors
    python export_voice_v2.py speaker.wav --name speaker --output-dir build/english/constants_bin/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _require_pocket_tts_2x():
    try:
        from pocket_tts import TTSModel  # noqa: F401
        from pocket_tts.models.tts_model import init_states  # noqa: F401
        from pocket_tts.models.tts_model import audio_read, convert_audio  # noqa: F401
        from pocket_tts.utils.utils import get_predefined_voice  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"pocket-tts >= 2.0.0 is required (got import error: {exc}). "
            "Install with: pip install 'pocket-tts==2.0.0'"
        ) from exc


def export_voice(wav: Path, out: Path, language: str, lsd_decode_steps: int) -> None:
    _require_pocket_tts_2x()
    from pocket_tts import TTSModel
    from pocket_tts.models.tts_model import init_states, audio_read, convert_audio

    logger.info("Loading TTSModel (language=%s)...", language)
    model = TTSModel.load_model(language=language, lsd_decode_steps=lsd_decode_steps)
    model.eval()

    logger.info("Reading %s", wav)
    audio, sr = audio_read(str(wav))
    logger.info("  raw: shape=%s sr=%d", tuple(audio.shape), sr)
    audio = convert_audio(audio, sr, model.config.mimi.sample_rate, 1)
    logger.info("  resampled: shape=%s sr=%d", tuple(audio.shape), model.config.mimi.sample_rate)

    with torch.no_grad():
        audio_prompt = model._encode_audio(audio.unsqueeze(0).to(model.device))
    logger.info("  audio_prompt: shape=%s std=%.4f", tuple(audio_prompt.shape), audio_prompt.std().item())

    prompt = audio_prompt
    if model.flow_lm.insert_bos_before_voice:
        prompt = torch.cat([model.flow_lm.bos_before_voice, prompt], dim=1)

    with torch.no_grad():
        state = init_states(model.flow_lm, batch_size=1, sequence_length=prompt.shape[1])
        model._run_flow_lm_and_increment_step(model_state=state, audio_conditioning=prompt)

    flat: dict[str, torch.Tensor] = {}
    layer_keys = sorted(
        (k for k in state if ".self_attn" in k and "transformer.layers." in k),
        key=lambda k: int(k.split(".layers.")[1].split(".")[0]),
    )
    if len(layer_keys) == 0:
        raise RuntimeError("No transformer.layers.*.self_attn entries in model_state; pocket-tts API changed?")

    for lk in layer_keys:
        s = state[lk]
        flat[f"{lk}/cache"] = s["cache"].to(torch.float32).contiguous().cpu()
        flat[f"{lk}/offset"] = s["offset"].to(torch.int64).contiguous().cpu()

    prompt_len = int(flat[layer_keys[0] + "/offset"].item())
    cache_std = flat[layer_keys[0] + "/cache"].std().item()
    logger.info("  baked cache: layers=%d prompt_len=%d L0_std=%.4f", len(layer_keys), prompt_len, cache_std)

    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(out))
    logger.info("Wrote %s (%d bytes)", out, out.stat().st_size)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bake a v2 KV-cache safetensors voice file from an audio sample.",
    )
    p.add_argument("wav", type=Path, help="Source audio file (wav/mp3/flac/...).")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .safetensors path. Mutually exclusive with --output-dir.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory; filename derived from --name or wav stem.",
    )
    p.add_argument(
        "--name", type=str, default=None,
        help="Voice name for filename when using --output-dir (default: wav stem).",
    )
    p.add_argument(
        "--language", type=str, default="english",
        help="pocket-tts language config (default: english; = english_2026-04 in 2.0.0).",
    )
    p.add_argument(
        "--lsd-decode-steps", type=int, default=8,
        help="LSD decode steps for model load (default: 8, matches deployed pipeline).",
    )
    args = p.parse_args()

    if not args.wav.is_file():
        logger.error("Input not found: %s", args.wav)
        sys.exit(1)

    if args.output is None and args.output_dir is None:
        logger.error("Specify either --output or --output-dir.")
        sys.exit(2)
    if args.output is not None and args.output_dir is not None:
        logger.error("--output and --output-dir are mutually exclusive.")
        sys.exit(2)

    if args.output is not None:
        out = args.output
    else:
        name = args.name or args.wav.stem
        out = args.output_dir / f"{name}.safetensors"

    export_voice(args.wav, out, args.language, args.lsd_decode_steps)


if __name__ == "__main__":
    main()
