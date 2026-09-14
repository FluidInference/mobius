"""PyTorch baseline for Chatterbox Nano (GPT2-small T3 + meanflow S3Gen).

Renders reference wavs with the stock pipeline (built-in voice from conds.pt)
so CoreML exports have ground truth to match. English-only; paralinguistic
tags ([chuckle], [laugh], ...) are part of the tokenizer vocab.

Usage:
    uv run python baseline-nano.py --out-dir build/baseline-nano
"""

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch

SENTENCES = {
    "plain": "The quick brown fox jumps over the lazy dog near the river bank.",
    "tags": "Hi there, Sarah here from MochaFone calling you back [chuckle], have you got one minute to chat about the billing issue?",
    "long": "On device speech synthesis has come a long way in the last few years, and small models now sound surprisingly natural even on older phones.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("build/baseline-nano"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.nano_ckpt import nano_ckpt_dir
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    t0 = time.time()
    model = ChatterboxTurboTTS.from_local(nano_ckpt_dir(), args.device, nano=True)
    print(f"loaded {model.model_label} in {time.time() - t0:.1f}s; sr={model.sr}")

    for name, text in SENTENCES.items():
        torch.manual_seed(args.seed)
        t0 = time.time()
        wav = model.generate(text)
        dur = wav.shape[-1] / model.sr
        wall = time.time() - t0
        out = args.out_dir / f"baseline_{name}.wav"
        sf.write(out, wav.squeeze(0).numpy(), model.sr)
        print(f"{name}: {dur:.2f}s audio in {wall:.1f}s wall (RTFx {dur / wall:.2f}) -> {out}")


if __name__ == "__main__":
    main()
