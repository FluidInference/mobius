"""PyTorch baseline for Chatterbox Multilingual (23-lang).

Renders reference wavs with the stock pipeline (built-in voice from conds.pt)
so CoreML exports have ground truth to match.

Usage:
    uv run python baseline.py --out-dir build/baseline
"""

import argparse
from pathlib import Path

import soundfile as sf
import torch

SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog near the river bank.",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund am Flussufer.",
    "fr": "Le renard brun rapide saute par-dessus le chien paresseux près de la rivière.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("build/baseline"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = ChatterboxMultilingualTTS.from_pretrained(device=args.device)
    print(f"loaded; sr={model.sr}, langs={sorted(model.get_supported_languages())}")

    for lang, text in SENTENCES.items():
        torch.manual_seed(args.seed)
        wav = model.generate(text, language_id=lang)
        out = args.out_dir / f"baseline_{lang}.wav"
        sf.write(out, wav.squeeze(0).numpy(), model.sr)
        print(f"{lang}: {wav.shape[-1] / model.sr:.2f}s -> {out}")


if __name__ == "__main__":
    main()
