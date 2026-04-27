"""Dump benchmark texts + G2P phonemes + vocab + voice pack into a directory.

The Swift / iOS consumer uses precomputed phonemes from JSON instead of
running G2P on-device. Run after editing TEXTS in benchmark.py:

    uv run python dump-benchmark-data.py --output-dir build/laishere-kokoro

Writes:
    af_heart.bin           voice pack [510, 256] flat float32
    vocab.json             phoneme→token id map (177 entries)
    benchmark_data.json    benchmark cases with precomputed phonemes
"""
import argparse
import json
import pathlib

import numpy as np
from huggingface_hub import hf_hub_download

from kokoro.pipeline import KPipeline

from benchmark import TEXTS, phonemize_for_benchmark


VOICE = "af_heart"
LANG = "a"


def main():
    parser = argparse.ArgumentParser(description="Dump vocab + voice pack + benchmark data")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True,
                        help="Directory to write af_heart.bin, vocab.json, benchmark_data.json into")
    parser.add_argument("--voice", default=VOICE, help=f"Voice id (default: {VOICE})")
    parser.add_argument("--lang", default=LANG, help=f"Kokoro lang code (default: {LANG})")
    args = parser.parse_args()
    voice_id = args.voice
    lang_code = args.lang
    resources = args.output_dir
    resources.mkdir(parents=True, exist_ok=True)
    # Phonemize texts via Kokoro G2P
    pipe = KPipeline(lang_code=lang_code, model=False)
    cases = []
    for i, text in enumerate(TEXTS):
        phonemes = phonemize_for_benchmark(pipe, text)
        cases.append({
            "id": i,
            "text": text,
            "phonemes": phonemes,
            "n_phonemes": len(phonemes),
        })
        print(f"  case {i}: T_enc={len(phonemes):3d}  '{text[:60]}{'...' if len(text) > 60 else ''}'")

    # Voice pack [510, 256] flat float32 → bin
    voice = pipe.load_voice(voice_id).cpu().numpy().astype(np.float32)
    assert voice.shape == (510, 1, 256), f"unexpected voice shape {voice.shape}"
    voice = voice.reshape(510, 256)
    voice_path = resources / f"{voice_id}.bin"
    voice_path.write_bytes(voice.tobytes())
    print(f"  voice {voice_id}: {voice.shape} → {voice_path.name} ({voice_path.stat().st_size} bytes)")

    # Vocab → vocab.json (from HF config, no weights needed)
    config_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="config.json")
    config = json.loads(pathlib.Path(config_path).read_text())
    vocab = config["vocab"]
    vocab_path = resources / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))
    print(f"  vocab: {len(vocab)} entries → {vocab_path.name}")

    # Benchmark cases → benchmark_data.json
    data = {
        "voice": voice_id,
        "lang": lang_code,
        "sample_rate": 24000,
        "cases": cases,
    }
    out = resources / "benchmark_data.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  cases: {len(cases)} → {out.name}")

    print(f"\nWrote to {resources}/")


if __name__ == "__main__":
    main()
