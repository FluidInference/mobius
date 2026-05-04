"""Dump benchmark texts + G2P phonemes + vocab + voice pack into a directory.

The Swift / iOS consumer uses precomputed phonemes from JSON instead of
running G2P on-device. Run after editing TEXTS in benchmark.py:

    uv run python dump-benchmark-data.py --output-dir build/kokoro-v1.1-zh

Writes (per --voice flag, defaults to zf_001 + zm_009 if --all-voices):
    <voice>.bin            voice pack [510, 256] flat float32
    vocab.json             phoneme→token id map (171 entries, v1.1-zh Bopomofo+IPA+digit)
    benchmark_data.json    benchmark cases with precomputed phonemes
"""
import argparse
import json
import pathlib

import numpy as np
from huggingface_hub import hf_hub_download

from kokoro.pipeline import KPipeline

from benchmark import TEXTS, phonemize_for_benchmark


VOICE = "zf_001"
LANG = "z"
REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
DEFAULT_VOICES = ["zf_001", "zm_009"]  # 1 female + 1 male Mandarin


def main():
    parser = argparse.ArgumentParser(description="Dump vocab + voice pack + benchmark data (v1.1-zh)")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True,
                        help="Directory to write *.bin, vocab.json, benchmark_data.json into")
    parser.add_argument("--voice", default=None,
                        help=f"Single voice id (overrides --voices). Default per --voices.")
    parser.add_argument("--voices", nargs="+", default=DEFAULT_VOICES,
                        help=f"Voice ids to dump (default: {' '.join(DEFAULT_VOICES)})")
    parser.add_argument("--lang", default=LANG, help=f"Kokoro lang code (default: {LANG})")
    parser.add_argument("--repo-id", default=REPO_ID, help=f"HF repo id (default: {REPO_ID})")
    args = parser.parse_args()
    voice_ids = [args.voice] if args.voice else args.voices
    lang_code = args.lang
    resources = args.output_dir
    resources.mkdir(parents=True, exist_ok=True)

    # Phonemize texts via misaki[zh] G2P (KPipeline lang_code='z')
    from kokoro import KModel
    pt_model = KModel(repo_id=args.repo_id); pt_model.eval()
    pipe = KPipeline(lang_code=lang_code, repo_id=args.repo_id, model=pt_model)
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

    # Voice packs [510, 256] flat float32 → .bin per voice
    for voice_id in voice_ids:
        voice = pipe.load_voice(voice_id).cpu().numpy().astype(np.float32)
        assert voice.shape == (510, 1, 256), f"unexpected voice shape {voice.shape}"
        voice = voice.reshape(510, 256)
        voice_path = resources / f"{voice_id}.bin"
        voice_path.write_bytes(voice.tobytes())
        print(f"  voice {voice_id}: {voice.shape} → {voice_path.name} ({voice_path.stat().st_size} bytes)")

    # Vocab → vocab.json (from v1.1-zh HF config, no weights needed)
    config_path = hf_hub_download(repo_id=args.repo_id, filename="config.json")
    config = json.loads(pathlib.Path(config_path).read_text())
    vocab = config["vocab"]
    vocab_path = resources / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False))
    print(f"  vocab: {len(vocab)} entries → {vocab_path.name}")

    # Benchmark cases → benchmark_data.json
    data = {
        "voices": voice_ids,
        "lang": lang_code,
        "repo_id": args.repo_id,
        "sample_rate": 24000,
        "cases": cases,
    }
    out = resources / "benchmark_data.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  cases: {len(cases)} → {out.name}")

    print(f"\nWrote to {resources}/")


if __name__ == "__main__":
    main()
