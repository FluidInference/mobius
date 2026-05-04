"""Batch-convert all v1.1-zh voice packs (.pt → .bin flat fp32 [510, 256]).

Enumerates every voice in `hexgrad/Kokoro-82M-v1.1-zh/voices/*.pt` via the HF
API and converts each into the `[510, 256]` flat float32 format consumed by
FluidAudio's `KokoroAneVoicePack`.

Usage:
    uv run python convert-voices.py --output-dir build/ANE-zh/voices
    uv run python convert-voices.py --output-dir /tmp/voices --skip "zf_001 zm_009"
"""
import argparse
import pathlib

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from kokoro import KModel
from kokoro.pipeline import KPipeline


REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"


def list_remote_voices(repo_id: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    voices = sorted(
        pathlib.Path(f).stem
        for f in files
        if f.startswith("voices/") and f.endswith(".pt")
    )
    return voices


def main():
    p = argparse.ArgumentParser(description="Batch convert v1.1-zh voice packs to .bin")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--skip", nargs="*", default=[],
                   help="Voice ids to skip (e.g. ones already converted)")
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to these voice ids (overrides remote enumeration)")
    p.add_argument("--lang", default="z")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        voices = sorted(args.only)
    else:
        print(f"Enumerating voices in {args.repo_id}...")
        voices = list_remote_voices(args.repo_id)
    print(f"Found {len(voices)} voices: {voices}")

    skip = set(args.skip)
    pending = [v for v in voices if v not in skip]
    print(f"Skipping {sorted(skip)}; converting {len(pending)} voices.")

    # Pipeline + KModel reused across all voices
    print(f"Loading KModel + KPipeline ({args.repo_id})...")
    model = KModel(repo_id=args.repo_id); model.eval()
    pipe = KPipeline(lang_code=args.lang, repo_id=args.repo_id, model=model)

    converted = 0
    skipped_existing = 0
    for vid in pending:
        out_path = args.output_dir / f"{vid}.bin"
        if out_path.exists():
            skipped_existing += 1
            print(f"  [skip-existing] {vid}.bin already at {out_path}")
            continue
        try:
            voice = pipe.load_voice(vid).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"  [FAIL] {vid}: {type(e).__name__}: {e}")
            continue
        if voice.shape != (510, 1, 256):
            print(f"  [FAIL] {vid}: unexpected shape {voice.shape}, want (510, 1, 256)")
            continue
        voice = voice.reshape(510, 256)
        out_path.write_bytes(voice.tobytes())
        converted += 1
        print(f"  [{converted}/{len(pending)}] {vid}.bin ({out_path.stat().st_size} bytes)")

    print(f"\nDone. converted={converted} skipped_existing={skipped_existing}")
    print(f"Wrote to {args.output_dir}/")


if __name__ == "__main__":
    main()
