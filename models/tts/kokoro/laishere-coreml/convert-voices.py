"""Extract Kokoro-82M voice packs (.pt -> .bin flat fp32 [510, 256]).

The laishere 7-stage CoreML graphs + vocab.json are produced from the shared
base `hexgrad/Kokoro-82M` acoustic model and are language-agnostic, so a new
language variant (e.g. ANE-ja/) reuses those bundles unchanged and only needs
its voice packs in FluidAudio's `[510, 256]` flat float32 format.

Unlike the v1.1-zh `convert-voices.py`, this loads each `.pt` tensor directly
(no `KPipeline`), so it needs no per-language G2P dependencies (misaki[ja],
fugashi/MeCab, ...). torch + huggingface_hub only.

Usage:
    # All Japanese voices into an ANE-ja staging dir:
    python convert-voices.py --prefix jf jm --output-dir build/ANE-ja/voices

    # Specific voices:
    python convert-voices.py --only jf_alpha jm_kumo --output-dir /tmp/voices
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "hexgrad/Kokoro-82M"
EXPECTED_SHAPE = (510, 1, 256)


def list_remote_voices(repo_id: str, prefixes: list[str] | None) -> list[str]:
    files = HfApi().list_repo_files(repo_id=repo_id)
    stems = sorted(
        pathlib.Path(f).stem
        for f in files
        if f.startswith("voices/") and f.endswith(".pt")
    )
    if prefixes:
        stems = [s for s in stems if any(s.startswith(p) for p in prefixes)]
    return stems


def main() -> None:
    p = argparse.ArgumentParser(description="Extract Kokoro-82M voice packs to flat .bin")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--prefix", nargs="*", default=None,
                   help="Voice-id prefixes to keep (e.g. jf jm for Japanese)")
    p.add_argument("--only", nargs="*", default=None,
                   help="Explicit voice ids (overrides remote enumeration)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    voices = sorted(args.only) if args.only else list_remote_voices(args.repo_id, args.prefix)
    print(f"Converting {len(voices)} voice(s) from {args.repo_id}: {voices}")

    converted = 0
    for vid in voices:
        out_path = args.output_dir / f"{vid}.bin"
        try:
            pt = hf_hub_download(args.repo_id, f"voices/{vid}.pt")
            tensor = torch.load(pt, weights_only=True)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  [FAIL] {vid}: {type(e).__name__}: {e}")
            continue
        arr = tensor.cpu().numpy().astype(np.float32)
        if arr.shape != EXPECTED_SHAPE:
            print(f"  [FAIL] {vid}: unexpected shape {arr.shape}, want {EXPECTED_SHAPE}")
            continue
        arr = arr.reshape(510, 256)
        out_path.write_bytes(arr.tobytes())
        converted += 1
        print(f"  [{converted}/{len(voices)}] {vid}.bin ({out_path.stat().st_size} bytes)")

    print(f"\nDone. converted={converted} -> {args.output_dir}/")


if __name__ == "__main__":
    main()
