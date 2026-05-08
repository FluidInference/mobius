"""Bootstrap StyleTTS2 LibriTTS ground-truth inference assets.

Run once after `uv sync`:

    uv run python scripts/bootstrap.py

Pulls:
  - vendor/StyleTTS2/                       (git clone of yl4579/StyleTTS2)
  - checkpoints/LibriTTS/config.yml         (yl4579/StyleTTS2-LibriTTS)
  - checkpoints/LibriTTS/epochs_2nd_00020.pth (~771 MB)
  - reference_audio/*.wav                   (yl4579/StyleTTS2-LibriTTS)

Idempotent: skips anything already present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent.parent  # models/tts/styletts2
VENDOR = HERE / "vendor" / "StyleTTS2"
CHECKPOINT_DIR = HERE / "checkpoints" / "LibriTTS"
REFERENCE_DIR = HERE / "reference_audio"

UPSTREAM_REPO = "https://github.com/yl4579/StyleTTS2.git"
HF_REPO = "yl4579/StyleTTS2-LibriTTS"


def clone_vendor() -> None:
    if VENDOR.exists():
        print(f"[1/3] vendor present: {VENDOR}")
        return
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] cloning {UPSTREAM_REPO} -> {VENDOR}")
    subprocess.run(["git", "clone", "--depth", "1", UPSTREAM_REPO, str(VENDOR)], check=True)


def fetch_checkpoint() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in ("config.yml", "epochs_2nd_00020.pth"):
        out = CHECKPOINT_DIR / fn
        if out.exists():
            print(f"[2/3] {fn} present ({out.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[2/3] downloading {fn}")
        cached = hf_hub_download(repo_id=HF_REPO, filename=f"Models/LibriTTS/{fn}")
        shutil.copy(cached, out)
        print(f"      -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def fetch_reference_audio() -> None:
    if REFERENCE_DIR.exists() and any(REFERENCE_DIR.glob("*.wav")):
        n = sum(1 for _ in REFERENCE_DIR.glob("*.wav"))
        print(f"[3/3] reference_audio present ({n} wavs)")
        return
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    print("[3/3] downloading reference_audio.zip")
    cached = hf_hub_download(repo_id=HF_REPO, filename="reference_audio.zip")
    with zipfile.ZipFile(cached) as zf:
        # archive root is "reference_audio/..." — extract to HERE so layout is HERE/reference_audio/*.wav
        zf.extractall(HERE)
    n = sum(1 for _ in REFERENCE_DIR.glob("*.wav"))
    print(f"      -> {REFERENCE_DIR} ({n} wavs)")


def main() -> None:
    clone_vendor()
    fetch_checkpoint()
    fetch_reference_audio()
    print("\nbootstrap complete. run inference:")
    print("  uv run python run_inference.py --output out.wav --seed 0")


if __name__ == "__main__":
    main()
