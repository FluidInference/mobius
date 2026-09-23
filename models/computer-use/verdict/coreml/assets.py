"""Fetch and verify the five pinned Verdict runtime assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "assets.lock.json").read_text())
SOURCE = ROOT / "build" / "source"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(download: bool = True, required: tuple[str, ...] | None = None) -> Path:
    """Return the local checkpoint only after all hashes match the lock."""
    SOURCE.mkdir(parents=True, exist_ok=True)
    names = required or tuple(LOCK["files"])
    for name in names:
        expected = LOCK["files"][name]
        path = SOURCE / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            cached = Path(hf_hub_download(LOCK["source_repo"], name, revision=LOCK["source_revision"]))
            path.symlink_to(cached)
        if sha256(path) != expected:
            raise ValueError(f"{name}: hash mismatch against pinned source revision")
    return SOURCE


if __name__ == "__main__":
    print(verify_assets())
