"""Fetch the pinned upstream scorer and fail clearly when Gemma access is absent."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "assets.lock.json").read_text())
SOURCE = ROOT / "build" / "source"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_base_access() -> Path:
    """Check a tiny config before attempting any model-weight download."""
    try:
        return Path(hf_hub_download(LOCK["base_repo"], "config.json", revision=LOCK["base_revision"]))
    except GatedRepoError as error:
        raise PermissionError(
            "Gemma 3 270M access denied. Sign in to Hugging Face, review and accept the terms at "
            "https://huggingface.co/google/gemma-3-270m, then rerun with that approved account. "
            "Do not substitute an unrelated Gemma checkpoint."
        ) from error


def fetch_source(*, include_adapter: bool = True) -> Path:
    """Fetch only the locked public GitHub files, without executing upstream code."""
    SOURCE.mkdir(parents=True, exist_ok=True)
    for relative, expected in LOCK["files"].items():
        if not include_adapter and relative.endswith("adapter_model.safetensors"):
            continue
        target = SOURCE / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/akash-kamat/system-one-gemma/{LOCK['source_revision']}/{relative}"
            with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as output:
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    output.write(block)
        if sha256(target) != expected:
            target.unlink(missing_ok=True)
            raise ValueError(f"Pinned source hash mismatch: {relative}")
    return SOURCE


def fetch_base() -> Path:
    """Download the exact licensed base only after account access is confirmed."""
    check_base_access()
    root = ROOT / "build" / "base"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        cached = Path(hf_hub_download(LOCK["base_repo"], name, revision=LOCK["base_revision"]))
        target = root / name
        if not target.exists():
            target.symlink_to(cached)
    return root


if __name__ == "__main__":
    check_base_access()
    print(fetch_source())
