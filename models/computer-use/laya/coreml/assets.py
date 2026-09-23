"""Pinned laya checkpoint download and integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "assets.lock.json"
ARTIFACTS = ROOT / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text())


def checkpoint_dir(variant: str) -> Path:
    return ARTIFACTS / variant


def verify_assets(variant: str = "multilingual", download: bool = True) -> dict:
    """Download missing pinned files for one checkpoint variant and verify every hash."""
    lock = load_lock()
    entry = lock["variants"][variant]
    target = checkpoint_dir(variant)
    for name, expected in entry["files"].items():
        path = target / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            subfolder = entry["subfolder"]
            remote = f"{subfolder}/{name}" if subfolder else name
            downloaded = Path(
                hf_hub_download(lock["repo"], remote, revision=lock["revision"], local_dir=str(ARTIFACTS / "_hub"))
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(downloaded.read_bytes())
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{path} sha256 {actual} != pinned {expected}")
    # transformers 5 needs extra_special_tokens as a mapping; the upstream file stores a list.
    config_path = target / "tokenizer" / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        config["extra_special_tokens"] = {f"extra_{i}": token for i, token in enumerate(extra)}
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    return entry


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="multilingual")
    args = parser.parse_args()
    entry = verify_assets(args.variant)
    print(f"Verified {len(entry['files'])} files for {args.variant} at {checkpoint_dir(args.variant)}")
