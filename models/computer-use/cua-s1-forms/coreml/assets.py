"""Fetch and verify pinned upstream weights, data, and reference implementation."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "assets.lock.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset_lock() -> dict:
    return json.loads(LOCK_PATH.read_text())


def verify_assets() -> dict:
    """Fail on missing or modified inputs instead of silently changing a benchmark."""
    lock = asset_lock()
    for entry in lock["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run `uv run assets.py` first")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    return lock


def load_demo() -> list[dict]:
    lock = verify_assets()
    rows = [json.loads(line) for line in (ROOT / lock["evaluation_file"]).read_text().splitlines() if line.strip()]
    if len(rows) != lock["evaluation_rows"]:
        raise ValueError("The pinned demo row count changed")
    return rows


def load_reference():
    """Load the unmodified upstream architecture with its signed real state dict."""
    verify_assets()
    vendor = str(ROOT / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from cua_s1.model import load_checkpoint

    return load_checkpoint(ROOT / "artifacts/cua-s1-forms.safetensors", "cpu")


def prepare_assets() -> None:
    for entry in asset_lock()["files"]:
        path = ROOT / entry["path"]
        if path.is_file():
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Existing file differs from pinned source: {path}")
            continue
        with urllib.request.urlopen(entry["url"], timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Downloaded SHA-256 mismatch: {entry['url']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(content)
        temporary.replace(path)
        print(f"Downloaded {entry['path']} ({len(content):,} bytes)", flush=True)
    verify_assets()
    print("Pinned source, weights, configuration, and 196-row demo verified.")


if __name__ == "__main__":
    prepare_assets()
