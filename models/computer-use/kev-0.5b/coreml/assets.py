"""Download the exact Kev adapter and Qwen base revisions used by this conversion."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from huggingface_hub import snapshot_download
from kev.checkpoint import Checkpoint, LoadOptions

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "assets.lock.json"


def load_model():
    lock = json.loads(LOCK_PATH.read_text())
    checkpoint = lock["checkpoint"]
    base = lock["base"]
    checkpoint_path = snapshot_download(checkpoint["repo"], revision=checkpoint["revision"])
    base_path = snapshot_download(base["repo"], revision=base["revision"])
    loader = Checkpoint(checkpoint_path)
    loader.meta = replace(loader.meta, base=base_path, base_revision=None)
    tokenizer, model = loader.load("cpu", LoadOptions())
    return lock, tokenizer, model
