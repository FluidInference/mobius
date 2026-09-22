"""Read pinned model and dataset revisions for deterministic training."""

from __future__ import annotations

import json
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parent / "assets.lock.json"


def load_lock() -> dict:
    """Load the checked-in asset revision lock."""
    return json.loads(LOCK_PATH.read_text())


def model_revision(repo: str) -> str:
    """Return the pinned revision for the base model."""
    entry = load_lock()["base_model"]
    if entry["repo"] != repo:
        raise KeyError(f"unlocked model: {repo}")
    return entry["revision"]


def dataset_revision(repo: str) -> str:
    """Return the pinned revision for a training dataset."""
    return load_lock()["datasets"][repo]
