"""Provenance helpers for reproducible Community-1 Core ML releases."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


UPSTREAM_REPOSITORY = "pyannote/speaker-diarization-community-1"
CONVERSION_REPOSITORY = "https://github.com/FluidInference/mobius"
MATERIAL_INPUTS = (
    "config.yaml",
    "segmentation/pytorch_model.bin",
    "embedding/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def require_commit_sha(value: str, label: str) -> str:
    """Reject mutable refs such as ``main`` and abbreviated commit IDs."""
    normalized = value.strip().lower()
    if not _COMMIT_PATTERN.fullmatch(normalized):
        raise ValueError(f"{label} must be a full 40-character Git commit SHA")
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: Path, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory_paths(root: Path, paths: Iterable[str]) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for relative_path in sorted(paths):
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Material input not found: {path}")
        inventory.append(describe_file(path, root))
    return inventory


def inventory_tree(root: Path, excluding: Iterable[str] = ()) -> list[dict[str, object]]:
    excluded = set(excluding)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]
    return [describe_file(path, root) for path in sorted(files)]


def git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return require_commit_sha(result.stdout.strip(), "conversion revision")


def git_is_dirty(repository: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def verified_git_revision(repository: Path, requested: str | None = None) -> str:
    """Return HEAD only when it exactly describes a clean converter checkout."""
    head = git_revision(repository)
    if git_is_dirty(repository):
        raise RuntimeError("Refusing to record provenance from a dirty converter checkout")
    if requested is None:
        return head

    claimed = require_commit_sha(requested, "conversion revision")
    if claimed != head:
        raise ValueError(
            f"conversion revision {claimed} does not match converter checkout HEAD {head}"
        )
    return head


def require_disjoint_paths(work_dir: Path, release_dir: Path) -> None:
    """Reject layouts where intermediate files would pollute the release."""
    if work_dir == release_dir or work_dir in release_dir.parents or release_dir in work_dir.parents:
        raise ValueError("work directory and release directory must not overlap")


def _command_output(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output or None


def environment_details() -> dict[str, object]:
    package_names = (
        "coremltools",
        "huggingface-hub",
        "numpy",
        "pyannote-audio",
        "scipy",
        "torch",
        "torchaudio",
    )
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": _command_output(["sw_vers", "-productVersion"]),
        "xcode": _command_output(["xcodebuild", "-version"]),
        "coremlcompiler": _command_output(["xcrun", "--find", "coremlcompiler"]),
        "packages": packages,
    }


def build_manifest(
    *,
    model_root: Path,
    release_root: Path,
    upstream_repository: str,
    upstream_revision: str,
    conversion_revision: str,
    commands: Sequence[Sequence[str]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC-BY-4.0",
        "source": {
            "repository": upstream_repository,
            "revision": require_commit_sha(upstream_revision, "upstream revision"),
            "files": inventory_paths(model_root, MATERIAL_INPUTS),
        },
        "conversion": {
            "repository": CONVERSION_REPOSITORY,
            "revision": require_commit_sha(conversion_revision, "conversion revision"),
            "commands": [list(command) for command in commands],
        },
        "environment": environment_details(),
        "artifacts": inventory_tree(release_root, excluding=("provenance.json",)),
    }


def write_manifest(manifest: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
