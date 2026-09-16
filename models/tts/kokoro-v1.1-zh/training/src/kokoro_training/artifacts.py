"""Pinned acquisition and portable provenance (no private paths in reports)."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "baseline.lock.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_assets(directory: Path) -> dict:
    lock = json.loads(LOCK.read_text())
    for name, expected in lock["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing or wrong-sized pinned asset: {name}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
    return lock


def fetch_assets(directory: Path) -> dict:
    """Explicit network operation; download only the three locked baseline assets."""
    lock = json.loads(LOCK.read_text())
    for name, expected in lock["files"].items():
        target = directory / name
        if target.is_file() and sha256(target) == expected["sha256"]:
            continue
        if target.exists():
            raise ValueError(f"Refusing to replace mismatched existing asset: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{lock['repo_id']}/resolve/{lock['revision']}/{name}"
        temporary = target.with_suffix(target.suffix + ".partial")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            if (
                temporary.stat().st_size != expected["bytes"]
                or sha256(temporary) != expected["sha256"]
            ):
                raise ValueError(f"Downloaded asset failed verification: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return verify_assets(directory)


def environment() -> dict:
    import torch
    import kokoro

    def command(args):
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip()

    kokoro_root = Path(kokoro.__file__).parent
    data = {
        "schema_version": 1,
        "python": platform.python_version(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["kokoro", "transformers", "numpy", "soundfile"]
        },
        "dependency_lock_sha256": sha256(ROOT / "uv.lock"),
        "baseline_lock_sha256": sha256(LOCK),
        "code_revision": command(["git", "rev-parse", "HEAD"]),
        "working_tree_dirty": bool(command(["git", "status", "--porcelain"])),
        "tool_source_sha256": {
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "kokoro_source_sha256": {p.name: sha256(p) for p in sorted(kokoro_root.glob("*.py"))},
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
    }
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.exists():
        data["container_memory_limit"] = limit.read_text().strip()
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        data["gpu"] = {
            "name": gpu.name,
            "vram_bytes": gpu.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
        data["driver"] = command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
    return data
