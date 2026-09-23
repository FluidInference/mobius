"""Bind a Kev parity report to the exact Core ML package being published."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_file_hashes(package: Path) -> dict[str, str]:
    if not package.is_dir():
        raise FileNotFoundError(package)
    hashes = {
        path.relative_to(package).as_posix(): sha256(path) for path in sorted(package.rglob("*")) if path.is_file()
    }
    if not hashes:
        raise ValueError(f"empty Core ML package: {package}")
    return hashes


def validate_release_report(report: dict, package_name: str, package_hashes: dict[str, str], lock_sha256: str) -> None:
    if report.get("package") != package_name or report.get("source_lock_sha256") != lock_sha256:
        raise ValueError("Kev parity report names a different package or source lock")
    if (report.get("length"), report.get("max_options"), report.get("compute_units")) != (128, 32, "cpu-ne"):
        raise ValueError("Kev release requires L128/options32 CPU+ANE parity")
    if (
        report.get("fixture_evaluated") != 4
        or report.get("fixture_agreements") != 4
        or report.get("evaluated", 0) < 4
        or report.get("agreements") != report.get("evaluated")
        or not math.isfinite(report.get("max_probability_error", float("inf")))
        or report.get("max_probability_error", float("inf")) > 0.02
        or report.get("passed") is not True
    ):
        raise ValueError("Kev release failed native decision or probability parity")
    if not package_hashes or report.get("package_files_sha256") != package_hashes:
        raise ValueError("Kev package differs from the one that passed parity")
