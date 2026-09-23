"""Bind an LFM Core ML package to its native-parity evidence before release."""

from __future__ import annotations

import hashlib
from pathlib import Path


def package_file_hashes(package: Path) -> dict[str, str]:
    if not package.is_dir():
        raise FileNotFoundError(package)
    hashes = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.relative_to(package).as_posix()] = digest.hexdigest()
    if not hashes:
        raise ValueError(f"empty Core ML package: {package}")
    return hashes


def validate_release_reports(conversion: dict, parity: dict, package_name: str, source_revision: str) -> None:
    if conversion.get("model") != package_name or parity.get("package") != package_name:
        raise ValueError("conversion and parity reports must name the release package")
    if conversion.get("source_revision") != source_revision or parity.get("source_revision") != source_revision:
        raise ValueError("conversion and parity reports must use the pinned source revision")
    if (conversion.get("precision"), conversion.get("length"), conversion.get("candidate_batch"),
        conversion.get("max_value_tokens")) != ("float16", 256, 8, 16):
        raise ValueError("conversion report does not describe the release shape")
    if parity.get("compute_units") != "all" or parity.get("fixture_set") != "upstream-rlcd-tasks":
        raise ValueError("release parity must use the selected upstream cases on ALL compute units")
    cases = parity.get("cases", [])
    if len(cases) != 9 or len({case.get("case_id") for case in cases}) != 9:
        raise ValueError("release parity must contain nine distinct upstream cases")
    if not parity.get("same_selected_values") or not all(case.get("same_selected_values") for case in cases):
        raise ValueError("release changes a native decision")
    if parity.get("max_log_likelihood_error", float("inf")) > 0.1:
        raise ValueError("release exceeds the selected-case log-likelihood tolerance")
    hashes = conversion.get("package_files_sha256")
    if not hashes or hashes != parity.get("package_files_sha256"):
        raise ValueError("conversion and parity must bind to the same package file hashes")
