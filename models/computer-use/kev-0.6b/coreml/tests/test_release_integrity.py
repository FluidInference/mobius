"""Ensure a parity result cannot authorize a different Kev release package."""

from __future__ import annotations

from pathlib import Path

import pytest

from release_integrity import sha256, validate_release_report

LOCK_PATH = Path(__file__).resolve().parents[1] / "assets.lock.json"


def verified_metadata() -> tuple[dict, dict[str, str]]:
    # Use the checked-in, pinned source manifest as real bytes for the hash-binding test.
    lock_hash = sha256(LOCK_PATH)
    files = {"assets.lock.json": lock_hash}
    report = {
        "package": "kev_0_6b_w8_L128_options32.mlpackage",
        "package_files_sha256": files,
        "source_lock_sha256": lock_hash,
        "length": 128,
        "max_options": 32,
        "compute_units": "cpu-ne",
        "evaluated": 4,
        "agreements": 4,
        "fixture_evaluated": 4,
        "fixture_agreements": 4,
        "max_probability_error": 0.008554,
        "passed": True,
    }
    return report, files


def test_rejects_package_changed_after_verification():
    report, files = verified_metadata()
    validate_release_report(report, report["package"], files, sha256(LOCK_PATH))
    with pytest.raises(ValueError, match="differs"):
        validate_release_report(report, report["package"], {"assets.lock.json": "0" * 64}, sha256(LOCK_PATH))


def test_rejects_failed_or_loose_parity():
    report, files = verified_metadata()
    for change in (
        {"agreements": 3},
        {"max_probability_error": 0.021},
        {"max_probability_error": float("nan")},
        {"fixture_agreements": 3},
        {"evaluated": 3},
    ):
        with pytest.raises(ValueError, match="parity"):
            validate_release_report({**report, **change}, report["package"], files, sha256(LOCK_PATH))


def test_rejects_different_source_or_compute_plan():
    report, files = verified_metadata()
    with pytest.raises(ValueError, match="source lock"):
        validate_release_report(report, report["package"], files, "0" * 64)
    with pytest.raises(ValueError, match=r"CPU\+ANE"):
        validate_release_report({**report, "compute_units": "all"}, report["package"], files, sha256(LOCK_PATH))
