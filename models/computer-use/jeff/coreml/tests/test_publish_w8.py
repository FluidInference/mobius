"""Check the recorded W8 release decision against the pinned trained model."""

import json
from pathlib import Path

import pytest

from publish import W8_PACKAGE, package_file_hashes, validate_w8_report

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_w8_release_report_passes():
    report = json.loads((ROOT / "reports/w8-validation.json").read_text())
    validate_w8_report(report)


def test_changed_native_choice_cannot_publish():
    report = json.loads((ROOT / "reports/w8-validation.json").read_text())
    report["native_choice_agreement"] = 3
    with pytest.raises(ValueError, match="four native decisions"):
        validate_w8_report(report)


def test_local_w8_package_matches_report_when_present():
    package = ROOT / "build" / W8_PACKAGE
    if not package.exists():
        pytest.skip("local W8 Core ML package is not present")
    report = json.loads((ROOT / "reports/w8-validation.json").read_text())
    assert package_file_hashes(package) == report["package_files_sha256"]
