"""Bind the optional embedding-W8 package to selected native-parity evidence."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/e8-selected20.json"


def test_recorded_e8_report_passes_selected_native_gate():
    report = json.loads(REPORT.read_text())
    assert report["evaluated"] == report["choice_agreement"] == 19
    assert report["maximum_probability_error_rounded"] <= report["probability_error_gate"]
    assert sum("skipped" in row for row in report["cases"]) == 4
    assert len(report["package_files_sha256"]) == 3


def test_local_e8_package_matches_recorded_hashes_when_present():
    report = json.loads(REPORT.read_text())
    package = ROOT / "build" / report["published_package_name"]
    if not package.exists():
        pytest.skip("local compressed Core ML package is not present")
    for name, expected in report["package_files_sha256"].items():
        digest = hashlib.sha256()
        with (package / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected, name
