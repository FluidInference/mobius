"""Check the release gate against the recorded real-model validation reports."""

import json
from pathlib import Path

import pytest

from decision import SOURCE_REVISION
from release_integrity import package_file_hashes, validate_release_reports

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "lfm350_rlcd_fp16_L256_B8_V16.mlpackage"


def reports():
    conversion = json.loads((ROOT / "reports/fp16-conversion.json").read_text())
    parity = json.loads((ROOT / "reports/fp16-all-upstream9.json").read_text())
    return conversion, parity


def test_recorded_release_passes_gate():
    conversion, parity = reports()
    validate_release_reports(conversion, parity, PACKAGE, SOURCE_REVISION)


def test_different_artifact_or_failed_case_cannot_publish():
    conversion, parity = reports()
    parity["package_files_sha256"] = {"Manifest.json": "different"}
    with pytest.raises(ValueError, match="same package file hashes"):
        validate_release_reports(conversion, parity, PACKAGE, SOURCE_REVISION)

    conversion, parity = reports()
    parity["cases"][0]["same_selected_values"] = False
    with pytest.raises(ValueError, match="changes a native decision"):
        validate_release_reports(conversion, parity, PACKAGE, SOURCE_REVISION)


def test_local_package_matches_recorded_release_when_present():
    package = ROOT / "build" / PACKAGE
    if not package.exists():
        pytest.skip("the validated 709 MB Core ML package is not present")
    conversion, _ = reports()
    assert package_file_hashes(package) == conversion["package_files_sha256"]
