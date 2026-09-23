"""Validate optional W8 packages against pinned native-result reports when present."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from publish import _validated_embedding_w8

COMPUTER_USE = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("model", "artifact_dir", "kinds"),
    [
        ("decision-kai", "embedding-w8", {"choice", "noul", "score"}),
        ("decision-lex", "lex-e8", {"noul", "score"}),
    ],
)
def test_real_embedding_w8_packages_pass_release_gate(model, artifact_dir, kinds):
    toolkit = COMPUTER_USE / model / "coreml"
    artifacts = toolkit / "build" / artifact_dir
    if not artifacts.exists():
        pytest.skip("local converted Core ML packages are required")
    results = _validated_embedding_w8(toolkit, artifacts)
    assert set(results) == kinds
    assert all(item["validation"]["choice_agreement"] for item in results.values())
