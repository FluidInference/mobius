"""Integration checks against the actual pinned Kai or Lex native release."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typed_coreml import KINDS, compare_native, load_native


@pytest.fixture(scope="module")
def native_and_records():
    native_dir = os.environ.get("DECISION_NATIVE_DIR")
    toolkit_dir = os.environ.get("DECISION_TOOLKIT_DIR")
    if not native_dir or not toolkit_dir:
        pytest.skip("Set DECISION_NATIVE_DIR and DECISION_TOOLKIT_DIR to the pinned real model")
    toolkit = Path(toolkit_dir)
    lock = json.loads((toolkit / "assets.lock.json").read_text())
    manifest = (Path(native_dir) / "MANIFEST.json").read_bytes()
    import hashlib

    assert hashlib.sha256(manifest).hexdigest() == lock["native_manifest_sha256"]
    model, collator, _ = load_native(Path(native_dir))
    records = []
    for fixture in ("upstream-decisions.jsonl", "upstream-system-one-rows.jsonl"):
        records.extend(json.loads(line) for line in (toolkit / fixture).read_text().splitlines() if line)
    return model, collator, records


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("marker_map", (False, True))
def test_typed_path_matches_full_native_with_padding(native_and_records, kind, marker_map):
    model, collator, records = native_and_records
    selected = [record for record in records if record["question"]["type"].lower() == kind]
    assert len(selected) == 2
    actual = collator(selected)[0]["marker_positions"].shape[1]
    report = compare_native(
        model, collator, selected, kind, tokens=128, candidates=actual + 2,
        marker_map=marker_map,
    )
    assert report["max_abs_logit_error"] <= 1e-4


def test_model_has_all_trained_paths(native_and_records):
    model, _, _ = native_and_records
    assert sum(parameter.numel() for parameter in model.parameters()) == 571_909_635
    assert len(model.choice_blocks) == 22
    assert len(model.score_blocks) == 22
    assert set(model.heads) == set(KINDS)
