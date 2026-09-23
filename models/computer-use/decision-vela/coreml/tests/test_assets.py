"""Check the pinned source and real fixtures used by the Core ML exporter."""

import hashlib
import json
from pathlib import Path

import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("model", ["decision-kai", "decision-lex"])
def test_source_lock_and_fixture_integrity(model):
    toolkit = TOOLKIT_ROOT / model / "coreml"
    lock = json.loads((toolkit / "assets.lock.json").read_text())
    assert lock["engine"] == model
    assert lock["full_params_from_current_config_or_metadata"] == 571_909_635
    assert len(lock["selected_weight_files"]) == 4
    assert len({file["path"] for file in lock["selected_weight_files"]}) == 4
    assert sum(file["bytes"] for file in lock["selected_weight_files"]) == 2_287_687_932
    for key, filename in (
        ("decision_fixture_sha256", "upstream-decisions.jsonl"),
        ("system_one_fixture_sha256", "upstream-system-one.json"),
        ("system_one_rows_sha256", "upstream-system-one-rows.jsonl"),
    ):
        digest = hashlib.sha256((toolkit / filename).read_bytes()).hexdigest()
        assert digest == lock[key]
    assert lock["pre_tracker_snapshot_native_manifest_same_as_current"] is True
    assert lock["tracker_renderer_and_served_checkpoint_unverified"] is True
