"""Unit tests validate guards/metadata, not acoustic quality or gradient readiness."""

import copy
import json
import io
import tarfile

import pytest
import torch

from kokoro_training.artifacts import ROOT, verify_assets
from kokoro_training.data import audit_data, audit_records
from kokoro_training.emime import inventory_emime
from kokoro_training.model import CheckpointError, map_checkpoint, select_style, validate_tokens
from kokoro_training.parity import cases_from_manifest, compare_tensors, compare_traces


def test_explicit_legacy_mapping():
    value = torch.ones(2, 1)
    target = "decoder.conv.parametrizations.weight.original0"
    mapped, report = map_checkpoint({target: value}, {"decoder": {"module.conv.weight_g": value}})
    assert report["passed"] and report["renamed_keys"] == 1
    assert torch.equal(mapped[target], value)


@pytest.mark.parametrize(
    "bad", [torch.zeros(3), torch.ones(2, dtype=torch.float64), torch.tensor([float("nan"), 0.0])]
)
def test_invalid_weights_fail(bad):
    with pytest.raises(CheckpointError):
        map_checkpoint({"bert.weight": torch.zeros(2)}, {"bert": {"module.weight": bad}})


def test_unknown_missing_and_collision_fail():
    for checkpoint in [
        {},
        {"other": {"weight": torch.zeros(2)}},
        {"bert": {"weight": torch.zeros(2), "module.weight": torch.zeros(2)}},
    ]:
        with pytest.raises(CheckpointError):
            map_checkpoint({"bert.weight": torch.zeros(2)}, checkpoint)


def test_identity_allowlist_does_not_hide_missing_learned_weights():
    expected = {"decoder.norm.weight": torch.ones(2), "bert.weight": torch.zeros(2)}
    with pytest.raises(CheckpointError) as exc:
        map_checkpoint(expected, {}, {"decoder.norm.weight": torch.ones(2)})
    assert exc.value.report["missing"] == ["bert.weight"]


@pytest.mark.parametrize(
    "ids",
    [[], [0, 0], [0, 178, 0], [0, True, 0], [0, 1, 0, 2, 0], [1, 2, 0], [0] + [1] * 511 + [0]],
)
def test_invalid_token_contract(ids):
    with pytest.raises(ValueError):
        validate_tokens(ids, 178, 512)


def test_style_boundary_rows():
    # Tensor indexing fixture only; never used as a model or speech sample.
    voice = torch.arange(510.0)[:, None, None].expand(510, 1, 256)
    assert select_style(voice, 1)[1] == 0
    assert select_style(voice, 510)[1] == 509
    for n in [0, 511]:
        with pytest.raises(ValueError):
            select_style(voice, n)


def test_different_or_nonfinite_trace_fails():
    assert not compare_tensors(torch.ones(2), torch.ones(3))["passed"]
    assert not compare_tensors(torch.ones(2), torch.tensor([1.0, float("nan")]))["passed"]
    assert not compare_tensors(torch.ones(2), torch.tensor([1.0, 1.0001]))["passed"]
    assert not compare_traces({"audio": torch.ones(2)}, {})["passed"]


def test_missing_assets_fail_without_network(tmp_path):
    with pytest.raises(ValueError):
        verify_assets(tmp_path)


def test_missing_and_empty_dataset_fail(tmp_path):
    assert not audit_data(tmp_path / "missing.jsonl", tmp_path)["passed"]
    assert not audit_records([], None)["passed"]


def record(index=1, **overrides):
    """Deliberately invented metadata fixture, never exported as a dataset."""
    data = {
        "schema_version": 1,
        "id": f"test-{index}",
        "corpus": "unit-test-only",
        "speaker_id": "unit-test-speaker",
        "speaker_gender": "female",
        "speaker_evidence_ref": "unit-test-ref",
        "session_id": f"session-{index}",
        "passage_id": f"passage-{index}",
        "duplicate_group": f"group-{index}",
        "language": "en" if index == 1 else "zh",
        "origin": "real_recording",
        "audio_path": f"unavailable-{index}.wav",
        "audio_sha256": str(index) * 64,
        "sample_rate": 24000,
        "channels": 1,
        "duration_seconds": 1.0,
        "raw_text": f"fixture {index}",
        "normalized_text": f"fixture {index}",
        "split": "train" if index == 1 else "dev",
        "rights": {"status": "approved", "evidence_ref": "unit-test-ref", "scope": "research_only"},
        "qc": {"status": "reviewed", "evidence_ref": "unit-test-ref"},
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"origin": "generated"}, "not a real"),
        (
            {"rights": {"status": "pending", "evidence_ref": "pending", "scope": "unknown"}},
            "rights review",
        ),
        ({"speaker_gender": "unknown"}, "female speaker"),
        ({"duration_seconds": float("nan")}, "nonfinite"),
        ({"split": "unassigned"}, "split not assigned"),
    ],
)
def test_unready_record_rejected(change, reason):
    report = audit_records([record(**change), record(2)], None)
    assert any(reason in e["reason"] for e in report["errors"])
    assert report["training_ready"] is False


@pytest.mark.parametrize(
    "field", ["session_id", "passage_id", "duplicate_group", "audio_sha256", "normalized_text"]
)
def test_split_leakage_rejected(field):
    a, b = record(), record(2)
    b[field] = a[field]
    report = audit_records([a, b], None)
    assert any("crosses splits" in e["reason"] for e in report["errors"])


def test_demo_prompts_cannot_be_training_data():
    manifest = json.loads((ROOT.parent / "coreml/bilingual-demo/manifest.json").read_text())
    report = audit_records([record(normalized_text=manifest["clips"][0]["text"]), record(2)], None)
    assert any("development-only" in e["reason"] for e in report["errors"])


def test_audio_path_escape_rejected(tmp_path):
    report = audit_records([record(audio_path="../outside.wav"), record(2)], tmp_path)
    assert any("escapes" in e["reason"] for e in report["errors"])


def test_two_speakers_not_single_voice():
    report = audit_records([record(), record(2, speaker_id="different")], None)
    assert any("exactly one" in e["reason"] for e in report["errors"])


def test_bad_json_manifest_fails(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("not JSON")
    assert not audit_data(path, tmp_path)["passed"]


@pytest.mark.parametrize("bad", [None, [], {"rights": []}, {"rights": None}])
def test_malformed_records_report_failure(bad):
    assert not audit_records([bad], None)["passed"]


def test_archive_traversal_is_rejected_without_extraction(tmp_path):
    archive = tmp_path / "unsafe.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 4
        tar.addfile(member, io.BytesIO(b"test"))
    with pytest.raises(ValueError, match="unsafe"):
        inventory_emime(archive)
    assert not (tmp_path.parent / "outside.txt").exists()


def test_empty_archive_cannot_pass_inventory(tmp_path):
    archive = tmp_path / "empty.tar.bz2"
    with tarfile.open(archive, "w:bz2"):
        pass
    assert not inventory_emime(archive)["passed"]


def test_saved_tokens_must_match_pinned_vocab(tmp_path):
    baseline = ROOT / ".artifacts/baseline/config.json"
    if not baseline.exists():
        pytest.skip("explicit baseline acquisition required for vocabulary integration check")
    config = json.loads(baseline.read_text())
    source = ROOT.parent / "coreml/bilingual-demo/manifest.json"
    cases = cases_from_manifest(source, config)
    assert len(cases) == 15
    manifest = copy.deepcopy(json.loads(source.read_text()))
    manifest["clips"][0]["input_ids"][1] = 1
    path = tmp_path / "bad-manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        cases_from_manifest(path, config)
