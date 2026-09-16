"""Failure modes in data boundaries, scoring, and checkpoint persistence."""

import json
from pathlib import Path

import pytest
import torch

from kokoro_training.artifacts import sha256
from kokoro_training.evaluate import error_counts, normalize
from kokoro_training.train import atomic_save, load_data
from kokoro_training.features import MelFeatures
from kokoro_training.supervised import DeterministicLeftReflection


def test_deterministic_reflection_matches_upstream_values_and_gradients():
    values = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 3, 12, dtype=torch.float64)
    expected = torch.nn.functional.pad(values, (1, 0), mode="reflect")
    actual = DeterministicLeftReflection()(values)
    assert torch.equal(actual, expected)
    original_grad = torch.autograd.grad((expected * weights).sum(), values)[0]
    fixed_grad = torch.autograd.grad((actual * weights).sum(), values)[0]
    assert torch.equal(original_grad, fixed_grad)


def test_mel_reflection_preserves_target_values():
    audio = torch.randn(1, 6000)
    mel = MelFeatures()
    original = mel.bank @ torch.stft(audio, n_fft=2048, hop_length=300, win_length=1200,
                                   window=mel.window, center=True, return_complex=True).abs().square()
    assert torch.equal(mel(audio), original)


def test_asr_normalization_preserves_lexical_errors():
    assert normalize("兒童，有二十二個人！", "zh") == "儿童有二十二个人"
    assert error_counts("The API is ready.", "the api is ready", "en")["errors"] == 0
    assert error_counts("二十三元", "十三元", "zh")["deletions"] == 1
    assert error_counts("Open API", "open happy", "en")["substitutions"] == 1
    assert error_counts("三点 meeting", "三点米厅", "mixed")["errors"] > 0


def test_atomic_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "last.pt"
    atomic_save({"step": 1, "tensor": torch.arange(5)}, path)
    atomic_save({"step": 2, "tensor": torch.arange(3)}, path)
    saved = torch.load(path, weights_only=True)
    assert saved["step"] == 2
    assert torch.equal(saved["tensor"], torch.arange(3))
    assert not path.with_suffix(".tmp").exists()


def write_manifest(directory: Path, records):
    manifest = directory / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    (directory / "data-contract.json").write_text(json.dumps({
        "speaker": "declared", "manifest_sha256": sha256(manifest)}))


def records_with_file(tmp_path):
    # A file-integrity fixture only: never passed into acoustic training.
    records = []
    for i, split in enumerate(("train", "dev")):
        target = tmp_path / f"target-{i}.pt"
        torch.save({"integrity_test_only": torch.tensor([i])}, target)
        records.append({"id": str(i), "origin": "real_recording", "speaker_id": "declared",
                        "passage_id": f"passage-{i}", "source_sha256": f"source-{i}",
                        "normalized_text": f"integrity fixture {i}",
                        "split": split, "targets": target.name, "targets_sha256": sha256(target)})
    return records


@pytest.mark.parametrize("field", ["passage_id", "source_sha256", "targets_sha256", "normalized_text"])
def test_reject_cross_split_leakage(tmp_path, field):
    records = records_with_file(tmp_path)
    records[1][field] = records[0][field]
    write_manifest(tmp_path, records)
    with pytest.raises(ValueError, match="leakage"):
        load_data(tmp_path)


def test_reject_modified_acoustic_targets(tmp_path):
    records = records_with_file(tmp_path)
    write_manifest(tmp_path, records)
    (tmp_path / "target-0.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        load_data(tmp_path)


def test_reject_modified_manifest(tmp_path):
    write_manifest(tmp_path, records_with_file(tmp_path))
    with (tmp_path / "manifest.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="manifest checksum"):
        load_data(tmp_path)
