"""Real-tokenizer regression test for unlabelled Kev 0.5B requests."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub import snapshot_download
from kev.api import SystemOneRequest, to_record
from kev.model import encode
from transformers import AutoTokenizer

from preprocessing import Shape, prepare_inputs, prepare_runtime_inputs
from runtime import stage_symlinked_package
from verify import fixtures


@pytest.fixture(scope="module")
def tokenizer():
    assets = snapshot_download(
        "FluidInference/kev-0-5b-coreml",
        revision="9df58dd3f4b3c12b89b30e46c2e520114f3e124d",
        allow_patterns=["tokenizer/*"],
        local_files_only=True,
    )
    return AutoTokenizer.from_pretrained(f"{assets}/tokenizer", local_files_only=True)


@pytest.mark.parametrize("fixture_index", range(3))
def test_unlabelled_serving_encoding_matches_upstream(tokenizer, fixture_index):
    labelled = fixtures()[fixture_index]
    request = copy.deepcopy(labelled)
    request["questions"]["q"].pop("label")
    request["questions"]["q"].pop("src")
    arrays, packed, metadata, parsed = prepare_runtime_inputs(tokenizer, request, Shape())
    record, expected_metadata = to_record(SystemOneRequest.model_validate(request))
    direct = encode(tokenizer, record, max_state=8192, max_branch=16384, option_isolation=False)
    assert parsed.questions
    assert metadata == expected_metadata
    assert packed["ids"] == direct["ids"]
    np.testing.assert_array_equal(arrays["input_ids"][0, : len(packed["ids"])], packed["ids"])
    formatter = SimpleNamespace(encode=lambda tok, rec, **kw: encode(tok, rec, option_isolation=False, **kw))
    labelled_arrays, _ = prepare_inputs(formatter, tokenizer, labelled, Shape())
    for name, serving_value in arrays.items():
        np.testing.assert_array_equal(serving_value, labelled_arrays[name])


def test_unlabelled_serving_rejects_multiple_questions(tokenizer):
    request = copy.deepcopy(fixtures()[0])
    request["questions"]["q"].pop("label")
    request["questions"]["q"].pop("src")
    request["questions"]["second"] = copy.deepcopy(request["questions"]["q"])
    with pytest.raises(ValueError, match="exactly one question"):
        prepare_runtime_inputs(tokenizer, request, Shape())


def test_hub_symlinks_are_staged_as_real_package_files(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"trained model bytes")
    package = tmp_path / "model.mlpackage"
    package.mkdir()
    (package / "weight.bin").symlink_to(blob)
    staged, temporary = stage_symlinked_package(package)
    try:
        assert temporary is not None
        assert staged != package
        assert (staged / "weight.bin").read_bytes() == blob.read_bytes()
        assert not (staged / "weight.bin").is_symlink()
    finally:
        temporary.cleanup()
