"""Exercise the published, weight-free Kev request path with the pinned tokenizer."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from huggingface_hub import snapshot_download
from kev.api import SystemOneRequest, to_record
from kev.model import encode
from transformers import AutoTokenizer

from assets import LOCK_PATH
from preprocessing import Shape, prepare_runtime_inputs
from verify import fixtures


@pytest.fixture(scope="module")
def tokenizer():
    lock = json.loads(LOCK_PATH.read_text())
    source = snapshot_download(
        lock["checkpoint"]["repo"],
        revision=lock["checkpoint"]["revision"],
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"],
    )
    return AutoTokenizer.from_pretrained(source, local_files_only=True)


def unlabelled(request: dict) -> dict:
    request = copy.deepcopy(request)
    for question in request["questions"].values():
        question.pop("label", None)
        question.pop("src", None)
    return request


@pytest.mark.parametrize(
    "sample", fixtures(), ids=["tetris-choice", "phishing-noul", "billing-choice", "delivery-score"]
)
def test_unlabelled_runtime_uses_upstream_serving_encoding(tokenizer, sample):
    sample = unlabelled(sample)
    arrays, encoded, metadata, parsed = prepare_runtime_inputs(tokenizer, sample, Shape())
    record, expected_metadata = to_record(SystemOneRequest.model_validate(sample))
    direct = encode(tokenizer, record, max_state=8192, max_branch=16384, option_isolation=False)
    assert encoded["ids"] == direct["ids"]
    assert metadata == expected_metadata
    assert parsed.questions
    assert arrays["input_ids"].shape == (1, 128)
    assert arrays["option_map"].shape == (1, 32, 128)
    np.testing.assert_array_equal(arrays["input_ids"][0, : len(encoded["ids"])], encoded["ids"])
    assert arrays["decide_map"][0, 0, encoded["decide_idx"][0]] == 1
    for option, position in enumerate(encoded["opt_idx"][0]):
        assert arrays["option_map"][0, option, position] == 1


def test_runtime_rejects_multiple_questions(tokenizer):
    request = unlabelled(fixtures()[0])
    request["questions"]["second"] = copy.deepcopy(request["questions"]["q"])
    with pytest.raises(ValueError, match="exactly one question"):
        prepare_runtime_inputs(tokenizer, request, Shape())


def test_runtime_rejects_overlong_question_branch(tokenizer):
    request = unlabelled(fixtures()[0])
    request["questions"]["q"]["instructions"] = "Which placement? " * 200
    with pytest.raises(ValueError, match="branch"):
        prepare_runtime_inputs(tokenizer, request, Shape())
