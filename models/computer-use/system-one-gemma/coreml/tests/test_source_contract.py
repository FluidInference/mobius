"""Small deterministic checks for the audited source contract, without loading Gemma."""

from __future__ import annotations

import json
import struct

import pytest

from assets import LOCK, fetch_source, sha256
from native_reference import TEMPERATURE, choose, encode, pad_batch, softmax


def test_real_trained_adapter_contains_saved_scalar_head() -> None:
    source = fetch_source()
    adapter = source / "pretrained-scorer" / "adapter_model.safetensors"
    with adapter.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    tensor = header[LOCK["native_serving"]["trained_score_tensor"]]
    assert tensor["shape"] == [1, 640]
    assert tensor["dtype"] == "BF16"
    assert sha256(adapter) == LOCK["files"]["pretrained-scorer/adapter_model.safetensors"]


def test_locked_metrics_match_released_app_calibration() -> None:
    source = fetch_source(include_adapter=False)
    metrics = json.loads((source / "pretrained-scorer" / "metrics.json").read_text())
    assert metrics["temperature"] == TEMPERATURE
    assert metrics["max_len"] == LOCK["native_serving"]["max_length"] == 256
    adapter_config = json.loads((source / "pretrained-scorer" / "adapter_config.json").read_text())
    assert "score" in adapter_config["modules_to_save"]
    assert adapter_config["base_model_name_or_path"] == LOCK["base_repo"]


def test_native_rendering_preserves_question_tail_and_right_padding() -> None:
    class CharacterTokenizer:
        def __call__(self, text: str, add_special_tokens: bool):
            assert not add_special_tokens
            return {"input_ids": [ord(character) for character in text]}

    tokens = encode(CharacterTokenizer(), "state", "question", "option", max_length=28)
    assert tokens[-len("\n\nOption:\noption") :] == [ord(c) for c in "\n\nOption:\noption"]
    ids, mask = pad_batch([[1, 2, 3], [4]], 0, 4)
    assert ids == [[1, 2, 3, 0], [4, 0, 0, 0]]
    assert mask == [[1, 1, 1, 0], [1, 0, 0, 0]]


def test_calibrated_choice_and_invalid_temperature() -> None:
    result = choose(["first", "second"], [0.0, 2.35])
    assert result["selected_index"] == 1
    assert result["probabilities"] == pytest.approx([0.2689414214, 0.7310585786])
    with pytest.raises(ValueError):
        softmax([1.0], temperature=0.0)
