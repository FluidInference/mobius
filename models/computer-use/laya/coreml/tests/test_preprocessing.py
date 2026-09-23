"""Shape and prompt-format checks that do not need the checkpoint weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assets import ROOT, checkpoint_dir, load_lock
from calibration import calibrated_probabilities, temperature_for
from preprocessing import (
    MAX_OPTIONS,
    Shape,
    encode,
    package_name,
    prepare_arrays,
    to_internal,
)

TOKENIZER = checkpoint_dir("multilingual") / "tokenizer" / "tokenizer.json"


@pytest.fixture(scope="module")
def tok():
    if not TOKENIZER.exists():
        pytest.skip("run assets.py first")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER.parent))


def test_lock_pins_every_required_file():
    lock = load_lock()
    files = lock["variants"]["multilingual"]["files"]
    assert {"model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"} <= set(
        files
    )
    assert all(len(digest) == 64 for digest in files.values())


def test_package_name():
    assert package_name("multilingual", 128, 32) == "laya_multilingual_fp16_L128_options32"
    assert package_name("english", 128, 32) == "laya_english_fp16_L128_options32"


def test_english_checkpoint_lock_and_calibration():
    lock = load_lock()
    assert lock["variants"]["english"]["files"]["model.safetensors"] == (
        "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
    )
    config_path = checkpoint_dir("english") / "rl_agent_config.json"
    if not config_path.exists():
        pytest.skip("run assets.py --variant english first")
    config = json.loads(config_path.read_text())
    assert temperature_for(config, 0, 2) == pytest.approx(1.9063563346862793)
    assert temperature_for(config, 0, 12) == pytest.approx(0.10058280825614929)
    assert calibrated_probabilities(np.array([0.0, 1.0]), 2.0).sum() == pytest.approx(1.0)
    from laya.common import temp_bucket

    for question_type in range(3):
        for option_count in (2, 3, 5, 6, 10, 11, 20, 32):
            upstream = config["temperature_by_options"].get(
                temp_bucket(question_type, option_count), config["temperature"][question_type]
            )
            assert temperature_for(config, question_type, option_count) == pytest.approx(float(upstream))


def test_to_internal_matches_agent_shapes():
    assert to_internal({"type": "choice", "instructions": "q", "criteria": ["a", "b"]}) == {
        "t": "choice",
        "ins": "q",
        "crit": {"a": None, "b": None},
    }
    assert to_internal({"type": "noul", "instructions": {"k": 1}})["ins"] == '{"k": 1}'


def test_prepare_arrays_layout():
    ids = [2, 10, 11, 1, 4, 12, 4, 13, 1, 20, 21, 1]
    arrays = prepare_arrays(ids, [4, 6], 2, Shape(16), pad_id=0)
    assert arrays["input_ids"].shape == (1, 16) and arrays["input_ids"].dtype == np.int32
    assert arrays["input_ids"][0, : len(ids)].tolist() == ids
    assert arrays["attention_mask"][0].tolist() == [1] * len(ids) + [0] * (16 - len(ids))
    assert arrays["marker_map"].shape == (1, MAX_OPTIONS, 16)
    assert arrays["marker_map"][0, 0, 4] == 1.0 and arrays["marker_map"][0, 1, 6] == 1.0
    assert arrays["marker_map"].sum() == 2.0
    assert arrays["question_type"][0].tolist() == [0.0, 0.0, 1.0]


def test_prepare_arrays_rejects_overflow():
    with pytest.raises(ValueError):
        prepare_arrays(list(range(20)), [1], 0, Shape(16), pad_id=0)


def test_encode_matches_reference_fixture(tok):
    fixture = ROOT / "fixtures" / "sequence-cases.json"
    if not fixture.exists():
        pytest.skip("run make_fixtures.py first")
    cases = json.loads(fixture.read_text())
    head_max_len = json.loads((checkpoint_dir("multilingual") / "rl_agent_config.json").read_text())["head_max_len"]
    for case in cases[:8]:
        question = {"type": case["type"], "instructions": case["instructions"]}
        if case["type"] == "choice":
            question["criteria"] = {label: description for label, description in case["options"]}
        elif case["type"] == "score":
            question["criteria"] = [label for label, _ in case["options"]]
        ids, markers, qtype = encode(tok, case["state"], question, Shape(case["length"]), head_max_len)
        assert ids == case["ids"] and markers == case["markers"] and qtype == case["qtype"]


def test_reports_exist_and_pass():
    reports = sorted(Path(ROOT / "reports").glob("verification-multilingual-L*.json"))
    assert reports, "run verify.py"
    published = 0
    for report in reports:
        data = json.loads(report.read_text())
        if data.get("precision", "fp16") not in ("fp16", "e8"):
            continue  # unpublished compression experiments keep their reports as evidence only
        published += 1
        assert data["passed"], report.name
        for run in data["runs"].values():
            assert run["argmax_agreements"] == run["questions"]
    assert published >= 4


def test_english_release_route_keeps_ane_failure_visible():
    paths = [ROOT / "reports" / f"verification-english-L{length}.json" for length in (128, 512)]
    if not all(path.exists() for path in paths):
        pytest.skip("run verify.py --variant english first")
    for path in paths:
        report = json.loads(path.read_text())
        assert report["validated_release_units"] == "ALL"
        assert report["runs"]["ALL"]["passed"]
        assert not report["runs"]["CPU_AND_NE"]["passed"]
        assert report["runs"]["CPU_AND_NE"]["max_calibrated_probability_error"] > 0.02
