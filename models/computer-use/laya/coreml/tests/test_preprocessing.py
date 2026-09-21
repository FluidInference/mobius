"""Shape and prompt-format checks that do not need the checkpoint weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assets import ROOT, checkpoint_dir, load_lock
from preprocessing import MAX_OPTIONS, Shape, encode, package_name, prepare_arrays, to_internal

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
