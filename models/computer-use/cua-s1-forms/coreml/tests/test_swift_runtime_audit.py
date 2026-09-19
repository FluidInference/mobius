"""Check runtime audit failures using the recorded real normalization output."""

import importlib.util
import json

import numpy as np
import pytest

from assets import ROOT

spec = importlib.util.spec_from_file_location("swift_runtime", ROOT / "validate-swift-runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture
def recorded(tmp_path):
    fixture = json.loads((ROOT / "tests/fixtures/synthetic-normalization-output.json").read_text())
    # One-record unit-test manifest, sourced from actual row 19270. These values
    # test auditing, not a substitute model or an additional accuracy benchmark.
    logits = np.asarray(fixture["output"]["logits"][0][: fixture["count"]], dtype=np.float64)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    record = {
        "row": 0,
        "model": "fp16",
        "label": fixture["label"],
        "selected": fixture["label"],
        "rawSelected": fixture["label"],
        "logits": logits.tolist(),
        "probabilities": probabilities.astype(np.float32).tolist(),
        "rawProbabilities": fixture["output"]["probabilities"][0][: fixture["count"]],
    }
    rows = [{"label": fixture["label"], "options": list(range(fixture["count"]))}]
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(record) + "\n")
    return path, rows, record


def test_keeps_raw_failure_visible_while_validating_stable_runtime(recorded):
    path, rows, record = recorded
    result = runtime.audit_trace(path, rows, {"fp16": [record["label"]]})["fp16"]
    assert result["passed"]
    assert result["correct"] == result["rows"] == 1
    assert result["raw_sum_failures"] == [0]
    assert result["runtime_sum_failures"] == result["changed_choices"] == []
    assert result["max_probability_change"] > 0.001


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong_label", "wrong_choice", "nonfinite"])
def test_rejects_incomplete_or_inconsistent_runtime_records(recorded, case):
    path, rows, record = recorded
    if case == "missing":
        path.write_text("")
    elif case == "duplicate":
        path.write_text((json.dumps(record) + "\n") * 2)
    else:
        if case == "wrong_label":
            record["label"] = 0
        elif case == "wrong_choice":
            record["selected"] = 0
        else:
            record["probabilities"][0] = float("nan")
        path.write_text(json.dumps(record) + "\n")
    with pytest.raises((ValueError, IndexError)):
        runtime.audit_trace(path, rows, {"fp16": [2]})


def test_a_changed_raw_choice_fails_the_runtime_gate(recorded):
    path, rows, _ = recorded
    result = runtime.audit_trace(path, rows, {"fp16": [0]})["fp16"]
    assert not result["passed"]
    assert result["raw_choice_changes"] == [0]
