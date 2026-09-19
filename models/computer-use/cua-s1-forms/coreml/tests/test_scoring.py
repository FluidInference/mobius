"""Ensure the upstream evaluator receives every saved real decision exactly once."""

import importlib.util
import json

import pytest

from assets import ROOT, load_demo


@pytest.fixture(scope="module")
def scoring():
    spec = importlib.util.spec_from_file_location("score_report", ROOT / "score-report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def decisions():
    report = json.loads((ROOT / "reports/verification.json").read_text())
    return report["backends"]["ALL"]["decisions"]


def test_rejects_missing_rows_even_when_upstream_treats_skip_as_abstention(scoring, decisions):
    rows = load_demo()
    with pytest.raises(ValueError, match="exactly once"):
        scoring.score_decisions(rows, decisions[:-1], "coreml")


def test_saved_real_decisions_pass_upstream_action_and_target_metrics(scoring, decisions):
    result = scoring.score_decisions(load_demo(), decisions, "coreml")
    assert result["examples"] == result["counts"]["correct"] == 196
    assert result["counts"]["acted"] == 46
    assert result["counts"]["abstained"] == 150
    assert result["wrong_action_rate"] == result["wrong_target_rate"] == result["unsafe_action_rate"] == 0
