"""Native protocol checks using the released, hash-locked calibrator."""

import json
from pathlib import Path

import pytest

from native_reference import ABSTAIN_ID, build_request, temperature


def test_native_choice_rendering_and_abstention():
    request = build_request(
        "I lost my wallet.",
        {
            "type": "choice",
            "question": "What happened?",
            "options": [
                {"id": "lost", "description": "a lost card"},
                {"id": "pin", "description": "a PIN reset"},
            ],
        },
    )
    assert request.ids == ("lost", "pin", ABSTAIN_ID)
    assert request.text == (
        "<<LABEL>>It is a lost card<<LABEL>>It is a PIN reset"
        "<<LABEL>>insufficient evidence<<SEP>>Question: What happened?\n\nContext:\nI lost my wallet."
    )


def test_native_capacity_rejects_dropped_candidates():
    options = [{"id": str(i), "description": str(i)} for i in range(25)]
    with pytest.raises(ValueError, match="24"):
        build_request("context", {"type": "choice", "question": "question", "options": options})


def test_reserved_abstention_id_cannot_be_a_substantive_option():
    with pytest.raises(ValueError, match="unique"):
        build_request(
            "context",
            {"type": "choice", "question": "question", "options": [
                {"id": ABSTAIN_ID, "description": "a candidate"}
            ]},
        )


def test_released_calibration_uses_candidate_count():
    path = Path(__file__).resolve().parents[1] / "build" / "source" / "calibrator.json"
    if not path.exists():
        pytest.skip("run assets.py first to fetch the released calibrator")
    released = json.loads(path.read_text())
    assert temperature(released, 3) == pytest.approx(5.0069)
    assert temperature(released, 8) == pytest.approx(2.8039)
