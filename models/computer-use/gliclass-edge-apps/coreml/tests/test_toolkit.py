from __future__ import annotations

import pytest
from decision_index.engines import Unsupported

from assets import dataset_revision, model_revision
from decision_index_engine import instruction_text, option_labels
from suite_mapping import labels_for
from verify import select_bucket


def test_select_bucket_uses_smallest_fitting_shape() -> None:
    lengths = [128, 256, 512]

    assert select_bucket(1, lengths) == 128
    assert select_bucket(128, lengths) == 128
    assert select_bucket(129, lengths) == 256
    assert select_bucket(256, lengths) == 256
    assert select_bucket(257, lengths) == 512


def test_select_bucket_truncates_to_largest_shape() -> None:
    assert select_bucket(513, [128, 256, 512]) == 512


def test_labels_preserve_benchmark_gold_indices() -> None:
    row = {
        "suite": "jev.ag_news",
        "options": [["world", "world news"], ["sports", "sports news"]],
    }

    assert labels_for(row) == [("world news", 0), ("sports news", 1)]
    assert labels_for(row, "full") == [("world: world news", 0), ("sports: sports news", 1)]


def test_binary_labels_are_explicit() -> None:
    row = {"suite": "app.email_spam", "options": []}

    assert labels_for(row) == [
        ("a legitimate personal or business email", 0),
        ("unsolicited spam or bulk marketing", 1),
    ]


def test_asset_revisions_are_pinned() -> None:
    assert model_revision("knowledgator/gliclass-edge-v3.0") == "df03993a2ed98e5e4a0d2dd7efbbd105abe874cf"
    assert dataset_revision("fancyzhx/ag_news") == "eb185aade064a813bc0b7f42de02595523103ca4"


def test_decision_index_labels_keep_keys_and_descriptions() -> None:
    keys, labels = option_labels(
        {
            "type": "choice",
            "criteria": {"A": "first answer", "B": None},
        }
    )

    assert keys == ["A", "B"]
    assert labels == ["A: first answer", "B"]


def test_decision_index_labels_reject_unsupported_types() -> None:
    with pytest.raises(Unsupported):
        option_labels({"type": "score"})


def test_decision_index_structured_instructions_are_rendered() -> None:
    rendered = instruction_text({"instructions": {"task": "rank", "candidate": {"name": "search"}}})

    assert rendered is not None
    assert "task" in rendered
    assert "rank" in rendered
    assert "candidate" in rendered
    assert "search" in rendered
