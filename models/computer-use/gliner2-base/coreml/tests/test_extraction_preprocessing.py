"""Capacity checks for the real pinned GLiNER2 tokenizer and schema."""

from pathlib import Path

import pytest
from gliner2 import Schema

from preprocessing import load_processor, prepare_extraction

SOURCE = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1/snapshots/1a8bc24e00dc7300b9017c81d63e3dcdabb26596"
)


def test_extraction_keeps_all_queries_and_offsets():
    processor = load_processor(str(SOURCE))
    text = "Alice founded Acme in Toronto in 2020."
    arrays, batch = prepare_extraction(
        processor, text, Schema().entities(["person", "organization", "location"]), 128, 64, 8
    )
    assert int(arrays["query_mask"].sum()) == 3
    assert int(arrays["text_mask"].sum()) == len(batch.start_mappings[0])
    assert text[batch.start_mappings[0][0] : batch.end_mappings[0][0]] == "Alice"


@pytest.mark.parametrize("bucket", [(10, 64, 8), (128, 2, 8), (128, 64, 2)])
def test_extraction_rejects_capacity_exceeded(bucket):
    processor = load_processor(str(SOURCE))
    with pytest.raises(ValueError, match="bucket holds"):
        prepare_extraction(
            processor,
            "Alice founded Acme in Toronto in 2020.",
            Schema().entities(["person", "organization", "location"]),
            *bucket,
        )


def test_extraction_rejects_classification_choice_capacity():
    processor = load_processor(str(SOURCE))
    choices = [chr(ord("a") + index) for index in range(9)]
    with pytest.raises(ValueError, match="choices; bucket holds 8"):
        prepare_extraction(processor, "A short report.", Schema().classification("topic", choices), 128, 64, 8)
