"""Check the pinned NanoJev renderer and tokenizer before a large export."""

from transformers import AutoTokenizer

from assets import snapshot
from fixtures import fixture
from preprocessing import prepare_request


def test_candidate_set_and_eos_positions():
    root = snapshot(with_weights=False)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs, mask, example = prepare_request(root, tokenizer, fixture(), 128, 4)
    assert example["candidate_ids"] == ["left", "right", "shoot", "noop"]
    assert mask.tolist() == [[1, 1, 1, 1]]
    assert inputs["attention_mask"].sum(axis=1).min() > 0
    for row in range(4):
        eos = inputs["eos_map"][row, 0].nonzero()[0]
        assert len(eos) == 1
        assert inputs["input_ids"][row, eos[0]] == tokenizer.eos_token_id
