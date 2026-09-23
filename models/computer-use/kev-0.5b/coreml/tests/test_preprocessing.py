import numpy as np

from preprocessing import Shape, prepare_inputs


class FakeTokenizer:
    pad_token_id = 7


class FakeModel:
    def encode(self, tokenizer, record, max_state, max_branch):
        del tokenizer, max_branch
        state = list(range(10))[:max_state]
        branch = [20, 21, 22, 23, 24]
        ids = state + branch
        start = len(state)
        return {
            "ids": ids,
            "seg": [0] * len(state) + [1] * len(branch),
            "decide_idx": [start + 4],
            "opt_idx": [[start + 1, start + 3]],
        }


def test_prepare_inputs_truncates_state_and_builds_readout_maps(monkeypatch):
    monkeypatch.setattr("preprocessing.materialize", lambda request: request)
    arrays, encoded = prepare_inputs(FakeModel(), FakeTokenizer(), {"questions": [{}]}, Shape(8, 4))

    assert len(encoded["ids"]) == 8
    assert arrays["input_ids"].dtype == np.int32
    assert arrays["input_ids"].tolist() == [[0, 1, 2, 20, 21, 22, 23, 24]]
    assert arrays["attention_mask"].sum() == 8
    assert arrays["decide_map"][0, 0, 7] == 1
    assert arrays["option_map"][0, 0, 4] == 1
    assert arrays["option_map"][0, 1, 6] == 1
    assert arrays["option_map"][0, 2:].sum() == 0
