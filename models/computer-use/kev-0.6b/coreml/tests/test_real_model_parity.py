"""Check the exporter and standalone renderer against the pinned trained Kev checkpoint."""

import copy

import numpy as np
import torch

from assets import load_model
from export_model import KevExport
from preprocessing import Shape, prepare_inputs, prepare_runtime_inputs
from verify import fixtures


def test_real_checkpoint_choice_and_noul_logits():
    _, tokenizer, model = load_model()
    model.eval()
    wrapper = KevExport(model, 128, 32).eval()
    for request in fixtures():
        arrays, encoded = prepare_inputs(model, tokenizer, request, Shape(128, 32))
        unlabelled = copy.deepcopy(request)
        for question in unlabelled["questions"].values():
            question.pop("label", None)
            question.pop("src", None)
        runtime_arrays, runtime_encoded, _, _ = prepare_runtime_inputs(tokenizer, unlabelled, Shape(128, 32))
        assert runtime_encoded["ids"] == encoded["ids"]
        for name, values in arrays.items():
            np.testing.assert_array_equal(runtime_arrays[name], values)
        inputs = tuple(torch.from_numpy(value) for value in arrays.values())
        with torch.no_grad():
            native = model.forward(encoded)[0]
            exported = wrapper(*inputs)[0][0, : len(encoded["opt_idx"][0])]
        assert native.argmax().item() == exported.argmax().item()
        assert torch.max(torch.abs(native - exported)).item() < 1e-4
