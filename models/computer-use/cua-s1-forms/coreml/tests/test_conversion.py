"""Regression checks use the pinned real checkpoint and existing demo records."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from assets import ROOT, load_demo, load_reference, verify_assets
from export_model import ExportScorer
from preprocessing import InputLimits, prepare_inputs
from verify import check_package, validate_prediction


@pytest.fixture(scope="module")
def demo():
    return load_demo()


@pytest.fixture(scope="module")
def upstream():
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    return load_reference()


def test_real_checkpoint_and_dataset(upstream, demo):
    model, _, config = upstream
    assert sum(parameter.numel() for parameter in model.parameters()) == 706048
    assert len(demo) == verify_assets()["evaluation_rows"] == 196
    assert config["encoder"] == "tinyx"


@pytest.mark.parametrize("row_index", [0, 63, 127, 195])
def test_host_encoding_matches_unmodified_upstream(upstream, demo, row_index):
    from cua_s1.model import validate_example

    _, collator, _ = upstream
    row = demo[row_index]
    limits = InputLimits()
    actual = prepare_inputs(row["context"], row["options"], limits)
    expected = collator([validate_example(row)])
    context_length = expected["context_ids"].shape[1]
    count, option_length = expected["option_ids"].shape[1:]
    np.testing.assert_array_equal(actual["context_ids"][:, :context_length], expected["context_ids"])
    np.testing.assert_array_equal(actual["option_ids"][:, :count, :option_length], expected["option_ids"])
    assert not actual["context_ids"][:, context_length:].any()
    assert not actual["option_ids"][:, count:].any()
    assert not actual["option_ids"][:, :, option_length:].any()
    np.testing.assert_array_equal(actual["option_mask"][0, :count], 1)
    assert not actual["option_mask"][0, count:].any()
    assert all(array.dtype == np.int32 for array in actual.values())


def test_utf8_byte_boundary_truncation_matches_upstream(upstream, demo):
    from cua_s1.model import validate_example

    _, collator, _ = upstream
    row = dict(demo[0])
    # Encoding regression only: extend real input text across both limits and
    # deliberately place a multibyte character across the byte boundary.
    row["context"] = (row["context"] * 3)[:223] + "🙂"
    row["options"] = [(option * 8)[:95] + "🙂" for option in row["options"]]
    actual = prepare_inputs(row["context"], row["options"], InputLimits())
    expected = collator([validate_example(row)])
    np.testing.assert_array_equal(actual["context_ids"], expected["context_ids"])
    np.testing.assert_array_equal(actual["option_ids"][:, : len(row["options"])], expected["option_ids"])


def test_option_overflow_is_never_silently_truncated(demo):
    row = demo[0]
    options = (row["options"] * 2)[:33]
    with pytest.raises(ValueError, match="re-export"):
        prepare_inputs(row["context"], options, InputLimits())


@pytest.mark.parametrize("context", ["", None])
def test_empty_context_is_rejected(demo, context):
    with pytest.raises(ValueError, match="nonempty"):
        prepare_inputs(context, demo[0]["options"], InputLimits())


@pytest.mark.parametrize("capacity", [32, 40])
@pytest.mark.parametrize("optimization", ["baseline", "ane-gather"])
def test_adapter_padding_preserves_real_predictions(upstream, demo, capacity, optimization):
    from cua_s1.model import validate_example

    model, collator, _ = upstream
    row = demo[63]
    arrays = prepare_inputs(row["context"], row["options"], InputLimits(max_options=capacity))
    with torch.no_grad():
        expected = model(collator([validate_example(row)]))[0].softmax(-1).numpy()
        logits, probabilities = ExportScorer(model, optimization)(
            *(torch.from_numpy(value) for value in arrays.values())
        )
    _, actual = validate_prediction(
        {"logits": logits.numpy(), "probabilities": probabilities.numpy()}, len(row["options"]), capacity
    )
    np.testing.assert_allclose(actual, expected, atol=0.00001, rtol=0.00001)


def test_shared_float_indices_preserve_every_byte_embedding(upstream):
    model, _, _ = upstream
    ids = torch.arange(257, dtype=torch.int32).reshape(257, 1)
    with torch.no_grad():
        actual = model._embed(ids.to(torch.float16).to(torch.int32))
        expected = model._embed(ids)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.fixture(scope="module")
def coreml_model():
    import coremltools as ct

    package, manifest = check_package(Path(os.environ.get("CUA_COREML_BUILD_DIR", ROOT / "build")))
    return ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL), InputLimits(**manifest["limits"])


@pytest.mark.parametrize("bounds", [(0, 128), (128, 257)])
def test_raw_byte_id_boundaries_preserve_model_scores(upstream, demo, coreml_model, bounds):
    model, _, _ = upstream
    coreml, limits = coreml_model
    row = demo[0]
    arrays = prepare_inputs(row["context"], row["options"], limits)
    # Tensor-domain regression: exercise every byte ID, including padding and
    # 256, with the real trained network. These are not benchmark examples.
    ids = np.arange(*bounds, dtype=np.int32)
    arrays["context_ids"][:] = 0
    arrays["context_ids"][0, :len(ids)] = ids
    with torch.no_grad():
        _, expected = ExportScorer(model)(*(torch.from_numpy(value) for value in arrays.values()))
    output = coreml.predict(arrays)
    _, probabilities = validate_prediction(output, len(row["options"]), limits.max_options)
    np.testing.assert_allclose(probabilities, expected.numpy()[0, :len(row["options"])], atol=0.005, rtol=0)


@pytest.mark.parametrize("mode", ["two_options", "full_capacity", "long_text", "reversed"])
def test_coreml_variable_content_against_upstream(upstream, demo, coreml_model, mode):
    from cua_s1.model import validate_example

    model, collator, _ = upstream
    coreml, limits = coreml_model
    row = dict(demo[0])
    row["label"] = 0  # The derived fixtures test numeric parity, not benchmark accuracy.
    if mode == "two_options":
        row["options"] = row["options"][:2]
    elif mode == "full_capacity":
        row["options"] = (row["options"] * 2)[: limits.max_options]
    elif mode == "long_text":
        row["context"] *= 3
        row["options"] = [option * 3 for option in row["options"]]
    elif mode == "reversed":
        row["options"] = list(reversed(row["options"]))
    with torch.no_grad():
        expected = model(collator([validate_example(row)]))[0].softmax(-1).numpy()
    output = coreml.predict(prepare_inputs(row["context"], row["options"], limits))
    _, probabilities = validate_prediction(output, len(row["options"]), limits.max_options)
    np.testing.assert_allclose(probabilities, expected, atol=0.005, rtol=0)
    assert probabilities.argmax() == expected.argmax()
