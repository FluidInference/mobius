"""Validate full released test coverage and paired/calibration metric edge cases."""

import json
import math

import pytest

from assets import ROOT
from preprocessing import InputLimits
from synthetic_test import MANIFEST, calibration, inspect_rows, load_test, paired_outcomes


@pytest.fixture(scope="module")
def published_test():
    manifest = json.loads(MANIFEST.read_text())
    if not (ROOT / manifest["path"]).exists():
        pytest.skip("Run benchmark-synthetic.py to download the separately pinned published test split")
    return load_test()


def test_entire_published_split_fits_without_filtering_or_truncation(published_test):
    rows, manifest = published_test
    census = inspect_rows(rows, InputLimits())
    assert census["rows"] == manifest["rows"] == 24370
    assert census["maxima"] == {"context_bytes": 217, "option_bytes": 92, "max_options": 32}
    assert census["excluded_rows"] == census["truncated_contexts"] == census["truncated_options"] == 0
    assert census["per_action"] == {"check": 816, "click": 1040, "fill": 9802, "skip": 12712}
    assert sum(census["option_count_histogram"].values()) == len(rows)


@pytest.mark.parametrize(
    "limits", [InputLimits(max_options=31), InputLimits(context_bytes=216), InputLimits(option_bytes=91)]
)
def test_exceeding_any_interface_limit_fails_instead_of_filtering(published_test, limits):
    with pytest.raises(ValueError, match="do not filter or truncate"):
        inspect_rows(published_test[0], limits)


def test_annotation_and_label_mismatch_is_rejected(published_test):
    row = published_test[0][0]
    changed = {**row, "meta": {**row["meta"], "action": "skip"}}
    with pytest.raises(ValueError, match="annotation"):
        inspect_rows([changed], InputLimits())
    with pytest.raises(ValueError, match="label"):
        inspect_rows([{**row, "label": True}], InputLimits())


def test_calibration_includes_confidence_one_and_clips_zero_gold_probability():
    result = calibration([0.5, 1.0], [0.5, 0.1], [True, False])
    assert result["ece"] == pytest.approx(0.75)
    assert result["nll"] == pytest.approx(-(math.log(0.5) + math.log(0.1)) / 2)
    assert sum(bin_["count"] for bin_ in result["bins"]) == 2
    assert math.isfinite(calibration([1.0], [0.0], [False])["nll"])
    with pytest.raises(ValueError):
        calibration([0.9], [0.9, 0.8], [True])
    with pytest.raises(ValueError):
        calibration([float("nan")], [1.0], [True])


def test_paired_outcomes_do_not_hide_harmful_flips_behind_equal_accuracy(published_test):
    labels = [row["label"] for row in published_test[0][:4]]
    wrong = [(label + 1) % len(row["options"]) for label, row in zip(labels, published_test[0][:4])]
    result = paired_outcomes(
        labels, [labels[0], wrong[1], wrong[2], labels[3]], [wrong[0], labels[1], wrong[2], labels[3]]
    )
    assert result["reference_correct_candidate_wrong"] == 1
    assert result["reference_wrong_candidate_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["argmax_agreement"] == 2
    assert result["disagreement_rows"] == [0, 1]
    with pytest.raises(ValueError):
        paired_outcomes(labels, labels[:-1], labels)


def test_recorded_real_normalization_failure_is_retained_without_renormalizing():
    import importlib.util
    import json

    import numpy as np

    from assets import ROOT
    from verify import validate_prediction

    spec = importlib.util.spec_from_file_location("benchmark_synthetic", ROOT / "benchmark-synthetic.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    # Recorded real output makes this metric regression independent of the test machine's backend.
    fixture = json.loads((ROOT / "tests/fixtures/synthetic-normalization-output.json").read_text())
    output = {key: np.asarray(value, dtype=np.float32) for key, value in fixture["output"].items()}
    with pytest.raises(ValueError, match="do not sum to one"):
        validate_prediction(output, fixture["count"], fixture["capacity"])
    probabilities, issues = benchmark.prediction_for_comparison(output, fixture["count"], fixture["capacity"])
    assert issues == ["Live option probabilities do not sum to one"]
    np.testing.assert_array_equal(probabilities, output["probabilities"][0, : fixture["count"]])
    assert probabilities.argmax() == fixture["label"]
    assert probabilities.sum() == pytest.approx(0.99893665, abs=0.00001)
