"""Check real compressed artifacts and prevent mislabeled benchmark comparisons."""

import copy
import importlib.util

import coremltools as ct
import numpy as np
import pytest

from assets import ROOT
from verify import check_package

spec = importlib.util.spec_from_file_location("benchmark_synthetic", ROOT / "benchmark-synthetic.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.fixture(scope="module")
def artifacts():
    if not (ROOT / "build/int8-weights/conversion.json").exists():
        pytest.skip("Run quantize-int8.py to prepare the real optional artifact")
    return {
        name: check_package(ROOT / path)
        for name, path in (("baseline", "build"), ("int8-weights", "build/int8-weights"))
    }


def test_real_int8_is_smaller_and_preserves_tensor_contract(artifacts):
    source, baseline = artifacts["baseline"]
    candidate, quantized = artifacts["int8-weights"]
    benchmark.validate_variants({"baseline": baseline, "int8-weights": quantized}, "int8-weights")
    base_spec, candidate_spec = ct.utils.load_spec(str(source)), ct.utils.load_spec(str(candidate))
    assert base_spec.description.input == candidate_spec.description.input
    assert base_spec.description.output == candidate_spec.description.output
    assert candidate_spec.specificationVersion == base_spec.specificationVersion
    assert sum(p.stat().st_size for p in candidate.rglob("*") if p.is_file()) < sum(
        p.stat().st_size for p in source.rglob("*") if p.is_file()
    )
    assert quantized["quantization"]["activations"] == "float16"
    assert quantized["quantization"]["operation_counts"]["constexpr_affine_dequantize"] > 0


def test_dequantized_real_weights_are_finite_and_zero_channels_stay_zero(artifacts):
    # The upstream embedding has an all-zero padding row; the quantizer warns on
    # its zero scale. Verify the resulting tensors, rather than suppressing it.
    original = ct.models.MLModel(str(artifacts["baseline"][0]), skip_model_load=True)
    compressed = ct.models.MLModel(str(artifacts["int8-weights"][0]), skip_model_load=True)
    restored = ct.optimize.coreml.decompress_weights(compressed)
    before = ct.optimize.coreml.get_weights_metadata(original)
    after = {
        name.removesuffix("_quantized"): metadata
        for name, metadata in ct.optimize.coreml.get_weights_metadata(restored).items()
    }
    assert set(before) == set(after)
    zero_channels = 0
    for name, metadata in before.items():
        old, new = metadata.val, after[name].val
        assert old.shape == new.shape
        assert np.isfinite(new).all()
        if old.ndim < 2:
            continue
        zero = np.all(old == 0, axis=tuple(range(1, old.ndim)))
        zero_channels += int(zero.sum())
        assert np.all(new[zero] == 0)
    assert zero_channels > 0


def test_existing_ane_gather_comparison_is_still_valid(artifacts):
    _, ane = check_package(ROOT / "build/ane-gather")
    benchmark.validate_variants({"baseline": artifacts["baseline"][1], "ane-gather": ane}, "ane-gather")


@pytest.mark.parametrize("mismatch", ["source_weights", "limits", "model_revision", "variant_label"])
def test_rejects_mislabeled_or_unrelated_candidates(artifacts, mismatch):
    baseline = artifacts["baseline"][1]
    candidate = copy.deepcopy(artifacts["int8-weights"][1])
    if mismatch == "source_weights":
        candidate["quantization"]["source_package_files"] = {}
    elif mismatch == "limits":
        candidate["limits"]["max_options"] = 64
    elif mismatch == "model_revision":
        candidate["model_revision"] = "different-checkpoint"
    else:
        candidate["quantization"]["name"] = "ane-gather"
    with pytest.raises(ValueError):
        benchmark.validate_variants({"baseline": baseline, "int8-weights": candidate}, "int8-weights")
