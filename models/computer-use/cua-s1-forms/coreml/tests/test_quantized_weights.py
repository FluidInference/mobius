"""Check real compressed artifacts and prevent mislabeled benchmark comparisons."""

import copy
import importlib.util
from collections import Counter

import coremltools as ct
import numpy as np
import pytest
from coremltools.proto import MIL_pb2

from assets import ROOT
from verify import check_package

spec = importlib.util.spec_from_file_location("benchmark_synthetic", ROOT / "benchmark-synthetic.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.fixture(scope="module", params=["int8-weights", "int4-weights"])
def artifacts(request):
    name = request.param
    baseline = "build" if name == "int8-weights" else "build/int4-source-fp16"
    if not (ROOT / "build" / name / "conversion.json").exists():
        pytest.skip("Run the quantization script to prepare the real optional artifact")
    return {name: check_package(ROOT / path) for name, path in (("baseline", baseline), ("candidate", f"build/{name}"))}


def test_real_compression_is_smaller_and_preserves_tensor_contract(artifacts):
    source, baseline = artifacts["baseline"]
    candidate, quantized = artifacts["candidate"]
    name = quantized["quantization"]["name"]
    benchmark.validate_variants({"baseline": baseline, name: quantized}, name)
    base_spec, candidate_spec = ct.utils.load_spec(str(source)), ct.utils.load_spec(str(candidate))
    assert base_spec.description.input == candidate_spec.description.input
    assert base_spec.description.output == candidate_spec.description.output
    assert candidate_spec.specificationVersion == base_spec.specificationVersion
    assert sum(p.stat().st_size for p in candidate.rglob("*") if p.is_file()) < sum(
        p.stat().st_size for p in source.rglob("*") if p.is_file()
    )
    assert quantized["quantization"]["activations"] == "float16"
    op_type = "constexpr_affine_dequantize" if name == "int8-weights" else "constexpr_blockwise_shift_scale"
    assert quantized["quantization"]["operation_counts"][op_type] > 0
    if name == "int4-weights":
        assert quantized["minimum_target"] == "iOS18/macOS15"
        assert candidate_spec.specificationVersion == 9
        packed = [
            op
            for function in candidate_spec.mlProgram.functions.values()
            for block in function.block_specializations.values()
            for op in block.operations
            if op.type == "constexpr_blockwise_shift_scale"
        ]
        assert len(packed) == quantized["quantization"]["packed_int4_tensors"] == 19
        data = [op.inputs["data"].arguments[0].value for op in packed]
        assert all(value.type.tensorType.dataType == MIL_pb2.INT4 for value in data)
        assert all(value.HasField("blobFileValue") for value in data)


def test_dequantized_real_weights_are_finite_and_zero_channels_stay_zero(artifacts):
    # The upstream embedding has an all-zero padding row; the quantizer warns on
    # its zero scale. Verify the resulting tensors, rather than suppressing it.
    original = ct.models.MLModel(str(artifacts["baseline"][0]), skip_model_load=True)
    compressed = ct.models.MLModel(str(artifacts["candidate"][0]), skip_model_load=True)
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


def test_real_fp16_control_keeps_original_weights(artifacts):
    original, original_manifest = check_package(ROOT / "build")
    old = ct.optimize.coreml.get_weights_metadata(ct.models.MLModel(str(original), skip_model_load=True))
    control = ct.optimize.coreml.get_weights_metadata(
        ct.models.MLModel(str(artifacts["baseline"][0]), skip_model_load=True)
    )

    # Target-specific lowering changes anonymous op names. Compare all real
    # weight tensors byte for byte, including shape, dtype and multiplicity.
    def tensors(metadata):
        return Counter((item.val.shape, item.val.dtype.str, item.val.tobytes()) for item in metadata.values())

    assert tensors(old) == tensors(control)
    if "target_upgrade" in artifacts["baseline"][1]:
        assert artifacts["baseline"][1]["target_upgrade"]["source_package_files"] == original_manifest["package_files"]
        specs = [ct.utils.load_spec(str(path)) for path in (original, artifacts["baseline"][0])]
        counts = [
            Counter(
                op.type
                for function in spec.mlProgram.functions.values()
                for block in function.block_specializations.values()
                for op in block.operations
                if op.type != "const"
            )
            for spec in specs
        ]
        assert counts[0] == counts[1]
        assert counts[1]["scaled_dot_product_attention"] == 0


@pytest.mark.parametrize("mismatch", ["source_weights", "limits", "model_revision", "variant_label", "dtype", "target"])
def test_rejects_mislabeled_or_unrelated_candidates(artifacts, mismatch):
    baseline = artifacts["baseline"][1]
    candidate = copy.deepcopy(artifacts["candidate"][1])
    name = candidate["quantization"]["name"]
    if mismatch == "source_weights":
        candidate["quantization"]["source_package_files"] = {}
    elif mismatch == "limits":
        candidate["limits"]["max_options"] = 64
    elif mismatch == "model_revision":
        candidate["model_revision"] = "different-checkpoint"
    elif mismatch == "dtype":
        candidate["quantization"]["settings"]["dtype"] = "uint8"
    elif mismatch == "target":
        candidate["minimum_target"] = "iOS16/macOS13"
    else:
        candidate["quantization"]["name"] = "ane-gather"
    with pytest.raises(ValueError):
        benchmark.validate_variants({"baseline": baseline, name: candidate}, name)
