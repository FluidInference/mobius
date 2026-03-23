"""Measure prediction latency by running the model with random inputs."""

from __future__ import annotations

import time
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np

import CoreML
from Foundation import NSURL

from .compute_plan import COMPUTE_UNITS


def _make_cold_config(compute_units_val: int) -> Any:
    """Create MLModelConfiguration that bypasses compilation cache.

    Uses private API: setExperimentalMLProgramEncryptedCacheUsage_(0)
    to disable the E5 runtime's encrypted bundle cache, forcing a full
    recompilation from the MIL program.
    """
    config = CoreML.MLModelConfiguration.alloc().init()
    config.setComputeUnits_(compute_units_val)
    config.setExperimentalMLProgramEncryptedCacheUsage_(0)
    return config


def _fill_multiarray(ml_array: Any, shape: tuple[int, ...], dtype: Any) -> None:
    """Fill MLMultiArray with random data by iterating over indices."""
    total = reduce(lambda a, b: a * b, shape, 1)
    is_int = dtype in (np.int32, np.int64)
    for i in range(total):
        # Compute multi-dimensional index
        idx = []
        remaining = i
        for s in reversed(shape):
            idx.insert(0, remaining % s)
            remaining //= s
        val = int(np.random.randint(0, 100)) if is_int else float(np.random.randn())
        ml_array.setObject_atIndexedSubscript_(val, i)


def _make_input_provider(model_desc: Any) -> Any:
    """Create an MLDictionaryFeatureProvider with random data for all inputs."""
    input_desc = model_desc.inputDescriptionsByName()
    input_dict = {}

    for name in input_desc:
        feat = input_desc[name]
        feat_type = feat.type()

        if feat_type == CoreML.MLFeatureTypeMultiArray:
            constraint = feat.multiArrayConstraint()
            shape = tuple(int(d) for d in constraint.shape())
            ml_dtype = constraint.dataType()

            # Map to numpy dtype for random generation
            dtype_map = {
                CoreML.MLMultiArrayDataTypeFloat16: np.float16,
                CoreML.MLMultiArrayDataTypeFloat32: np.float32,
                CoreML.MLMultiArrayDataTypeFloat64: np.float64,
                CoreML.MLMultiArrayDataTypeInt32: np.int32,
            }
            np_dtype = dtype_map.get(ml_dtype, np.float32)

            ml_array, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
                list(shape), ml_dtype, None
            )
            if err:
                raise RuntimeError(f"Failed to create MLMultiArray for '{name}': {err}")

            _fill_multiarray(ml_array, shape, np_dtype)
            input_dict[name] = CoreML.MLFeatureValue.featureValueWithMultiArray_(ml_array)

        elif feat_type == CoreML.MLFeatureTypeState:
            # State inputs — create zeroed multi-array from constraint
            constraint = feat.multiArrayConstraint()
            if constraint:
                shape = tuple(int(d) for d in constraint.shape())
                ml_dtype = constraint.dataType()
                ml_array, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
                    list(shape), ml_dtype, None
                )
                if err:
                    raise RuntimeError(f"Failed to create state array for '{name}': {err}")
                input_dict[name] = CoreML.MLFeatureValue.featureValueWithMultiArray_(ml_array)
            else:
                pass
        else:
            pass

    provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
        input_dict, None
    )
    if err:
        raise RuntimeError(f"Failed to create input provider: {err}")
    return provider


def _compute_stats(times_ms: list[float]) -> dict:
    times_ms.sort()
    n = len(times_ms)
    mean = sum(times_ms) / n
    median = times_ms[n // 2] if n % 2 else (times_ms[n // 2 - 1] + times_ms[n // 2]) / 2
    variance = sum((t - mean) ** 2 for t in times_ms) / n
    return {
        "median_ms": round(median, 3),
        "mean_ms": round(mean, 3),
        "min_ms": round(times_ms[0], 3),
        "max_ms": round(times_ms[-1], 3),
        "std_ms": round(variance ** 0.5, 3),
    }


def measure_cold_compile(model_path: Path) -> float:
    """Measure cold compile time by bypassing the E5 compilation cache.

    Uses private API setExperimentalMLProgramEncryptedCacheUsage_(0).
    For a true first-launch measurement, restart ANECompilerService first:
        sudo killall ANECompilerService
    """
    url = NSURL.fileURLWithPath_(str(model_path))
    cold_config = _make_cold_config(CoreML.MLComputeUnitsAll)
    cold_start = time.perf_counter()
    model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, cold_config, None)
    cold_ms = (time.perf_counter() - cold_start) * 1000
    if error or model is None:
        return -1.0
    return cold_ms


def measure_latency(
    model_path: Path,
    compute_units: str,
    warmup: int = 5,
    iterations: int = 10,
) -> dict:
    """Load model via PyObjC and measure warm compile + prediction latency."""
    url = NSURL.fileURLWithPath_(str(model_path))
    config = CoreML.MLModelConfiguration.alloc().init()
    config.setComputeUnits_(COMPUTE_UNITS[compute_units])

    # Prime — first load populates E5 bundle cache for this compute unit config
    model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
    if error or model is None:
        return {"error": str(error) if error else "failed to load model"}
    del model

    # Warm compile — measure cached reload
    compile_start = time.perf_counter()
    model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
    compile_ms = (time.perf_counter() - compile_start) * 1000

    if error or model is None:
        return {"error": str(error) if error else "failed to load model"}

    model_desc = model.modelDescription()

    try:
        provider = _make_input_provider(model_desc)
    except Exception as e:
        return {"error": f"failed to create inputs: {e}"}

    # Warmup
    for _ in range(warmup):
        result, err = model.predictionFromFeatures_error_(provider, None)
        if err:
            return {
                "compile_ms": round(compile_ms, 3),
                "error": f"prediction failed: {err}",
            }

    # Timed runs
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predictionFromFeatures_error_(provider, None)
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    stats = _compute_stats(times_ms)
    return {
        "compile_ms": round(compile_ms, 3),
        **stats,
        "iterations": iterations,
    }
