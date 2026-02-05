#!/usr/bin/env python3
"""Measure CoreML RAM including inference to touch all weights."""
import sys
import os
import warnings
import json
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_rss_mb():
    import subprocess
    pid = os.getpid()
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'rss='], capture_output=True, text=True)
    return int(result.stdout.strip()) / 1024 if result.stdout.strip() else 0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python measure_with_inference.py <model.mlpackage> <CPU_AND_GPU|ALL>")
        sys.exit(1)

    path = sys.argv[1]
    compute_units = sys.argv[2]

    import time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import coremltools as ct

    baseline_rss = get_rss_mb()

    # Load model
    cu = getattr(ct.ComputeUnit, compute_units)
    model = ct.models.MLModel(path, compute_units=cu)

    after_load_rss = get_rss_mb()

    # Create dummy inputs and run inference
    spec = model.get_spec()
    inputs = {}
    for inp in spec.description.input:
        if inp.type.HasField("multiArrayType"):
            shape = [d if d > 0 else 1 for d in inp.type.multiArrayType.shape]
            dtype_map = {65568: np.float16, 65552: np.float32, 131104: np.int32}
            np_dtype = dtype_map.get(inp.type.multiArrayType.dataType, np.float32)
            inputs[inp.name] = np.zeros(shape, dtype=np_dtype)

    # Run inference to touch all weights
    try:
        _ = model.predict(inputs)
        after_infer_rss = get_rss_mb()
        inference_ran = True
    except Exception as e:
        after_infer_rss = after_load_rss
        inference_ran = False

    result = {
        "baseline_rss_mb": baseline_rss,
        "after_load_rss_mb": after_load_rss,
        "after_inference_rss_mb": after_infer_rss,
        "model_load_ram_mb": after_load_rss - baseline_rss,
        "peak_ram_mb": after_infer_rss - baseline_rss,
        "inference_ran": inference_ran
    }
    print(json.dumps(result))
