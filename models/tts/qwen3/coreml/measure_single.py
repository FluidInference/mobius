#!/usr/bin/env python3
"""Single model measurement - run directly."""
import sys
import os
import warnings
import json

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_rss_mb():
    import subprocess
    pid = os.getpid()
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'rss='], capture_output=True, text=True)
    return int(result.stdout.strip()) / 1024 if result.stdout.strip() else 0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python measure_single.py <model.mlpackage> <CPU_AND_GPU|ALL>")
        sys.exit(1)

    path = sys.argv[1]
    compute_units = sys.argv[2]

    # Baseline before import
    baseline_rss = get_rss_mb()

    # Import coremltools (this adds overhead)
    import time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import coremltools as ct

    after_import_rss = get_rss_mb()

    # Load model
    load_start = time.time()
    cu = getattr(ct.ComputeUnit, compute_units)
    model = ct.models.MLModel(path, compute_units=cu)
    load_time = time.time() - load_start

    after_load_rss = get_rss_mb()

    # Output
    result = {
        "baseline_rss_mb": baseline_rss,
        "after_import_rss_mb": after_import_rss,
        "after_load_rss_mb": after_load_rss,
        "coremltools_overhead_mb": after_import_rss - baseline_rss,
        "model_ram_mb": after_load_rss - after_import_rss,
        "total_delta_mb": after_load_rss - baseline_rss,
        "load_time_s": load_time
    }

    print(json.dumps(result))
