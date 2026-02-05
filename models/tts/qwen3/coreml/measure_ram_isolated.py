#!/usr/bin/env python3
"""Measure CoreML RAM usage with process isolation for clean baselines."""
import sys
import os
import subprocess
import json

MEASURE_SCRIPT = '''
import sys
import os
import time
import coremltools as ct

def get_memory_mb():
    """Get current process memory in MB."""
    pid = os.getpid()
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'rss='], capture_output=True, text=True)
    return int(result.stdout.strip()) / 1024

import subprocess

path = sys.argv[1]
compute_units = sys.argv[2]

# Measure baseline before any CoreML loading
baseline = get_memory_mb()

# Load model
load_start = time.time()
cu = getattr(ct.ComputeUnit, compute_units)
model = ct.models.MLModel(path, compute_units=cu)
load_time = time.time() - load_start

# Measure after load
after_load = get_memory_mb()
delta = after_load - baseline

# Output JSON result
import json
print(json.dumps({
    "baseline_mb": baseline,
    "after_load_mb": after_load,
    "delta_mb": delta,
    "load_time_s": load_time
}))
'''

def get_dir_size_mb(path: str) -> float:
    """Get directory size in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def measure_model_isolated(path: str, compute_units: str, venv_python: str):
    """Run measurement in isolated subprocess."""
    result = subprocess.run(
        [venv_python, '-c', MEASURE_SCRIPT, path, compute_units],
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONWARNINGS': 'ignore'}
    )

    if result.returncode != 0:
        return {"error": result.stderr.strip() or "Unknown error"}

    # Parse the last line as JSON (skip warnings)
    for line in reversed(result.stdout.strip().split('\n')):
        if line.startswith('{'):
            return json.loads(line)
    return {"error": "No JSON output"}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_ram_isolated.py <model.mlpackage> [model2.mlpackage ...]")
        print("\nMeasures CoreML RAM in isolated subprocesses for clean baselines.")
        sys.exit(0)

    model_paths = sys.argv[1:]
    venv_python = sys.executable

    # Summary table
    results = []

    for path in model_paths:
        print("\n" + "=" * 70)
        print(f"Model: {os.path.basename(path)}")
        print("=" * 70)

        disk_size = get_dir_size_mb(path)
        print(f"Disk size: {disk_size:.1f} MB")

        model_results = {"name": os.path.basename(path), "disk_mb": disk_size}

        for cu in ["CPU_AND_GPU", "ALL"]:
            print(f"\n[{cu}]")
            result = measure_model_isolated(path, cu, venv_python)

            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Baseline: {result['baseline_mb']:.1f} MB")
                print(f"  After load: {result['after_load_mb']:.1f} MB")
                print(f"  Delta (CoreML RAM): {result['delta_mb']:.1f} MB")
                print(f"  Load time: {result['load_time_s']:.2f}s")
                model_results[cu] = result['delta_mb']

        results.append(model_results)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: CoreML RAM Usage (Delta from Python baseline)")
    print("=" * 70)
    print(f"{'Model':<45} {'Disk':>10} {'GPU':>12} {'ALL':>12}")
    print("-" * 70)
    for r in results:
        name = r["name"][:44]
        disk = f"{r['disk_mb']:.0f} MB"
        gpu = f"{r.get('CPU_AND_GPU', 'N/A'):.0f} MB" if isinstance(r.get('CPU_AND_GPU'), (int, float)) else "N/A"
        all_cu = f"{r.get('ALL', 'N/A'):.0f} MB" if isinstance(r.get('ALL'), (int, float)) else "N/A"
        print(f"{name:<45} {disk:>10} {gpu:>12} {all_cu:>12}")

    print("=" * 70)
