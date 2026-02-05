#!/usr/bin/env python3
"""Measure CoreML RAM usage with minimal Python overhead."""
import sys
import os
import subprocess
import time

def get_memory_mb():
    """Get current process memory in MB using ps."""
    pid = os.getpid()
    result = subprocess.run(
        ['ps', '-p', str(pid), '-o', 'rss='],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024  # KB to MB

def measure_model(path: str, compute_units: str = "ALL"):
    """Load a CoreML model and measure RAM."""
    import coremltools as ct

    baseline = get_memory_mb()
    print(f"  Baseline RAM: {baseline:.1f} MB")

    # Load the model
    load_start = time.time()
    model = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, compute_units))
    load_time = time.time() - load_start

    after_load = get_memory_mb()
    load_ram = after_load - baseline

    print(f"  Load time: {load_time:.2f}s")
    print(f"  RAM after load: {load_ram:.1f} MB")
    print(f"  Peak RAM: {after_load:.1f} MB")

    return load_ram

def get_dir_size_mb(path: str) -> float:
    """Get directory size in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_ram_pure.py <model.mlpackage> [model2.mlpackage ...]")
        print("\nMeasures CoreML RAM usage with minimal Python overhead.")
        sys.exit(0)

    model_paths = sys.argv[1:]

    for path in model_paths:
        print("\n" + "=" * 60)
        print(f"Model: {os.path.basename(path)}")
        print("=" * 60)

        disk_size = get_dir_size_mb(path)
        print(f"Disk size: {disk_size:.1f} MB")

        for cu in ["CPU_AND_NEURAL_ENGINE", "CPU_AND_GPU", "ALL"]:
            print(f"\n[{cu}]")
            try:
                measure_model(path, cu)
            except Exception as e:
                print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Done.")
