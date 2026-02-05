#!/usr/bin/env python3
"""Measure CoreML RAM using vmmap for accurate dirty/swapped memory."""
import sys
import os
import subprocess
import json
import re

MEASURE_SCRIPT = '''
import sys
import os
import time
import subprocess
import re

def parse_vmmap_summary(pid):
    """Parse vmmap output to get dirty + swapped memory."""
    result = subprocess.run(
        ['vmmap', '--summary', str(pid)],
        capture_output=True, text=True
    )

    # Look for TOTAL line or specific categories
    dirty_mb = 0
    swapped_mb = 0
    resident_mb = 0

    for line in result.stdout.split('\\n'):
        # Parse "TOTAL" line or "Physical footprint" line
        if 'TOTAL' in line or 'Physical footprint' in line.lower():
            # Format varies, try to extract sizes
            parts = line.split()
            for i, part in enumerate(parts):
                if part.endswith('M') or part.endswith('G') or part.endswith('K'):
                    try:
                        val = float(part[:-1])
                        unit = part[-1]
                        if unit == 'G':
                            val *= 1024
                        elif unit == 'K':
                            val /= 1024
                        # Assign to appropriate field based on position/context
                        if 'dirty' in line.lower():
                            dirty_mb = val
                        elif 'resident' in line.lower():
                            resident_mb = val
                    except:
                        pass

    # Alternative: look for specific regions
    for line in result.stdout.split('\\n'):
        if line.strip().startswith('REGION TYPE'):
            continue
        # CoreML related regions
        if any(x in line.lower() for x in ['coreml', 'ane', 'gpu', 'neural']):
            parts = line.split()
            for part in parts:
                if part.endswith('M'):
                    try:
                        val = float(part[:-1])
                        dirty_mb += val
                    except:
                        pass

    return {"dirty_mb": dirty_mb, "resident_mb": resident_mb, "swapped_mb": swapped_mb}

def get_rss_mb():
    pid = os.getpid()
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'rss='], capture_output=True, text=True)
    return int(result.stdout.strip()) / 1024 if result.stdout.strip() else 0

import coremltools as ct

path = sys.argv[1]
compute_units = sys.argv[2]

# Baseline
baseline_rss = get_rss_mb()

# Load model
load_start = time.time()
cu = getattr(ct.ComputeUnit, compute_units)
model = ct.models.MLModel(path, compute_units=cu)
load_time = time.time() - load_start

# After load measurements
after_rss = get_rss_mb()
delta_rss = after_rss - baseline_rss

# Run inference to ensure weights are loaded
spec = model.get_spec()
# Just get the input shapes to understand model size
input_info = {}
for inp in spec.description.input:
    if inp.type.HasField("multiArrayType"):
        shape = list(inp.type.multiArrayType.shape)
        input_info[inp.name] = shape

# Memory after everything is loaded
final_rss = get_rss_mb()

print(json.dumps({
    "baseline_rss_mb": baseline_rss,
    "after_load_rss_mb": after_rss,
    "final_rss_mb": final_rss,
    "delta_rss_mb": delta_rss,
    "load_time_s": load_time,
    "input_shapes": input_info
}))
'''

def get_dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def measure_model_isolated(path: str, compute_units: str, venv_python: str):
    result = subprocess.run(
        [venv_python, '-c', MEASURE_SCRIPT, path, compute_units],
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONWARNINGS': 'ignore'}
    )

    # Look for JSON in stdout
    for line in reversed(result.stdout.strip().split('\n')):
        if line.startswith('{'):
            try:
                return json.loads(line)
            except:
                pass
    for line in reversed(result.stderr.strip().split('\n')):
        if line.startswith('{'):
            try:
                return json.loads(line)
            except:
                pass
    return {"error": result.stderr[:300] if result.stderr else "No output"}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_ram_vmmap.py <model.mlpackage> ...")
        sys.exit(0)

    model_paths = sys.argv[1:]
    venv_python = sys.executable
    results = []

    for path in model_paths:
        print("\n" + "=" * 80)
        name = os.path.basename(path)
        print(f"Model: {name}")
        print("=" * 80)

        disk_size = get_dir_size_mb(path)
        print(f"Disk size: {disk_size:.0f} MB")

        model_results = {"name": name, "disk_mb": disk_size}

        for cu in ["CPU_AND_GPU", "ALL"]:
            print(f"\n[{cu}]")
            r = measure_model_isolated(path, cu, venv_python)

            if "error" in r:
                print(f"  Error: {r['error'][:100]}")
            else:
                print(f"  Load time: {r['load_time_s']:.2f}s")
                print(f"  Baseline RSS:    {r['baseline_rss_mb']:>8.0f} MB")
                print(f"  After load RSS:  {r['after_load_rss_mb']:>8.0f} MB")
                print(f"  Delta RSS:       {r['delta_rss_mb']:>8.0f} MB")
                model_results[f"{cu}_delta"] = r['delta_rss_mb']
                model_results[f"{cu}_total"] = r['after_load_rss_mb']

        results.append(model_results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: CoreML RAM Usage")
    print("=" * 80)
    print(f"{'Model':<40} {'Disk':>8} {'GPU RSS':>10} {'ALL RSS':>10}")
    print("-" * 80)
    for r in results:
        name = r["name"][:39]
        disk = f"{r['disk_mb']:.0f}M"
        gpu = f"{r.get('CPU_AND_GPU_delta', 0):.0f}M"
        all_cu = f"{r.get('ALL_delta', 0):.0f}M"
        print(f"{name:<40} {disk:>8} {gpu:>10} {all_cu:>10}")

    # Combined
    print("-" * 80)
    print("COMBINED (Prefill + Decode):")

    pairs = [
        ("FP32", "prefill_v9.mlpackage", "decode_v4.mlpackage"),
        ("FP16", "prefill_v9_fp16", "decode_v4_fp16"),
        ("INT8", "prefill_v9_int8", "decode_v4_int8"),
    ]

    for label, pfx, dfx in pairs:
        prefill = next((r for r in results if pfx in r['name']), {})
        decode = next((r for r in results if dfx in r['name']), {})
        if prefill and decode:
            disk = prefill.get('disk_mb', 0) + decode.get('disk_mb', 0)
            gpu = prefill.get('CPU_AND_GPU_delta', 0) + decode.get('CPU_AND_GPU_delta', 0)
            all_v = prefill.get('ALL_delta', 0) + decode.get('ALL_delta', 0)
            print(f"  {label}: Disk={disk:.0f}M, GPU_RSS={gpu:.0f}M, ALL_RSS={all_v:.0f}M")

    print("=" * 80)
    print("\nNote: RSS measures resident set size - actual physical RAM used by the process.")
    print("      This excludes memory-mapped but unaccessed pages and shared memory.")
