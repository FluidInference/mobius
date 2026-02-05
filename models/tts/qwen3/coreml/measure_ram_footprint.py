#!/usr/bin/env python3
"""Measure CoreML RAM using macOS footprint tool for accurate accounting."""
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
import coremltools as ct

def get_footprint_mb():
    """Get memory footprint using macOS footprint tool."""
    pid = os.getpid()
    result = subprocess.run(
        ['footprint', '-p', str(pid), '--swapped'],
        capture_output=True, text=True
    )
    # Parse output for "dirty" memory (actual RAM usage)
    # Format: "Dirty size:     123.45M"
    for line in result.stdout.split('\\n'):
        if 'dirty' in line.lower() and ('size' in line.lower() or ':' in line):
            # Extract number and unit
            match = re.search(r'([\\d.]+)\\s*([KMGT]?B?)', line, re.I)
            if match:
                val = float(match.group(1))
                unit = match.group(2).upper()
                if 'G' in unit:
                    return val * 1024
                elif 'K' in unit:
                    return val / 1024
                return val
    # Fallback to phys_footprint line
    for line in result.stdout.split('\\n'):
        if 'phys_footprint' in line.lower():
            match = re.search(r'([\\d.]+)\\s*([KMGT]?B?)', line, re.I)
            if match:
                val = float(match.group(1))
                unit = match.group(2).upper()
                if 'G' in unit:
                    return val * 1024
                elif 'K' in unit:
                    return val / 1024
                return val
    return -1

def get_memory_stats():
    """Get detailed memory stats."""
    pid = os.getpid()

    # RSS via ps
    rss_result = subprocess.run(['ps', '-p', str(pid), '-o', 'rss='], capture_output=True, text=True)
    rss_mb = int(rss_result.stdout.strip()) / 1024 if rss_result.stdout.strip() else 0

    # Virtual size via ps
    vsz_result = subprocess.run(['ps', '-p', str(pid), '-o', 'vsz='], capture_output=True, text=True)
    vsz_mb = int(vsz_result.stdout.strip()) / 1024 if vsz_result.stdout.strip() else 0

    # Footprint
    footprint_result = subprocess.run(['footprint', '-p', str(pid)], capture_output=True, text=True)

    # Parse physical footprint
    phys_footprint = 0
    for line in footprint_result.stdout.split('\\n'):
        if 'phys_footprint' in line.lower() or 'physical footprint' in line.lower():
            match = re.search(r'([\\d.]+)\\s*([KMGT]?)', line, re.I)
            if match:
                val = float(match.group(1))
                unit = match.group(2).upper() if match.group(2) else 'M'
                if 'G' in unit:
                    phys_footprint = val * 1024
                elif 'K' in unit:
                    phys_footprint = val / 1024
                else:
                    phys_footprint = val
                break

    return {
        "rss_mb": rss_mb,
        "vsz_mb": vsz_mb,
        "phys_footprint_mb": phys_footprint
    }

import re

path = sys.argv[1]
compute_units = sys.argv[2]

# Measure baseline
baseline = get_memory_stats()

# Load model
load_start = time.time()
cu = getattr(ct.ComputeUnit, compute_units)
model = ct.models.MLModel(path, compute_units=cu)
load_time = time.time() - load_start

# Measure after load
after = get_memory_stats()

# Calculate deltas
delta_rss = after["rss_mb"] - baseline["rss_mb"]
delta_vsz = after["vsz_mb"] - baseline["vsz_mb"]
delta_phys = after["phys_footprint_mb"] - baseline["phys_footprint_mb"]

print(json.dumps({
    "baseline": baseline,
    "after_load": after,
    "delta_rss_mb": delta_rss,
    "delta_vsz_mb": delta_vsz,
    "delta_phys_mb": delta_phys,
    "load_time_s": load_time
}))
'''

def get_dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
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

    # Look for JSON in stdout regardless of return code
    for line in reversed(result.stdout.strip().split('\n')):
        if line.startswith('{'):
            try:
                return json.loads(line)
            except:
                pass

    # Also check stderr for JSON (some outputs go there)
    for line in reversed(result.stderr.strip().split('\n')):
        if line.startswith('{'):
            try:
                return json.loads(line)
            except:
                pass

    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200] or "Unknown error"}
    return {"error": "No JSON output"}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_ram_footprint.py <model.mlpackage> [model2...]")
        sys.exit(0)

    model_paths = sys.argv[1:]
    venv_python = sys.executable
    results = []

    for path in model_paths:
        print("\n" + "=" * 80)
        print(f"Model: {os.path.basename(path)}")
        print("=" * 80)

        disk_size = get_dir_size_mb(path)
        print(f"Disk size: {disk_size:.0f} MB")

        model_results = {"name": os.path.basename(path), "disk_mb": disk_size}

        for cu in ["CPU_AND_GPU", "ALL"]:
            print(f"\n[{cu}]")
            r = measure_model_isolated(path, cu, venv_python)

            if "error" in r:
                print(f"  Error: {r['error']}")
            else:
                print(f"  Load time: {r['load_time_s']:.2f}s")
                print(f"  RSS delta:          {r['delta_rss_mb']:>8.0f} MB")
                print(f"  Virtual size delta: {r['delta_vsz_mb']:>8.0f} MB")
                print(f"  Phys footprint Δ:   {r['delta_phys_mb']:>8.0f} MB")
                model_results[f"{cu}_rss"] = r['delta_rss_mb']
                model_results[f"{cu}_vsz"] = r['delta_vsz_mb']
                model_results[f"{cu}_phys"] = r['delta_phys_mb']

        results.append(model_results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Physical Footprint (True RAM Impact)")
    print("=" * 80)
    print(f"{'Model':<40} {'Disk':>8} {'GPU Phys':>10} {'ALL Phys':>10}")
    print("-" * 80)
    for r in results:
        name = r["name"][:39]
        disk = f"{r['disk_mb']:.0f}M"
        gpu = f"{r.get('CPU_AND_GPU_phys', 0):.0f}M" if r.get('CPU_AND_GPU_phys') else "N/A"
        all_cu = f"{r.get('ALL_phys', 0):.0f}M" if r.get('ALL_phys') else "N/A"
        print(f"{name:<40} {disk:>8} {gpu:>10} {all_cu:>10}")

    # Combined totals
    print("-" * 80)
    prefill_fp32 = next((r for r in results if 'prefill_v9.mlpackage' in r['name']), {})
    prefill_fp16 = next((r for r in results if 'prefill_v9_fp16' in r['name']), {})
    prefill_int8 = next((r for r in results if 'prefill_v9_int8' in r['name']), {})
    decode_fp32 = next((r for r in results if 'decode_v4.mlpackage' in r['name']), {})
    decode_fp16 = next((r for r in results if 'decode_v4_fp16' in r['name']), {})
    decode_int8 = next((r for r in results if 'decode_v4_int8' in r['name']), {})

    print("\nCOMBINED (Prefill + Decode):")
    if prefill_fp32 and decode_fp32:
        disk = prefill_fp32.get('disk_mb',0) + decode_fp32.get('disk_mb',0)
        gpu = prefill_fp32.get('CPU_AND_GPU_phys',0) + decode_fp32.get('CPU_AND_GPU_phys',0)
        all_p = prefill_fp32.get('ALL_phys',0) + decode_fp32.get('ALL_phys',0)
        print(f"  FP32: Disk={disk:.0f}M, GPU={gpu:.0f}M, ALL={all_p:.0f}M")
    if prefill_fp16 and decode_fp16:
        disk = prefill_fp16.get('disk_mb',0) + decode_fp16.get('disk_mb',0)
        gpu = prefill_fp16.get('CPU_AND_GPU_phys',0) + decode_fp16.get('CPU_AND_GPU_phys',0)
        all_p = prefill_fp16.get('ALL_phys',0) + decode_fp16.get('ALL_phys',0)
        print(f"  FP16: Disk={disk:.0f}M, GPU={gpu:.0f}M, ALL={all_p:.0f}M")
    if prefill_int8 and decode_int8:
        disk = prefill_int8.get('disk_mb',0) + decode_int8.get('disk_mb',0)
        gpu = prefill_int8.get('CPU_AND_GPU_phys',0) + decode_int8.get('CPU_AND_GPU_phys',0)
        all_p = prefill_int8.get('ALL_phys',0) + decode_int8.get('ALL_phys',0)
        print(f"  INT8: Disk={disk:.0f}M, GPU={gpu:.0f}M, ALL={all_p:.0f}M")

    print("=" * 80)
