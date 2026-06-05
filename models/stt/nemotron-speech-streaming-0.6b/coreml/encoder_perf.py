#!/usr/bin/env python3
"""Isolated load-time / inference-latency / peak-memory probe for one encoder config.

Run as a subprocess (fresh process => clean peak-RSS) per config:
  encoder_perf.py <encoder.mlpackage|.mlmodelc> <CPU_ONLY|CPU_AND_NE|CPU_AND_GPU> <metadata.json>
Prints one line: load_s, infer_ms_median, peak_rss_mb.
"""
import os
os.environ["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:" + os.environ.get("PATH", "")
import sys, json, time, resource
import numpy as np
import coremltools as ct

path, unit_name, meta_path = sys.argv[1], sys.argv[2], sys.argv[3]
unit = {"CPU_ONLY": ct.ComputeUnit.CPU_ONLY, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU}[unit_name]
md = json.load(open(meta_path))
tmf = md["total_mel_frames"]; cc = md["cache_channel_shape"]; ctt = md["cache_time_shape"]
feed = {"mel": np.random.randn(1, 128, tmf).astype(np.float32),
        "mel_length": np.array([tmf], dtype=np.int32),
        "cache_channel": np.random.randn(*cc).astype(np.float32),
        "cache_time": np.random.randn(*ctt).astype(np.float32),
        "cache_len": np.array([0], dtype=np.int32)}

t0 = time.perf_counter()
if path.endswith(".mlmodelc"):
    m = ct.models.CompiledMLModel(path, unit)
else:
    m = ct.models.MLModel(path, compute_units=unit)
load_s = time.perf_counter() - t0

for _ in range(3):
    m.predict(feed)
t = []
for _ in range(25):
    s = time.perf_counter(); m.predict(feed); t.append((time.perf_counter() - s) * 1000)
t = sorted(t)
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS: bytes
line = f"{sys.argv[5] if len(sys.argv) > 5 else path} {unit_name} load={load_s:.2f}s infer={t[len(t)//2]:.2f}ms mem={peak_mb:.0f}MB"
print(f"RESULT {line}")
if len(sys.argv) > 4 and sys.argv[4]:
    with open(sys.argv[4], "a") as f:
        f.write(line + "\n")
