"""Measure diffusion_step perf across buckets to inform pruning decision."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import coremltools as ct

PKG = Path("/Users/kikow/brandon/voicelink/mobius-styletts2/models/tts/styletts2/coreml")
BUCKETS = (32, 64, 128, 256, 512)

# 5-step ADPM2 = 9 model calls per generation (n*2 - 1)
# Use 5 calls (one per sigma) as proxy
N_CALLS = 9

print(f"{'bucket':>8s}  {'1-call':>10s}  {'5-step':>10s}")
for tb in BUCKETS:
    m = ct.models.MLModel(str(PKG / f"styletts2_diffusion_step_{tb}.mlpackage"),
                          compute_units=ct.ComputeUnit.CPU_ONLY)
    x = np.random.randn(1, 1, 256).astype(np.float32)
    sigma = np.array([1.5], dtype=np.float32)
    emb = np.random.randn(1, tb, 768).astype(np.float32)
    feat = np.random.randn(1, 256).astype(np.float32)
    inputs = {"x_noisy": x, "sigma": sigma, "embedding": emb, "features": feat}

    # warmup
    m.predict(inputs)
    # time
    t0 = time.time()
    for _ in range(N_CALLS):
        m.predict(inputs)
    dt = (time.time() - t0) / N_CALLS

    print(f"  {tb:>4d}    {dt*1000:>8.1f}ms   {dt*N_CALLS*1000:>8.1f}ms")
