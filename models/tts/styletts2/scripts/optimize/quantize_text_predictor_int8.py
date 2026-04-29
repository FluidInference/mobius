"""Quantize all 5 text_predictor buckets to int8."""
import time
from pathlib import Path
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

PKG = Path(__file__).resolve().parents[2] / "coreml"

op_cfg = OpLinearQuantizerConfig(
    mode="LINEAR_SYMMETRIC",
    dtype="int8",
    granularity="per_channel",
    weight_threshold=200_000,
)
config = OptimizationConfig(global_config=op_cfg)

def du(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024

total_fp16 = total_int8 = 0
for tok in (32, 64, 128, 256, 512):
    src = PKG / f"styletts2_text_predictor_{tok}.mlpackage"
    dst = PKG / f"styletts2_text_predictor_{tok}_int8.mlpackage"
    print(f"[{tok}] loading …")
    t0 = time.time()
    m = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f"  loaded ({time.time()-t0:.1f}s); quantizing …")
    t0 = time.time()
    mq = linear_quantize_weights(m, config)
    print(f"  quantized ({time.time()-t0:.1f}s); saving …")
    mq.save(str(dst))
    fp16 = du(src); int8 = du(dst)
    total_fp16 += fp16; total_int8 += int8
    print(f"  fp16={fp16:.1f} MB  int8={int8:.1f} MB  reduction={1-int8/fp16:.1%}")

print()
print(f"TOTAL fp16: {total_fp16:.1f} MB")
print(f"TOTAL int8: {total_int8:.1f} MB")
print(f"saved: {total_fp16-total_int8:.1f} MB ({1-total_int8/total_fp16:.1%})")
