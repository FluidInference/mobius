import sys, coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights)
src, dst = sys.argv[1], sys.argv[2]
m = ct.models.MLModel(src, compute_units=ct.ComputeUnit.CPU_AND_NE)
try: m.minimum_deployment_target = ct.target.iOS17
except Exception: pass
cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
    mode="linear_symmetric", granularity="per_channel", dtype="int8"))
q = linear_quantize_weights(m, cfg)
q.save(dst)
print("saved", dst)
