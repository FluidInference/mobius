import sys, coremltools as ct
from coremltools.optimize.coreml import (OpLinearQuantizerConfig, OpPalettizerConfig,
    OptimizationConfig, linear_quantize_weights, palettize_weights)
src, dst, kind = sys.argv[1], sys.argv[2], sys.argv[3]
m = ct.models.MLModel(src, compute_units=ct.ComputeUnit.CPU_AND_NE)
try: m.minimum_deployment_target = ct.target.iOS17
except Exception: pass
if kind == "int4":
    cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", granularity="per_channel", dtype="int4"))
    q = linear_quantize_weights(m, cfg)
elif kind == "pal6":
    cfg = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=6))
    q = palettize_weights(m, cfg)
elif kind == "pal4":
    cfg = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=4))
    q = palettize_weights(m, cfg)
q.save(dst); print("saved", dst)
