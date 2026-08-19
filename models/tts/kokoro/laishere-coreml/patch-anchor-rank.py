#!/usr/bin/env python3
"""Rewrite KokoroVocoder's `anchor` output chain to be rank-stable.

Usage: python patch-anchor-rank.py <src.mlpackage> <dst.mlpackage>

Before:  reduce_mean(x_pre, keep_dims=false) -> rank-0
         expand_dims(axes=[0])               -> [1]  (= anchor)
After:   reduce_mean(x_pre, axes=[1,2], keep_dims=false) -> [1]  (= anchor)

macOS 14's Espresso runtime mishandles the rank-0 intermediate
("Output rank has changed after reshaping espresso network ...
blob = anchor"), producing rank 2 and failing output validation
(issue FluidInference/FluidAudio#836). The value of `anchor` is
discarded by all hosts, and `x_pre` is untouched by this patch
(verified bit-exact on the en and zh vocoders).

`convert-coreml.py` now emits the rank-stable form natively
(`x_pre.mean(dim=(1, 2))`); this script retrofits already-published
mlpackages (ANE / ANE-zh; ANE-ja's vocoder is byte-identical to
ANE's) without re-running conversion and palettization. Recompile to
mlmodelc afterwards via `MLModel.get_compiled_model_path()` or
`coremlcompiler compile`.
"""
import sys

import coremltools as ct
from coremltools.proto import MIL_pb2

src, dst = sys.argv[1], sys.argv[2]

spec = ct.utils.load_spec(src)
prog = spec.mlProgram
fn = prog.functions["main"]
block = fn.block_specializations[fn.opset]
ops = block.operations

# Locate expand_dims producing `anchor`, its reduce_mean input, and the
# const feeding expand_dims' axes.
expand_idx = None
for i, op in enumerate(ops):
    if op.type == "expand_dims" and any(o.name == "anchor" for o in op.outputs):
        expand_idx = i
        break
assert expand_idx is not None, "expand_dims -> anchor not found"
expand_op = ops[expand_idx]

reduce_name = expand_op.inputs["x"].arguments[0].name
axes_const_name = expand_op.inputs["axes"].arguments[0].name

reduce_idx = axes_const_idx = None
for i, op in enumerate(ops):
    if any(o.name == reduce_name for o in op.outputs):
        reduce_idx = i
    if op.type == "const" and any(o.name == axes_const_name for o in op.outputs):
        axes_const_idx = i
assert reduce_idx is not None, "reduce_mean not found"
reduce_op = ops[reduce_idx]
assert reduce_op.type == "reduce_mean", reduce_op.type
assert "axes" not in reduce_op.inputs, "reduce_mean already has axes"

# Repurpose the old expand_dims axes const as the reduce axes [1, 2].
assert axes_const_idx is not None, "axes const not found"
axes_const = ops[axes_const_idx]
val = axes_const.attributes["val"]
assert val.type.tensorType.dataType == MIL_pb2.DataType.INT32
val.type.tensorType.dimensions[0].constant.size = 2
del val.immediateValue.tensor.ints.values[:]
val.immediateValue.tensor.ints.values.extend([1, 2])
axes_const.outputs[0].type.tensorType.dimensions[0].constant.size = 2

# reduce_mean: add axes input, rename output to `anchor`, type fp16 [1].
arg = reduce_op.inputs["axes"].arguments.add()
arg.name = axes_const_name
out = reduce_op.outputs[0]
out.name = "anchor"
tt = out.type.tensorType
tt.rank = 1
del tt.dimensions[:]
tt.dimensions.add().constant.size = 1

# Drop the expand_dims op and move the axes const before the reduce_mean
# (MIL requires topological op order).
new_ops = []
for i, op in enumerate(ops):
    if i == expand_idx or i == axes_const_idx:
        continue
    if i == reduce_idx:
        new_ops.append(MIL_pb2.Operation())
        new_ops[-1].CopyFrom(axes_const)
    new_ops.append(MIL_pb2.Operation())
    new_ops[-1].CopyFrom(op)
del ops[:]
ops.extend(new_ops)

model = ct.models.MLModel(
    spec,
    weights_dir=src + "/Data/com.apple.CoreML/weights",
    skip_model_load=True,
)
model.save(dst)
print(f"patched: {dst}")
