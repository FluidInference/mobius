#!/usr/bin/env python
"""Find the minimal set of encoder MatMuls whose INT8 weight-compression
breaks OV 2026.2.1 CPU compile, so we can exclude only those and compress
everything else (near-full weight-only int8)."""
import openvino as ov, nncf, sys
core = ov.Core()
SRC = "build_ov/nemotron_encoder.xml"
m0 = core.read_model(SRC)
mm = [op.get_friendly_name() for op in m0.get_ops()
      if op.get_type_name() == "MatMul"
      and any(op.input_value(i).get_node().get_type_name() == "Constant" for i in range(2))]
print(f"compressible matmuls: {len(mm)}", flush=True)

def compiles(compress_list, extra_ignore):
    m = core.read_model(SRC)
    ignore = [n for n in mm if n not in set(compress_list)] + list(extra_ignore)
    q = nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT8_SYM,
            ignored_scope=nncf.IgnoredScope(names=ignore))
    try:
        core.compile_model(q, "CPU"); return True
    except Exception:
        return False

offenders = []
# iteratively peel offenders: compress all matmuls except known offenders;
# if it fails, bisect the prefix to find the first new offender.
while True:
    active = [n for n in mm if n not in set(offenders)]
    if compiles(active, offenders):
        print("ALL-CLEAR with offenders excluded:", offenders, flush=True)
        break
    # bisect prefix of `active`: find smallest k where compress(active[:k]) fails
    lo, hi = 0, len(active)            # compiles(active[:0]) trivially true
    # ensure full active fails (it does, since we're here)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if compiles(active[:mid], offenders):
            lo = mid
        else:
            hi = mid
    bad = active[hi-1]
    offenders.append(bad)
    print(f"[offender #{len(offenders)}] idx-in-active={hi-1}: {bad}", flush=True)
    if len(offenders) > 30:
        print("too many offenders, aborting", flush=True); sys.exit(1)

print("FINAL OFFENDERS:", offenders, flush=True)
print(f"can compress {len(mm)-len(offenders)}/{len(mm)} matmuls", flush=True)
