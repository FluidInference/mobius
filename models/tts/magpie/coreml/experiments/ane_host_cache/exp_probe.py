"""Probe ANE admission for OLD vs NEW decoder_step under REAL incrementing position.

The documented failure (SWIFT_PORT_FINDINGS.md) is not at convert time — it's a per-call
ANE recompile that fails once `position` starts incrementing during synthesis. So we must
drive an actual decode loop, not a single dummy predict. Runs the loop on CPU_AND_NE,
CPU_AND_GPU, CPU_ONLY and reports admission + p50 latency.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import coremltools as ct

MASK_NEG = -3.0e4
N_LAYERS = 12
D_MODEL = 768
SA_HEADS = 12
D_HEAD = 64
MAX_SEQ = 512
T_ENC = 256


def is_old(mlmodel) -> bool:
    return any(i.name == "position0" for i in mlmodel.get_spec().description.input)


def run_loop(pkg: str, cu_name: str, steps: int):
    cu = getattr(ct.ComputeUnit, cu_name)
    try:
        m = ct.models.MLModel(pkg, compute_units=cu)
    except Exception as e:
        return f"LOAD FAIL: {str(e).replace(chr(10),' ')[:110]}"

    old = is_old(m)
    spec = m.get_spec()
    # Map outputs by shape: cache slices are [.,.,12,64]; logits is the big one.
    slice_out_names = []
    for o in spec.description.output:
        shp = tuple(int(d) for d in o.type.multiArrayType.shape)
        if shp[-2:] == (SA_HEADS, D_HEAD):
            slice_out_names.append(o.name)
    audio = (np.random.randn(1, 1, D_MODEL) * 0.1).astype(np.float32)
    enc = (np.random.randn(1, T_ENC, D_MODEL) * 0.1).astype(np.float32)

    # host cache buffers
    pk = [np.zeros((1, MAX_SEQ, SA_HEADS, D_HEAD), np.float32) for _ in range(N_LAYERS)]
    pv = [np.zeros((1, MAX_SEQ, SA_HEADS, D_HEAD), np.float32) for _ in range(N_LAYERS)]

    times = []
    try:
        for pos in range(steps):
            feed = {"audio_embed": audio, "encoder_output": enc}
            if old:
                feed["encoder_mask"] = np.ones((1, T_ENC), np.float32)
                for i in range(N_LAYERS):
                    feed[f"cache_k{i}"] = pk[i]
                    feed[f"cache_v{i}"] = pv[i]
                    feed[f"position{i}"] = np.array([float(pos)], np.float32)
            else:
                mem = np.zeros((1, 1, 1, T_ENC), np.float32)
                am = np.full((1, 1, 1, MAX_SEQ + 1), MASK_NEG, np.float32)
                am[..., :pos] = 0.0        # past positions written so far
                am[..., MAX_SEQ] = 0.0     # current token (last col)
                feed["mem_mask_add"] = mem
                feed["attn_mask"] = am
                for i in range(N_LAYERS):
                    feed[f"cache_k{i}"] = pk[i]
                    feed[f"cache_v{i}"] = pv[i]

            t0 = time.time()
            out = m.predict(feed)
            times.append((time.time() - t0) * 1000)

            # write back caches for the next step
            if old:
                for i in range(N_LAYERS):
                    pk[i] = out[f"new_ck{i}"] if f"new_ck{i}" in out else pk[i]
                    pv[i] = out[f"new_cv{i}"] if f"new_cv{i}" in out else pv[i]
            else:
                # cache-slice outputs in spec order are [nk0,nv0,nk1,nv1,...]; host appends at `pos`
                for i in range(N_LAYERS):
                    nk = out[slice_out_names[2 * i]]
                    nv = out[slice_out_names[2 * i + 1]]
                    pk[i][:, pos] = nk.reshape(1, SA_HEADS, D_HEAD)
                    pv[i][:, pos] = nv.reshape(1, SA_HEADS, D_HEAD)
    except Exception as e:
        msg = str(e).replace(chr(10), " ")
        marker = "ANECompile FAIL (-14 / ANECCompile)" if ("-14" in msg or "ANECompile" in msg or "ANECCompile" in msg) else msg[:90]
        return f"FAIL @ step {len(times)}: {marker}"

    t = np.array(times[2:]) if len(times) > 2 else np.array(times)
    return f"OK  {len(times)} steps  p50 {np.percentile(t,50):.1f}ms  p99 {np.percentile(t,99):.1f}ms"


def device_breakdown(pkg: str) -> str:
    """Per-op preferred device under CPU_AND_NE via MLComputePlan (like the Qwen probe)."""
    try:
        from coremltools.models.compute_plan import MLComputePlan
        _keep = ct.models.MLModel(pkg)  # keep alive so temp compiled dir survives
        mlmodelc = _keep.get_compiled_model_path()
        plan = MLComputePlan.load_from_path(mlmodelc, compute_units=ct.ComputeUnit.CPU_AND_NE)
        counts = {}
        for func in plan.model_structure.program.functions.values():
            for op in func.block.operations:
                du = plan.get_compute_device_usage_for_mlprogram_operation(op)
                dev = type(du.preferred_compute_device).__name__ if du else "None"
                counts[dev] = counts.get(dev, 0) + 1
        ne = counts.get("MLNeuralEngineComputeDevice", 0)
        cpu = counts.get("MLCPUComputeDevice", 0)
        gpu = counts.get("MLGPUComputeDevice", 0)
        assigned = ne + cpu + gpu
        pct = f"{100*ne/assigned:.1f}%" if assigned else "n/a"
        return f"ANE {ne} / CPU {cpu} / GPU {gpu} (device-assigned) -> {pct} ANE"
    except Exception as e:
        return f"unavailable: {str(e).replace(chr(10),' ')[:80]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=str(Path(__file__).parent / "build_ane"))
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    bd = Path(args.build_dir)
    for label, fname in [("OLD (in-graph blend + pos compares)", "old_decoder_step.mlpackage"),
                         ("NEW (§6.3 host-owned cache)", "new_decoder_step.mlpackage")]:
        pkg = bd / fname
        print(f"\n=== {label} : {fname} ===")
        if not pkg.exists():
            print("  (missing — run exp_convert.py)")
            continue
        for cu in ["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"]:
            print(f"  {cu:14s} : {run_loop(str(pkg), cu, args.steps)}", flush=True)
        print(f"  {'placement':14s} : {device_breakdown(str(pkg))}", flush=True)


if __name__ == "__main__":
    main()
