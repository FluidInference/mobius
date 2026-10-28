"""Weight-only int8 (per-channel linear, the upstream paradee_int8.onnx scheme) copies of the fp32 packages.
The phase-lock filter's 1024-point DFT bases stay fp32 (upstream regenerates them exactly at load time too).
usage: quantize.py [--src build] [--out build-int8]"""
import argparse, pathlib, shutil, coremltools as ct, coremltools.optimize.coreml as cto

N_FFT = 1024


def dft_linear_ops(path):
    blk = ct.utils.load_spec(path).mlProgram.functions["main"].block_specializations["CoreML7"]
    big = {op.outputs[0].name for op in blk.operations if op.type == "const"
           and N_FFT in [d.constant.size for d in op.outputs[0].type.tensorType.dimensions]}
    return [op.outputs[0].name for op in blk.operations if op.type == "linear"
            and any(b.name in big for b in op.inputs["weight"].arguments)]


ap = argparse.ArgumentParser(); ap.add_argument("--src", default="build"); ap.add_argument("--out", default="build-int8"); A = ap.parse_args()
out = pathlib.Path(A.out); out.mkdir(exist_ok=True)
glob_cfg = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=1024)
for name in ("ParadeeText", "ParadeeAcoustic"):
    p = f"{A.src}/{name}.mlpackage"
    keep = dft_linear_ops(p) if name == "ParadeeAcoustic" else []
    print(name, "kept fp32:", keep)
    cfg = cto.OptimizationConfig(global_config=glob_cfg, op_name_configs={k: None for k in keep})
    cto.linear_quantize_weights(ct.models.MLModel(p, skip_model_load=True), cfg).save(str(out / f"{name}.mlpackage"))
shutil.copy(f"{A.src}/ref.npz", out / "ref.npz")
