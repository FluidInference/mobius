"""Weight-only int8 (per-channel linear, the upstream paradee_int8.onnx scheme) copies of the v1.2 fp32 packages.
The head's DFT bases (600- and 150-point STFT / iSTFT) and its two frequency-warp matrices stay fp32: upstream
regenerates the bases exactly at load time, and the warp matrices are interpolation weights (96 KB together).
usage: quantize12.py [--src build-v1.2] [--out build-v1.2-int8]"""
import argparse
import pathlib
import shutil

import coremltools as ct
import coremltools.optimize.coreml as cto

KEEP_SHAPES = {(602, 600), (600, 602), (152, 150), (150, 152), (301, 80), (76, 24)}


def head_constant_ops(path):
    blk = ct.utils.load_spec(path).mlProgram.functions["main"].block_specializations["CoreML7"]
    keep = {op.outputs[0].name for op in blk.operations if op.type == "const"
            and tuple(d.constant.size for d in op.outputs[0].type.tensorType.dimensions) in KEEP_SHAPES}
    return [op.outputs[0].name for op in blk.operations if op.type == "linear"
            and any(b.name in keep for b in op.inputs["weight"].arguments)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="build-v1.2")
    ap.add_argument("--out", default="build-v1.2-int8")
    A = ap.parse_args()
    out = pathlib.Path(A.out)
    out.mkdir(exist_ok=True)
    glob_cfg = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel",
                                           weight_threshold=1024)
    for name in ("ParadeeText", "ParadeeAcoustic"):
        p = f"{A.src}/{name}.mlpackage"
        keep = head_constant_ops(p) if name == "ParadeeAcoustic" else []
        print(name, f"kept fp32 ({len(keep)}):", keep)
        cfg = cto.OptimizationConfig(global_config=glob_cfg, op_name_configs={k: None for k in keep})
        cto.linear_quantize_weights(ct.models.MLModel(p, skip_model_load=True), cfg).save(str(out / f"{name}.mlpackage"))
    shutil.copy(f"{A.src}/ref.npz", out / "ref.npz")


if __name__ == "__main__":
    main()
