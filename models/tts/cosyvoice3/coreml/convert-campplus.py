"""Convert CAMPPlus speaker-embedding ONNX model to CoreML mlpackage.

CAMPPlus takes FBANK features (1, T, 80) and produces a pooled 192-dim speaker
embedding (1, 192). T is the number of FBANK frames of the prompt wav.

We pick a fixed bucket T=300 (3 s at 10 ms/frame) for ANE-friendly static shape.
Callers pad or truncate FBANK input to T frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx2torch import convert as onnx_to_torch

# Register `prod` op (onnx2torch emits torch.prod for ReduceProd in shape math).
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil import Builder as mb


@register_torch_op
def prod(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    if len(inputs) >= 2 and inputs[1] is not None:
        axis = inputs[1].val if hasattr(inputs[1], "val") else int(inputs[1])
        keep_dims = bool(inputs[2].val) if len(inputs) >= 3 and inputs[2] is not None else False
        res = mb.reduce_prod(x=x, axes=[int(axis)], keep_dims=keep_dims, name=node.name)
    else:
        res = mb.reduce_prod(x=x, keep_dims=False, name=node.name)
    context.add(res)

HERE = Path(__file__).parent
ONNX_PATH = HERE / "cosyvoice3_dl" / "campplus.onnx"


def _make_precision(fp16: bool):
    """FP16 everywhere except BatchNorm-style ops.

    CAMPPlus is mostly TDNN (conv+BN+relu) and a statistics-pooling head.
    Keep the BN normalization math (pow / reduce_mean / rsqrt) and any
    softmax in fp32 to protect the running statistics from fp16 drift.
    """
    if not fp16:
        return ct.precision.FLOAT32
    FP32_OPS = {"pow", "reduce_mean", "rsqrt", "softmax"}
    return ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in FP32_OPS
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(HERE / "build" / "campplus"))
    p.add_argument("--t-frames", type=int, default=300, help="Fixed fbank length")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading ONNX: {ONNX_PATH.name}")
    onnx_model = onnx.load(str(ONNX_PATH))
    print(f"      nodes={len(onnx_model.graph.node)} opset={onnx_model.opset_import[0].version}")

    # Sanity-check input/output shapes with onnxruntime.
    T = args.t_frames
    print(f"[2/4] Sanity check with onnxruntime, T={T}")
    s = ort.InferenceSession(str(ONNX_PATH))
    x = np.random.randn(1, T, 80).astype(np.float32)
    ref = s.run(None, {"input": x})[0]
    print(f"      input  (1, {T}, 80)  output {ref.shape}  range=[{ref.min():.3f}, {ref.max():.3f}]")

    print(f"[3/4] ONNX → PyTorch via onnx2torch, then trace")
    torch_model = onnx_to_torch(onnx_model).eval()
    dummy = torch.from_numpy(x)
    with torch.no_grad():
        torch_out = torch_model(dummy).numpy()
    d_onnx_torch = np.abs(ref - torch_out)
    print(f"      onnx↔torch  MAE={d_onnx_torch.mean():.3e}  max={d_onnx_torch.max():.3e}")
    traced = torch.jit.trace(torch_model, dummy, strict=False)

    print(f"      Converting to CoreML mlpackage (T={T} static)")
    precision = _make_precision(args.fp16)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, T, 80), dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )

    tag = "fp16" if args.fp16 else "fp32"
    mlp = out_dir / f"CAMPPlus-T{T}-{tag}.mlpackage"
    mlmodel.save(str(mlp))
    print(f"      saved: {mlp}")

    # Parity check
    print(f"[4/4] Parity check CoreML vs ONNX")
    ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    cm = ml.predict({"input": x})
    cm_out = np.asarray(list(cm.values())[0])
    d = np.abs(ref - cm_out)
    print(f"      output shape: {cm_out.shape}")
    print(f"      MAE={d.mean():.3e}  max={d.max():.3e}")
    corr = np.corrcoef(ref.flatten(), cm_out.flatten())[0, 1]
    print(f"      corr={corr:.6f}")


if __name__ == "__main__":
    main()
