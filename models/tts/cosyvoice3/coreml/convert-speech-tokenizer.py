"""Convert speech_tokenizer_v3.onnx to a CoreML mlpackage.

Speech-tokenizer-v3 is the discrete-token extractor that turns a prompt wav's
Whisper log-mel (n_mels=128) into integer speech-token IDs the LLM consumes.

Pipeline (upstream):
    prompt_wav (16 kHz, mono)
        → whisper.log_mel_spectrogram(n_mels=128)   # shape (1, 128, T)
        → speech_tokenizer_v3(feats, feats_length)  # → (1, T/4) int32

T is the number of mel frames (10 ms stride / 25 ms window at 16 kHz
via whisper's default n_fft=400, hop=160 ... actually whisper.log_mel_spectrogram
uses hop=160 so 1 s ≈ 100 frames).  The token stream is downsampled 4×.

We pick a fixed static T so the model lands on ANE cleanly.  Default T=500
covers 5 s of prompt audio (= 125 output tokens), which is enough for Flow-N125.
For Flow-N250 use T=1000.

Usage:
    uv run python convert-speech-tokenizer.py --output-dir ./build/speech-tok \
        --t-frames 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import onnxruntime as ort
import torch

# Register `greater_equal` and `less_equal` torch-ops for coremltools 9.
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil import Builder as mb


@register_torch_op()
def greater_equal(context, node):
    inputs = _get_inputs(context, node, expected=2)
    res = mb.greater_equal(x=inputs[0], y=inputs[1], name=node.name)
    context.add(res)


@register_torch_op()
def less_equal(context, node):
    inputs = _get_inputs(context, node, expected=2)
    res = mb.less_equal(x=inputs[0], y=inputs[1], name=node.name)
    context.add(res)


# Patch onnx2torch to handle GreaterOrEqual v16 (ONNX opset 16).
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
_GE12 = OperationDescription(domain="", operation_type="GreaterOrEqual", version=12)
for _v in (13, 14, 15, 16):
    _CONVERTER_REGISTRY[
        OperationDescription(domain="", operation_type="GreaterOrEqual", version=_v)
    ] = _CONVERTER_REGISTRY[_GE12]

from onnx2torch import convert as onnx_to_torch  # noqa: E402


HERE = Path(__file__).parent
ONNX_PATH = HERE / "cosyvoice3_dl" / "speech_tokenizer_v3.onnx"


def _make_precision(fp16: bool):
    """FP16 everywhere except numerically-sensitive ops.

    SpeechTokenizerV3 is a Conformer encoder + vector-quantizer head.  The VQ
    does (x - codebook)**2 → sum → argmin, which is highly sensitive to fp16
    rounding (tiny errors flip the argmin to a neighbor).  Keep RMSNorm,
    softmax attention, and the VQ distance math in fp32; convolutional /
    linear bulk can land on ANE in fp16.
    """
    if not fp16:
        return ct.precision.FLOAT32
    FP32_OPS = {
        "pow", "reduce_mean", "rsqrt", "softmax",
        "reduce_sum", "sub",  # VQ distance computation
    }
    return ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in FP32_OPS
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(HERE / "build" / "speech-tok"))
    p.add_argument("--t-frames", type=int, default=500,
                   help="Fixed Whisper mel-frame length. Output tokens = T/4.")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = args.t_frames
    print(f"[1/4] Loading ONNX: {ONNX_PATH.name}")
    onnx_model = onnx.load(str(ONNX_PATH))
    print(f"      nodes={len(onnx_model.graph.node)} opset={onnx_model.opset_import[0].version}")

    # Deterministic sanity input.
    torch.manual_seed(0)
    feat_np = (torch.randn(1, 128, T) * 0.3).numpy().astype(np.float32)
    flen_np = np.array([T], dtype=np.int32)

    print(f"[2/4] onnxruntime reference  T={T}")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {
        sess.get_inputs()[0].name: feat_np,
        sess.get_inputs()[1].name: flen_np,
    })[0]
    print(f"      output {ref.shape}  dtype={ref.dtype}  "
          f"range=[{ref.min()}, {ref.max()}]  first10={ref.flatten()[:10].tolist()}")

    print(f"[3/4] ONNX → PyTorch via onnx2torch, then trace")
    torch_model = onnx_to_torch(onnx_model).eval()
    feat_t = torch.from_numpy(feat_np)
    flen_t = torch.from_numpy(flen_np)
    with torch.no_grad():
        torch_out = torch_model(feat_t, flen_t).numpy()
    mism = int((torch_out != ref).sum())
    print(f"      onnx↔torch  mismatches={mism}/{ref.size}")
    if mism:
        print(f"        torch first10={torch_out.flatten()[:10].tolist()}")

    # trace.  strict=False because of dynamic control flow inside attention mask.
    traced = torch.jit.trace(torch_model, (feat_t, flen_t), strict=False)

    print(f"      Converting to CoreML mlpackage (T={T} static)")
    precision = _make_precision(args.fp16)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="feats",        shape=(1, 128, T), dtype=np.float32),
            ct.TensorType(name="feats_length", shape=(1,),        dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="indices", dtype=np.int32)],
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )

    tag = "fp16" if args.fp16 else "fp32"
    mlp = out_dir / f"SpeechTokenizerV3-T{T}-{tag}.mlpackage"
    mlmodel.save(str(mlp))
    print(f"      saved: {mlp}")

    # Parity check
    print(f"[4/4] Parity check CoreML vs ONNX")
    ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    cm = ml.predict({"feats": feat_np, "feats_length": flen_np})
    cm_out = np.asarray(list(cm.values())[0]).astype(np.int32)
    mism = int((cm_out != ref).sum())
    print(f"      coreml output {cm_out.shape}  mismatches vs onnx: {mism}/{ref.size}")
    if mism:
        # Show first few differences for diagnosis.
        diff_idx = np.argwhere(cm_out != ref)[:10]
        for idx in diff_idx:
            i, j = idx.tolist()
            print(f"        @[{i},{j}]  onnx={ref[i,j]}  coreml={cm_out[i,j]}")


if __name__ == "__main__":
    main()
