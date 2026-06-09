"""Convert the ANE-eligible FlowLM step (rank-4, scatter-free) to CoreML.

Emits `flowlm_step_ane.mlpackage`. See `traceable_flowlm_step_ane.py` for
why this variant exists (Phase 7: rank-5 KV cache -> ANECCompile FAILED).

Verify placement afterwards with:
    uv run coreml-cli build/<language>/flowlm_step_ane.mlpackage --fallback
"""
import argparse
import os
import sys

import coremltools as ct
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, _SCRIPT_DIR)  # for: from _language_arg import ...
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))  # for: from traceable_* import ...

from _language_arg import (
    add_compute_args,
    add_language_arg,
    build_output_dir,
    resolve_compute_precision,
    resolve_compute_units,
)
from traceable_flowlm_step_ane import TraceableFlowLMStepANE


def convert_flowlm_step_ane(
    language: str,
    compute_precision: str = "fp16",
    compute_units: str = "ALL",
):
    print(f"Loading model (language={language})...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    print("Creating ANE step model...")
    max_seq_len = 512
    step_model = TraceableFlowLMStepANE.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    step_model.eval()
    num_layers = step_model.num_layers
    H = step_model.num_heads
    D = step_model.head_dim
    print(f"num_layers={num_layers}, max_seq_len={max_seq_len}")

    print("Creating example inputs...")
    sequence = torch.randn(1, 1, 32)
    bos_emb = model.flow_lm.bos_emb.data

    prefix_len = 136
    trace_inputs = [sequence, bos_emb]
    for _ in range(num_layers):
        k_cache = torch.zeros(1, max_seq_len, H, D)
        v_cache = torch.zeros(1, max_seq_len, H, D)
        k_cache[:, :prefix_len] = torch.randn(1, prefix_len, H, D)
        v_cache[:, :prefix_len] = torch.randn(1, prefix_len, H, D)
        trace_inputs.extend([k_cache, v_cache, torch.tensor([float(prefix_len)])])

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(step_model, tuple(trace_inputs))

    # NOTE: fp32 IO contract (`dtype=np.float32` on every TensorType + fp32
    # outputs) — same reasoning as convert_flowlm_step.py: internal ops run
    # fp16, coremltools inserts boundary casts, and Swift drives the model
    # with `MLMultiArrayDataType.float32` buffers (the macOS MLE5 binder
    # rejects fp16 MLMultiArrays).
    print("Converting to CoreML (precision={}, IO=fp32)...".format(compute_precision))
    inputs = [
        ct.TensorType(name="sequence", shape=(1, 1, 32), dtype=np.float32),
        ct.TensorType(name="bos_emb", shape=(32,), dtype=np.float32),
    ]
    for i in range(num_layers):
        inputs.append(
            ct.TensorType(name=f"k_cache{i}", shape=(1, max_seq_len, H, D), dtype=np.float32))
        inputs.append(
            ct.TensorType(name=f"v_cache{i}", shape=(1, max_seq_len, H, D), dtype=np.float32))
        inputs.append(ct.TensorType(name=f"position{i}", shape=(1,), dtype=np.float32))

    # Outputs: (x, is_eos, then per layer (new_k, new_v, new_position)).
    # EXPLICIT names (positional match against the traced return tuple): the
    # k/v caches share one shape, so the Swift host cannot disambiguate them
    # by shape-bucketing the way it does for the rank-5 packs — stable names
    # replace discovery entirely.
    ct_outputs = [
        ct.TensorType(name="transformer_out", dtype=np.float32),
        ct.TensorType(name="is_eos", dtype=np.float32),
    ]
    for i in range(num_layers):
        ct_outputs.append(ct.TensorType(name=f"new_k_cache{i}", dtype=np.float32))
        ct_outputs.append(ct.TensorType(name=f"new_v_cache{i}", dtype=np.float32))
        ct_outputs.append(ct.TensorType(name=f"new_position{i}", dtype=np.float32))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "flowlm_step_ane.mlpackage")
    print(f"Saving to {output_path} (precision={compute_precision})...")
    mlmodel.save(output_path)

    # Smoke test + parity vs the traced torch module on the same inputs.
    print(f"\nTesting CoreML model (compute_units={compute_units})...")
    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))

    feed = {"sequence": np.random.randn(1, 1, 32).astype(np.float32),
            "bos_emb": bos_emb.numpy().astype(np.float32)}
    torch_inputs = [torch.from_numpy(feed["sequence"]), bos_emb]
    rng = np.random.default_rng(0)
    for i in range(num_layers):
        k = np.zeros((1, max_seq_len, H, D), dtype=np.float32)
        v = np.zeros((1, max_seq_len, H, D), dtype=np.float32)
        k[:, :prefix_len] = rng.standard_normal((1, prefix_len, H, D), dtype=np.float32)
        v[:, :prefix_len] = rng.standard_normal((1, prefix_len, H, D), dtype=np.float32)
        feed[f"k_cache{i}"] = k
        feed[f"v_cache{i}"] = v
        feed[f"position{i}"] = np.array([float(prefix_len)], dtype=np.float32)
        torch_inputs.extend(
            [torch.from_numpy(k), torch.from_numpy(v), torch.tensor([float(prefix_len)])])

    outputs = coreml_model.predict(feed)
    with torch.no_grad():
        ref = step_model(*torch_inputs)

    got_out = outputs["transformer_out"]
    got_eos = outputs["is_eos"]
    d_out = np.abs(ref[0].numpy() - got_out).max()
    d_eos = np.abs(ref[1].numpy() - got_eos).max()
    # Layer-output naming check: caches/positions must map positionally.
    for li in range(num_layers):
        d_k = np.abs(ref[2 + 3 * li].numpy() - outputs[f"new_k_cache{li}"]).max()
        d_p = np.abs(ref[4 + 3 * li].numpy() - outputs[f"new_position{li}"]).max()
        assert d_k < 0.25 and d_p < 1e-3, f"layer {li} output name mismatch (d_k={d_k}, d_p={d_p})"
    print(f"CoreML vs torch: d_transformer_out={d_out:.3e}, d_eos={d_eos:.3e}")
    print(f"Output keys: {list(outputs.keys())}")

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    args = parser.parse_args()
    convert_flowlm_step_ane(args.language, args.compute_precision, args.compute_units)
