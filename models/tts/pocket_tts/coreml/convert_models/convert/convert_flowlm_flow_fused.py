"""Convert the fused FlowLM step + flow decoder to CoreML.

Emits `flowlm_flow_fused.mlpackage`: the Trial 19 ANE step model and the
Trial 16 fused (N-step Euler) flow decoder concatenated into ONE graph, so
the host makes one ANE dispatch per audio frame instead of two and the
[1, 1, 1024] `transformer_out` boundary tensor disappears (stays internal
fp16). See `traceable_flowlm_flow_fused.py`.

Verify placement afterwards with:
    uv run coreml-cli build/<language>/flowlm_flow_fused.mlpackage --fallback
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
from traceable_flowlm_flow_fused import TraceableFlowLMFlowFused


def convert_flowlm_flow_fused(
    language: str,
    compute_precision: str = "fp16",
    compute_units: str = "ALL",
    num_steps: int = 8,
):
    print(f"Loading model (language={language}, num_steps={num_steps})...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(language=language, lsd_decode_steps=num_steps)
    model.eval()

    print("Creating fused flowlm+flowdec model...")
    max_seq_len = 512
    fused_model = TraceableFlowLMFlowFused.from_flowlm(
        model.flow_lm, max_seq_len=max_seq_len, num_steps=num_steps
    )
    fused_model.eval()
    num_layers = fused_model.num_layers
    H = fused_model.flowlm_step.num_heads
    D = fused_model.flowlm_step.head_dim
    print(f"num_layers={num_layers}, max_seq_len={max_seq_len}, num_steps={num_steps}")

    print("Creating example inputs...")
    sequence = torch.randn(1, 1, 32)
    bos_emb = model.flow_lm.bos_emb.data
    latent_init = torch.randn(1, 32)

    prefix_len = 136
    trace_inputs = [sequence, bos_emb, latent_init]
    for _ in range(num_layers):
        k_cache = torch.zeros(1, max_seq_len, H, D)
        v_cache = torch.zeros(1, max_seq_len, H, D)
        k_cache[:, :prefix_len] = torch.randn(1, prefix_len, H, D)
        v_cache[:, :prefix_len] = torch.randn(1, prefix_len, H, D)
        trace_inputs.extend([k_cache, v_cache, torch.tensor([float(prefix_len)])])

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(fused_model, tuple(trace_inputs))

    # NOTE: fp32 IO contract (`dtype=np.float32` on every TensorType + fp32
    # outputs) — same reasoning as convert_flowlm_step_ane.py: internal ops
    # run fp16, coremltools inserts boundary casts, and Swift drives the model
    # with `MLMultiArrayDataType.float32` buffers (the macOS MLE5 binder
    # rejects fp16 MLMultiArrays). Fusion removes the [1, 1, 1024]
    # transformer_out from the IO surface entirely.
    print("Converting to CoreML (precision={}, IO=fp32)...".format(compute_precision))
    inputs = [
        ct.TensorType(name="sequence", shape=(1, 1, 32), dtype=np.float32),
        ct.TensorType(name="bos_emb", shape=(32,), dtype=np.float32),
        ct.TensorType(name="latent_init", shape=(1, 32), dtype=np.float32),
    ]
    for i in range(num_layers):
        inputs.append(
            ct.TensorType(name=f"k_cache{i}", shape=(1, max_seq_len, H, D), dtype=np.float32))
        inputs.append(
            ct.TensorType(name=f"v_cache{i}", shape=(1, max_seq_len, H, D), dtype=np.float32))
        inputs.append(ct.TensorType(name=f"position{i}", shape=(1,), dtype=np.float32))

    # Outputs: (latent_final, is_eos, then per layer (new_k, new_v, new_position)).
    n_outputs = 2 + 3 * num_layers
    ct_outputs = [ct.TensorType(dtype=np.float32) for _ in range(n_outputs)]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "flowlm_flow_fused.mlpackage")
    print(f"Saving to {output_path} (precision={compute_precision}, num_steps={num_steps})...")
    mlmodel.save(output_path)

    # Smoke test + parity vs the traced torch module on the same inputs.
    print(f"\nTesting CoreML model (compute_units={compute_units})...")
    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))

    feed = {"sequence": np.random.randn(1, 1, 32).astype(np.float32),
            "bos_emb": bos_emb.numpy().astype(np.float32),
            "latent_init": np.random.randn(1, 32).astype(np.float32)}
    torch_inputs = [
        torch.from_numpy(feed["sequence"]), bos_emb, torch.from_numpy(feed["latent_init"])]
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
        ref = fused_model(*torch_inputs)

    # Output names are anonymous (var_*); match by shape: latent_final is the
    # only [1, 32], is_eos the only [1, 1, 1].
    by_shape = {tuple(v.shape): v for v in outputs.values() if isinstance(v, np.ndarray)}
    got_latent = by_shape.get((1, 32))
    got_eos = by_shape.get((1, 1, 1))
    assert got_latent is not None and got_eos is not None, f"output shapes: {list(by_shape)}"
    d_latent = np.abs(ref[0].numpy() - got_latent).max()
    d_eos = np.abs(ref[1].numpy() - got_eos).max()
    print(f"CoreML vs torch: d_latent_final={d_latent:.3e}, d_eos={d_eos:.3e}")
    print(f"Output keys: {list(outputs.keys())}")

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=8,
        help=(
            "Number of LSD Euler integration steps to fuse into the graph "
            "(default 8, matching the shipped lsd_decode_steps). Must match "
            "the host's expectation; verify lower counts with Whisper before "
            "shipping."
        ),
    )
    args = parser.parse_args()
    convert_flowlm_flow_fused(
        args.language, args.compute_precision, args.compute_units, num_steps=args.num_steps
    )
