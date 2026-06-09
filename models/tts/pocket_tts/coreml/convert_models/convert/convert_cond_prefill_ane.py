"""Convert the ANE-eligible cond_prefill (rank-4, scatter-free) to CoreML.

Emits `cond_prefill_ane.mlpackage`. See `traceable_cond_prefill_ane.py`
(Trial 20) — the Trial 19 treatment applied to the conditioning prefill,
the last GPU model in the synthesis path.
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
from traceable_cond_prefill_ane import TraceableCondPrefillANE


def convert_cond_prefill_ane(
    language: str,
    compute_precision: str = "fp16",
    compute_units: str = "ALL",
    t_max: int = 256,
):
    print(f"Loading model (language={language})...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    print("Creating ANE prefill model...")
    max_seq_len = 512
    prefill = TraceableCondPrefillANE.from_flowlm(
        model.flow_lm, max_seq_len=max_seq_len, t_max=t_max)
    prefill.eval()
    num_layers = prefill.num_layers
    H = prefill.num_heads
    D = prefill.head_dim
    print(f"num_layers={num_layers}, max_seq_len={max_seq_len}, t_max={t_max}")

    print("Creating example inputs...")
    trace_inputs = [torch.randn(1, t_max, 1024), torch.tensor([141.0])]
    for _ in range(num_layers):
        trace_inputs.extend([
            torch.zeros(1, max_seq_len, H, D),
            torch.zeros(1, max_seq_len, H, D),
            torch.zeros(1),
        ])

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(prefill, tuple(trace_inputs))

    # fp32 IO contract — same reasoning as the other converters: internal
    # ops run fp16, boundary casts let Swift use float32 MLMultiArrays.
    print("Converting to CoreML (precision={}, IO=fp32)...".format(compute_precision))
    inputs = [
        ct.TensorType(name="conditioning", shape=(1, t_max, 1024), dtype=np.float32),
        ct.TensorType(name="valid_len", shape=(1,), dtype=np.float32),
    ]
    for i in range(num_layers):
        inputs.append(
            ct.TensorType(name=f"k_cache{i}", shape=(1, max_seq_len, H, D), dtype=np.float32))
        inputs.append(
            ct.TensorType(name=f"v_cache{i}", shape=(1, max_seq_len, H, D), dtype=np.float32))
        inputs.append(ct.TensorType(name=f"position{i}", shape=(1,), dtype=np.float32))

    n_outputs = 3 * num_layers  # (new_k, new_v, new_position) per layer
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
    output_path = os.path.join(output_dir, "cond_prefill_ane.mlpackage")
    print(f"Saving to {output_path} (precision={compute_precision})...")
    mlmodel.save(output_path)

    # Smoke test + fp16-vs-fp32 parity on the valid prefix.
    print(f"\nTesting CoreML model (compute_units={compute_units})...")
    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))

    rng = np.random.default_rng(0)
    N = 141
    feed = {"conditioning": rng.standard_normal((1, t_max, 1024), dtype=np.float32),
            "valid_len": np.array([float(N)], dtype=np.float32)}
    torch_inputs = [torch.from_numpy(feed["conditioning"]), torch.tensor([float(N)])]
    for i in range(num_layers):
        k = np.zeros((1, max_seq_len, H, D), dtype=np.float32)
        v = np.zeros((1, max_seq_len, H, D), dtype=np.float32)
        feed[f"k_cache{i}"] = k
        feed[f"v_cache{i}"] = v
        feed[f"position{i}"] = np.array([0.0], dtype=np.float32)
        torch_inputs.extend([torch.from_numpy(k), torch.from_numpy(v), torch.zeros(1)])

    outputs = coreml_model.predict(feed)
    with torch.no_grad():
        ref = prefill(*torch_inputs)

    # Anonymous output names; the K/V caches are the only [1, L, H, D]
    # tensors. Compare the worst-case diff over all of them on [0, N).
    cache_outs = [v for v in outputs.values()
                  if isinstance(v, np.ndarray) and v.shape == (1, max_seq_len, H, D)]
    assert len(cache_outs) == 2 * num_layers, f"got {len(cache_outs)} cache outputs"
    ref_caches = [ref[3 * li + j].numpy() for li in range(num_layers) for j in (0, 1)]
    worst = 0.0
    for got in cache_outs:
        best = min(np.abs(r[:, :N] - got[:, :N]).max() for r in ref_caches)
        worst = max(worst, best)
    print(f"CoreML vs torch, worst valid-prefix cache diff: {worst:.3e}")
    print(f"Output keys: {list(outputs.keys())}")

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    parser.add_argument("--t-max", type=int, default=256, help="fixed conditioning block length")
    args = parser.parse_args()
    convert_cond_prefill_ane(args.language, args.compute_precision, args.compute_units, args.t_max)
