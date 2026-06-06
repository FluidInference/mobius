"""Convert the conditioning PREFILL model to CoreML (one call for the whole
voice+text block instead of one call per token).

Drop-in faster replacement for the per-token `cond_step.mlpackage` on the
prefill path. Static T_max shape (no dynamic dims -> sidesteps the Trial 5
RangeDim scatter failure); a runtime `valid_len` value gates cache writes and
position advance so host-side padding to T_max cannot corrupt the KV cache
(the Trial 8 failure mode).

IO contract:
    inputs : conditioning [1, T_max, 1024], valid_len [1],
             cache{i} [2,1,512,16,64], position{i} [1]   (per layer)
    output : cache{i}_out, position{i}_out                (per layer)

Host change (FluidAudio): build the full conditioning embedding block
(voice tokens followed by text tokens), pad to T_max, set valid_len to the
true token count, and call once — replacing the per-token cond_step loop.
cond_step stays `.cpuAndGPU` (rank-5 KV blocks ANE; see IOS_COREML_ISSUES.md).
"""
import argparse
import torch
import numpy as np
import coremltools as ct
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))

from _language_arg import (
    add_compute_args,
    add_language_arg,
    build_output_dir,
    resolve_compute_precision,
    resolve_compute_units,
)
from traceable_cond_prefill import TraceableCondPrefill


def convert_cond_prefill(
    language: str,
    compute_precision: str = "fp16",
    compute_units: str = "CPU_AND_GPU",
    t_max: int = 256,
):
    print(f"Loading model (language={language}, T_max={t_max})...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    max_seq_len = 512
    prefill = TraceableCondPrefill.from_flowlm(model.flow_lm, max_seq_len=max_seq_len, t_max=t_max)
    prefill.eval()
    num_layers = prefill.num_layers
    print(f"num_layers={num_layers}")

    B, H, D = 1, 16, 64
    # Example inputs: partially-populated cache + mid-cache position, mirroring
    # convert_cond_step.py's trace style (avoids the all-NaN single-position
    # trace shortcut that folds constant-softmax / dead branches).
    trace_valid = 100
    conditioning = torch.randn(B, t_max, 1024)
    valid_len = torch.tensor([float(trace_valid)])
    cache = torch.zeros(2, B, max_seq_len, H, D)
    cache[:, :, :trace_valid, :, :] = torch.randn(2, B, trace_valid, H, D)
    pos = torch.tensor([0.0])

    inputs_list = [conditioning, valid_len]
    for _ in range(num_layers):
        inputs_list.append(cache.clone())
        inputs_list.append(pos.clone())
    example_inputs = tuple(inputs_list)

    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(prefill, example_inputs)

    # fp32 IO contract (see convert_cond_step.py) — internal compute stays fp16.
    print("Converting to CoreML (precision={}, IO=fp32, T_max={})...".format(compute_precision, t_max))
    ct_inputs = [
        ct.TensorType(name="conditioning", shape=(1, t_max, 1024), dtype=np.float32),
        ct.TensorType(name="valid_len", shape=(1,), dtype=np.float32),
    ]
    for i in range(num_layers):
        ct_inputs.append(ct.TensorType(name=f"cache{i}", shape=(2, 1, max_seq_len, H, D), dtype=np.float32))
        ct_inputs.append(ct.TensorType(name=f"position{i}", shape=(1,), dtype=np.float32))

    n_outputs = 2 * num_layers
    ct_outputs = [ct.TensorType(dtype=np.float32) for _ in range(n_outputs)]

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cond_prefill.mlpackage")
    mlmodel.save(output_path)
    print(f"Saved to {output_path} (precision={compute_precision})")

    # Parity test vs per-token cond_step over the valid prefix.
    print(f"\nTesting (compute_units={compute_units})...")
    from traceable_cond_step import TraceableCondStep

    step = TraceableCondStep.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    step.eval()

    N = min(141, t_max)
    cond_real = torch.randn(1, N, 1024)
    cond_pad = torch.cat([cond_real, torch.randn(1, t_max - N, 1024)], dim=1)

    seq_caches = [torch.full((2, 1, max_seq_len, H, D), float("nan")) for _ in range(num_layers)]
    seq_pos = [torch.zeros(1) for _ in range(num_layers)]
    with torch.no_grad():
        for ti in range(N):
            args = [cond_real[:, ti : ti + 1, :]]
            for li in range(num_layers):
                args.extend([seq_caches[li], seq_pos[li]])
            out = step(*args)
            for li in range(num_layers):
                seq_caches[li] = out[2 * li]
                seq_pos[li] = out[2 * li + 1]

    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))
    feed = {
        "conditioning": cond_pad.numpy().astype(np.float32),
        "valid_len": np.array([float(N)], dtype=np.float32),
    }
    # Per-token reference started from NaN caches; CoreML can't take NaN inputs
    # cleanly, so feed zeros and compare the valid prefix only (NaN->0 inside).
    for i in range(num_layers):
        feed[f"cache{i}"] = np.zeros((2, 1, max_seq_len, H, D), dtype=np.float32)
        feed[f"position{i}"] = np.array([0.0], dtype=np.float32)
    cm_out = coreml_model.predict(feed)

    # Outputs are anonymous/interleaved; rely on shape: caches are rank-5,
    # positions rank-1.
    caches_out = [v for v in cm_out.values() if getattr(v, "ndim", 0) == 5]
    max_diff = 0.0
    for li in range(min(num_layers, len(caches_out))):
        ref = seq_caches[li][:, :, :N].numpy()
        ref = np.nan_to_num(ref)
        got = np.nan_to_num(caches_out[li][:, :, :N])
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
    print(
        f"CoreML prefill vs PyTorch per-token KV max abs diff (valid prefix): "
        f"{max_diff:.3e} ({'PASS' if max_diff < 5e-2 else 'CHECK fp16 drift'})"
    )
    print("Note: cache output order is anonymous; verify per-layer mapping in the host schema discovery.")
    print("\nDone!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    parser.add_argument(
        "--t-max",
        type=int,
        default=256,
        help=(
            "Fixed conditioning block length to compile (default 256). Host "
            "pads the real voice+text block to this length and passes the true "
            "count as valid_len. Must be >= the longest voice+text conditioning."
        ),
    )
    args = parser.parse_args()
    convert_cond_prefill(args.language, args.compute_precision, args.compute_units, t_max=args.t_max)
