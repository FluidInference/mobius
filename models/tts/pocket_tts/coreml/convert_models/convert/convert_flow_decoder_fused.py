"""Convert the fused (full N-step Euler) flow decoder to CoreML.

Drop-in faster replacement for `flow_decoder.mlpackage`: instead of the host
calling an N=8 single-step decoder 8× per audio frame, this artifact runs the
whole LSD integration in one `predict()`. For a 42-frame utterance that is
42 dispatches instead of 336 — see TRIALS.md / IOS_COREML_ISSUES.md for why
the tiny per-step kernel never amortized ANE residency.

IO contract:
    inputs : transformer_out [1, 1024], latent_init [1, 32]
    output : latent_final    [1, 32]

Host change required (FluidAudio StreamingGenerator): replace the per-frame
8-iteration Euler loop + 8 predict() calls with a single predict() that passes
z_0 as `latent_init` and reads `latent_final`. The `s`/`t` inputs disappear
(baked in at conversion as i/N, (i+1)/N for the chosen --num-steps).
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
from traceable_flow_decoder_fused import TraceableFlowDecoderFused


def convert_flow_decoder_fused(
    language: str,
    compute_precision: str = "fp16",
    compute_units: str = "ALL",
    num_steps: int = 8,
):
    print(f"Loading model (language={language}, num_steps={num_steps})...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(language=language, lsd_decode_steps=num_steps)
    model.eval()

    print("Creating fused traceable flow decoder...")
    flow_decoder = TraceableFlowDecoderFused.from_flowlm(model.flow_lm, num_steps=num_steps)
    flow_decoder.eval()

    print("Creating example inputs...")
    transformer_out = torch.randn(1, 1024)
    latent_init = torch.randn(1, 32)

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(flow_decoder, (transformer_out, latent_init))

    # NOTE: Force fp32 IO contract (see convert_flow_decoder_v2.py). Internal
    # compute stays fp16 for the perf win; coremltools inserts fp16↔fp32 casts
    # only at the two IO tensors so Swift can drive float32 MLMultiArrays.
    # Fusion shrinks the IO surface from 4 inputs / 1 output × 8 calls to
    # 2 inputs / 1 output × 1 call, so the cast overhead drops ~12×.
    print("Converting to CoreML (precision={}, IO=fp32)...".format(compute_precision))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="transformer_out", shape=(1, 1024), dtype=np.float32),
            ct.TensorType(name="latent_init", shape=(1, 32), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="latent_final", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "flow_decoder_fused.mlpackage")
    print(f"Saving to {output_path} (precision={compute_precision}, num_steps={num_steps})...")
    mlmodel.save(output_path)

    # Test: parity against the host-side Euler loop using the single-step decoder.
    print(f"\nTesting CoreML model (compute_units={compute_units})...")
    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))

    test_transformer = np.random.randn(1, 1024).astype(np.float32)
    test_z0 = np.random.randn(1, 32).astype(np.float32)
    out = coreml_model.predict({
        "transformer_out": test_transformer,
        "latent_init": test_z0,
    })
    latent_final = list(out.values())[0]
    print(f"latent_final shape: {latent_final.shape}, range [{latent_final.min():.4f}, {latent_final.max():.4f}]")

    # PyTorch host-loop reference for parity.
    from traceable_flow_decoder import TraceableFlowDecoder
    single = TraceableFlowDecoder.from_flowlm(model.flow_lm)
    single.eval()
    dt = 1.0 / num_steps
    latent = torch.from_numpy(test_z0)
    cond = torch.from_numpy(test_transformer)
    with torch.no_grad():
        for step in range(num_steps):
            s = torch.tensor([[step * dt]])
            t = torch.tensor([[(step + 1) * dt]])
            velocity = single(cond, latent, s, t)
            latent = latent + velocity * dt
    ref = latent.numpy()
    max_diff = np.abs(ref - latent_final).max()
    print(f"CoreML-fused vs PyTorch host-loop max abs diff: {max_diff:.3e} "
          f"({'PASS' if max_diff < 1e-2 else 'CHECK fp16 drift'})")

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
            "(default 8, matching the shipped lsd_decode_steps). Lower values "
            "(e.g. 4) trade a small quality delta for ~2× fewer internal "
            "flow_net evals; verify with Whisper before shipping."
        ),
    )
    args = parser.parse_args()
    convert_flow_decoder_fused(
        args.language, args.compute_precision, args.compute_units, num_steps=args.num_steps
    )
