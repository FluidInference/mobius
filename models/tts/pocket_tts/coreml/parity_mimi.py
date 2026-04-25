"""Numerical parity: upstream TraceableMimiDecoder vs CoreML mimi_decoder.

cond_step, flowlm_step, and flow_decoder all match upstream PyTorch to
numerical noise (<=1e-4). The remaining suspects are:

  (a) the per-language mimi_decoder.mlpackage conversion itself, or
  (b) how generate_coreml_v4.py wires mimi state inputs/outputs between
      autoregressive frames.

This script addresses (a): feed a fixed sequence of latents through BOTH
upstream TraceableMimiDecoder (same wrapper that was traced for CoreML)
and our CoreML mimi_decoder.mlpackage, seeded from the same zero state,
and compare audio frames + updated state tensors per frame.

If they diverge bit-for-bit between frame 0 (single latent) and frame N,
the CoreML conversion is fine and the bug is in our loop wiring (b).
If frame 0 already differs beyond fp16 precision, the conversion itself
drifted and we'll look at the trace path.

Usage:
  uv run python parity_mimi.py --language german --num-frames 4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "convert_models", "traceable"))


def _upstream_mimi(language: str, latents: np.ndarray) -> tuple[list[np.ndarray], list[tuple[np.ndarray, ...]]]:
    from pocket_tts.models.tts_model import TTSModel
    from traceable_mimi_decoder import TraceableMimiDecoder, MIMI_STATE_SPEC

    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()
    traceable = TraceableMimiDecoder.from_tts_model(model)
    traceable.eval()

    # Zero initial state per MIMI_STATE_SPEC
    state_tensors = [torch.zeros(*shape, dtype=torch.float32) for _, shape in MIMI_STATE_SPEC]

    audios = []
    states_per_frame = []
    for t in range(latents.shape[0]):
        latent_t = torch.from_numpy(latents[t:t+1]).to(torch.float32)  # [1, 32]
        with torch.no_grad():
            outs = traceable(latent_t, *state_tensors)
        audio = outs[0].detach().cpu().numpy()  # [1, 1, 1920]
        new_states = tuple(s.detach().cpu().numpy() for s in outs[1:])
        audios.append(audio)
        states_per_frame.append(new_states)
        state_tensors = [torch.from_numpy(s) for s in new_states]
    return audios, states_per_frame


def _coreml_mimi(language: str, latents: np.ndarray) -> tuple[list[np.ndarray], list[tuple[np.ndarray, ...]], list[str], list[str]]:
    import coremltools as ct
    from traceable_mimi_decoder import MIMI_STATE_SPEC

    model_path = os.path.join(_SCRIPT_DIR, "build", language, "mimi_decoder.mlpackage")
    coreml_mimi = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = coreml_mimi.get_spec()

    # Walk inputs in spec order, skipping 'latent'. Initialize to zeros of
    # the shape the spec declares.
    mimi_input_order = []
    coreml_state = {}
    for inp in spec.description.input:
        if inp.name == "latent":
            continue
        shape = tuple(int(d) for d in inp.type.multiArrayType.shape)
        coreml_state[inp.name] = np.zeros(shape, dtype=np.float32)
        mimi_input_order.append(inp.name)
    output_names = [out.name for out in spec.description.output]

    audios = []
    states_per_frame = []
    for t in range(latents.shape[0]):
        inp = {"latent": latents[t:t+1].astype(np.float32), **coreml_state}
        out = coreml_mimi.predict(inp)
        audio = out[output_names[0]]
        new_state_tuple = tuple(out[n] for n in output_names[1:])
        audios.append(audio)
        states_per_frame.append(new_state_tuple)
        # Pair positionally: input_order[i] <- output[1+i]
        for state_name, out_name in zip(mimi_input_order, output_names[1:]):
            coreml_state[state_name] = out[out_name]
    return audios, states_per_frame, mimi_input_order, output_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Fixed deterministic latents (bit-identical across both runs).
    latents = np.random.randn(args.num_frames, 32).astype(np.float32) * (0.7 ** 0.5)
    print(f"Latents: shape={latents.shape} range=[{latents.min():.4f}, {latents.max():.4f}]")

    print("\n[upstream] running TraceableMimiDecoder...")
    up_audios, up_states = _upstream_mimi(args.language, latents)

    print("\n[coreml] running mimi_decoder.mlpackage...")
    cm_audios, cm_states, input_names, output_names = _coreml_mimi(args.language, latents)

    # Report input/output name mapping for diagnostic.
    print(f"\n[coreml] input_order (non-latent): {input_names[:3]} ... "
          f"({len(input_names)} states)")
    print(f"[coreml] output_names:            {output_names[:3]} ... "
          f"({len(output_names)} outputs)")

    # Re-load SPEC order for comparison.
    from traceable_mimi_decoder import MIMI_STATE_SPEC
    spec_names = [n for n, _ in MIMI_STATE_SPEC]

    # Check if CoreML input order matches SPEC order (positionally) — if not,
    # the `zip(input_order, output_names[1:])` pairing in generate_v4 is wrong.
    if input_names == spec_names:
        print("[coreml] input order matches MIMI_STATE_SPEC order (OK)")
    else:
        print("[coreml] *** input order does NOT match MIMI_STATE_SPEC ***")
        for i, (cm, sp_name) in enumerate(zip(input_names, spec_names)):
            marker = "OK" if cm == sp_name else "MISMATCH"
            print(f"  [{i:2d}] coreml='{cm}' vs spec='{sp_name}' {marker}")

    print("\nPer-frame audio diff (upstream TraceableMimi vs CoreML):")
    for i, (u, c) in enumerate(zip(up_audios, cm_audios)):
        diff = np.abs(u - c)
        rel = diff.mean() / (np.abs(u).mean() + 1e-9)
        print(
            f"  frame {i}: u|mean={np.abs(u).mean():.5f} "
            f"c|mean={np.abs(c).mean():.5f} "
            f"abs_max={diff.max():.5f} abs_mean={diff.mean():.5f} "
            f"rel={rel:.5f}"
        )

    # Per-frame, per-state diff on the LAST frame (accumulates any drift).
    print("\nLast-frame per-state diff (upstream vs CoreML):")
    u_state = up_states[-1]
    c_state = cm_states[-1]
    for i, (spec_name, _) in enumerate(MIMI_STATE_SPEC):
        u_arr = u_state[i]
        c_arr = c_state[i]
        if u_arr.shape != c_arr.shape:
            print(f"  [{i:2d}] {spec_name}: SHAPE MISMATCH u={u_arr.shape} c={c_arr.shape}")
            continue
        diff = np.abs(u_arr - c_arr)
        rel = diff.mean() / (np.abs(u_arr).mean() + 1e-9)
        print(
            f"  [{i:2d}] {spec_name:24s} "
            f"u|mean={np.abs(u_arr).mean():.5f} "
            f"abs_max={diff.max():.5f} abs_mean={diff.mean():.5f} "
            f"rel={rel:.5f}"
        )


if __name__ == "__main__":
    main()
