"""Convert cond_step traced with a chunk size N>1 (instead of T=1).

The existing TraceableCondStep already handles arbitrary T inside its forward
pass; the canonical convert_cond_step.py just hardcodes T=1 at trace time.
This script retraces with T=N so a single CoreML call advances the KV cache
by N positions, replacing N round-trips.

Per CONVERSION.md the conditioning prefill is 141 tokens (125 voice + 16 text).
Two natural targets:
    --chunk 16  → fold all text in one call (~9x reduction for the text part)
    --chunk 125 → fold the entire voice block in one call
    --chunk 141 → fold the whole prefill into one call

Usage:
    uv run --no-project --python 3.10 \
        --with "pocket-tts>=1.0.3" --with "coremltools>=8.0" \
        --with "torch>=2.5.0" --with "numpy>=2" \
        --with "safetensors>=0.4.0" --with "sentencepiece>=0.2.1" \
        --with "scipy>=1.5.0" --with "huggingface_hub>=0.10" \
        --with "einops>=0.4.0" \
        python convert_cond_step_chunked.py --chunk 16
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
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))

from traceable_cond_step import TraceableCondStep  # noqa: E402

MAX_SEQ_LEN = 512
NUM_LAYERS = 6


def make_inputs(chunk: int):
    conditioning = torch.randn(1, chunk, 1024)
    cache = torch.full((2, 1, MAX_SEQ_LEN, 16, 64), float("nan"))
    pos = torch.zeros(1)
    inputs = [conditioning]
    for _ in range(NUM_LAYERS):
        inputs.extend([cache.clone(), pos.clone()])
    return tuple(inputs)


def reference_per_token_loop(cond_step, conditioning_chunk, cache_init, pos_init):
    """Run the per-token (T=1) reference loop on the same chunk, return final state."""
    caches = [cache_init.clone() for _ in range(NUM_LAYERS)]
    positions = [pos_init.clone() for _ in range(NUM_LAYERS)]
    T = conditioning_chunk.shape[1]
    with torch.no_grad():
        for t in range(T):
            tok = conditioning_chunk[:, t : t + 1, :]
            args = [tok]
            for li in range(NUM_LAYERS):
                args.extend([caches[li], positions[li]])
            out = cond_step(*args)
            for li in range(NUM_LAYERS):
                caches[li] = out[2 * li]
                positions[li] = out[2 * li + 1]
    return caches, positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"cond_step_chunk{args.chunk}.mlpackage"

    print("Loading model...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    cond_step = TraceableCondStep.from_flowlm(model.flow_lm, max_seq_len=MAX_SEQ_LEN)
    cond_step.eval()

    # ---- PyTorch parity: chunk-of-N call vs N per-token calls ----
    print(f"\n[PyTorch] N={args.chunk} chunk vs {args.chunk} per-token calls...")
    torch.manual_seed(0)
    cond_chunk = torch.randn(1, args.chunk, 1024)
    cache_init = torch.full((2, 1, MAX_SEQ_LEN, 16, 64), float("nan"))
    pos_init = torch.zeros(1)

    with torch.no_grad():
        chunk_args = [cond_chunk]
        for _ in range(NUM_LAYERS):
            chunk_args.extend([cache_init.clone(), pos_init.clone()])
        chunk_out = cond_step(*chunk_args)

    ref_caches, ref_positions = reference_per_token_loop(
        cond_step, cond_chunk, cache_init, pos_init
    )

    # Compare layer 5 final cache (last layer most exercises depth)
    chunk_cache5 = chunk_out[2 * 5]
    chunk_pos5 = chunk_out[2 * 5 + 1]
    ref_cache5 = ref_caches[5]
    ref_pos5 = ref_positions[5]

    # NaN-safe diff: only compare positions that have been written
    valid = ~torch.isnan(ref_cache5) & ~torch.isnan(chunk_cache5)
    pt_diff = (chunk_cache5[valid] - ref_cache5[valid]).abs().max().item()
    print(f"  layer-5 cache max abs diff: {pt_diff:.3e}")
    print(f"  layer-5 position chunk={chunk_pos5.item()} ref={ref_pos5.item()}")

    # ---- Trace + convert ----
    print(f"\nTracing chunk={args.chunk}...")
    example_inputs = make_inputs(args.chunk)
    with torch.no_grad():
        traced = torch.jit.trace(cond_step, example_inputs)

    print("Converting to CoreML (FP32, iOS17)...")
    coreml_inputs = [ct.TensorType(name="conditioning", shape=(1, args.chunk, 1024))]
    for i in range(NUM_LAYERS):
        coreml_inputs.append(
            ct.TensorType(name=f"cache{i}", shape=(2, 1, MAX_SEQ_LEN, 16, 64))
        )
        coreml_inputs.append(ct.TensorType(name=f"position{i}", shape=(1,)))

    mlmodel = ct.convert(
        traced,
        inputs=coreml_inputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    print(f"Saving to {out_path}...")
    mlmodel.save(out_path)

    # ---- CoreML parity check ----
    print("\nLoading CoreML model on CPU+GPU...")
    coreml_model = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

    test_inputs = {"conditioning": cond_chunk.numpy().astype(np.float32)}
    for i in range(NUM_LAYERS):
        test_inputs[f"cache{i}"] = np.where(
            np.isnan(cache_init.numpy()), 0.0, cache_init.numpy()
        ).astype(np.float32)
        test_inputs[f"position{i}"] = np.array([0.0], dtype=np.float32)

    print("Predicting...")
    out = coreml_model.predict(test_inputs)
    out_keys = list(out.keys())
    print(f"  {len(out_keys)} outputs returned")

    print("\n=== SUMMARY ===")
    print(f"  PyTorch: chunk-{args.chunk} vs per-token loop max abs diff = {pt_diff:.3e}")
    print(f"  Position advanced by: {chunk_pos5.item()} (expected {args.chunk}.0)")
    print("\nDone!")


if __name__ == "__main__":
    main()
