"""Convert N=2 unrolled decoder_step + LT + audio_embed lookup to CoreML.

Pre-flight for Trial #4 (AR loop unroll): builds the smallest unroll
(N=2) and reports compile time + output spec. Use ``coreml-cli`` against
the resulting mlmodelc to measure ANE residency vs the per-iter
baseline.

Usage:
    uv run python experiments/convert_decoder_step_n2.py \\
        --output build/fused_decoder_step_n2.mlpackage

Loads weights from:
    - HF: nvidia/magpie_tts_multilingual_357m (decoder layers via NeMo)
    - ~/.cache/fluidaudio/Models/magpie-tts/constants/local_transformer/  (LT)
    - ~/.cache/fluidaudio/Models/magpie-tts/constants/audio_embedding_*.npy
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import coremltools as ct

# Local imports. Script lives in ``coreml/experiments/``; add the parent
# (``coreml/``) to sys.path so the production wrappers under
# ``coreml/traceable/`` and the experimental wrappers under
# ``coreml/experiments/traceable/`` both resolve via absolute imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traceable.traceable_decoder_step import TraceableDecoderStep
from experiments.traceable.traceable_decoder_step_n2 import (
    FusedDecoderN2,
    NUM_CODEBOOKS,
    NUM_CODES_PER_CODEBOOK,
    LOCAL_DIM,
    D_MODEL,
    DEFAULT_TOP_K,
)


def _load_lt_weights(lt_dir: str) -> Dict[str, np.ndarray]:
    needed = [
        "in_proj_weight", "in_proj_bias",
        "pos_emb",
        "norm1_weight", "norm2_weight",
        "sa_qkv_weight", "sa_o_weight",
        "ffn_conv1_weight", "ffn_conv2_weight",
    ]
    out: Dict[str, np.ndarray] = {}
    for n in needed:
        p = os.path.join(lt_dir, f"{n}.npy")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing LT weight: {p}")
        out[n] = np.load(p).astype(np.float32)
    return out


def _load_out_projections(lt_dir: str):
    weights, biases = [], []
    for i in range(NUM_CODEBOOKS):
        weights.append(np.load(os.path.join(lt_dir, f"out_proj_{i}_weight.npy")).astype(np.float32))
        biases.append(np.load(os.path.join(lt_dir, f"out_proj_{i}_bias.npy")).astype(np.float32))
    return weights, biases


def _load_audio_embeddings(const_dir: str) -> List[np.ndarray]:
    return [
        np.load(os.path.join(const_dir, f"audio_embedding_{i}.npy")).astype(np.float32)
        for i in range(NUM_CODEBOOKS)
    ]


def _project_audio_embeddings(audio_emb, in_proj_w, in_proj_b):
    out = np.empty((NUM_CODEBOOKS, NUM_CODES_PER_CODEBOOK, LOCAL_DIM), dtype=np.float32)
    for i in range(NUM_CODEBOOKS):
        out[i] = audio_emb[i] @ in_proj_w.T + in_proj_b
    return out


def convert(
    cache_dir: str,
    output_path: str,
    nemo_path: str | None = None,
    max_seq_len: int = 512,
    max_text_len: int = 256,
    top_k: int = DEFAULT_TOP_K,
):
    print("Loading MagpieTTS model (NeMo)...")
    from nemo.collections.tts.models import MagpieTTSModel
    if nemo_path:
        model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
    else:
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()

    print("Building TraceableDecoderStep from MagpieTTS weights...")
    decoder = TraceableDecoderStep.from_magpie(model)
    decoder.eval()

    cfg = model.cfg
    dec_cfg = dict(cfg.decoder)
    n_layers = dec_cfg["n_layers"]
    sa_n_heads = dec_cfg["sa_n_heads"]
    H = sa_n_heads
    D = D_MODEL // sa_n_heads

    # Free the heavy NeMo model — we have what we need.
    del model

    lt_dir = os.path.join(cache_dir, "constants", "local_transformer")
    const_dir = os.path.join(cache_dir, "constants")

    print(f"Loading LT weights from {lt_dir}")
    lt_w = _load_lt_weights(lt_dir)
    out_w, out_b = _load_out_projections(lt_dir)

    print(f"Loading audio embeddings from {const_dir}")
    audio_emb_list = _load_audio_embeddings(const_dir)
    audio_emb_full = np.stack(audio_emb_list, axis=0)
    assert audio_emb_full.shape == (NUM_CODEBOOKS, NUM_CODES_PER_CODEBOOK, D_MODEL)

    print("Pre-projecting audio embeddings through LT in_proj ...")
    proj_audio_emb = _project_audio_embeddings(
        audio_emb_list, lt_w["in_proj_weight"], lt_w["in_proj_bias"])

    print("Building FusedDecoderN2 module ...")
    module = FusedDecoderN2(
        decoder=decoder,
        lt_w=lt_w,
        out_proj_weights=out_w,
        out_proj_biases=out_b,
        proj_audio_emb=proj_audio_emb,
        audio_emb_full=audio_emb_full,
        top_k=top_k,
    )
    module.eval()

    # Example inputs for tracing.
    rng = np.random.default_rng(0)
    audio_embed = torch.randn(1, 1, D_MODEL)
    encoder_output = torch.randn(1, max_text_len, D_MODEL)
    encoder_mask = torch.ones(1, max_text_len, dtype=torch.bool)

    state_args = []
    for _ in range(n_layers):
        ck = torch.zeros(1, max_seq_len, H, D)
        cv = torch.zeros(1, max_seq_len, H, D)
        ck[:, :10, :, :] = torch.randn(1, 10, H, D) * 0.1
        cv[:, :10, :, :] = torch.randn(1, 10, H, D) * 0.1
        state_args.extend([ck, cv, torch.tensor([10.0])])

    uniforms_1 = torch.from_numpy(rng.uniform(0.0, 1.0, NUM_CODEBOOKS).astype(np.float32))
    uniforms_2 = torch.from_numpy(rng.uniform(0.0, 1.0, NUM_CODEBOOKS).astype(np.float32))
    forbid_eos_1 = torch.tensor([1.0], dtype=torch.float32)
    forbid_eos_2 = torch.tensor([1.0], dtype=torch.float32)
    temperature = torch.tensor([0.6], dtype=torch.float32)

    example_inputs = (audio_embed, encoder_output, encoder_mask,
                      *state_args,
                      uniforms_1, uniforms_2,
                      forbid_eos_1, forbid_eos_2,
                      temperature)

    print("Tracing module ...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, check_trace=False)
    print(f"  Trace took {time.time() - t0:.2f}s")

    # Eager-vs-traced sanity check.
    print("Comparing eager vs traced ...")
    with torch.no_grad():
        ref_out = module(*example_inputs)
        traced_out = traced(*example_inputs)
    if not torch.equal(ref_out[0], traced_out[0]) or not torch.equal(ref_out[1], traced_out[1]):
        print(f"  WARN: codes diverged. eager={ref_out[0].tolist()}, traced={traced_out[0].tolist()}")
    else:
        print(f"  Trace ok. codes_1={ref_out[0].tolist()}, codes_2={ref_out[1].tolist()}")

    print("Converting to CoreML (mlprogram, fp16, iOS17) ...")
    inputs = [
        ct.TensorType(name="audio_embed", shape=(1, 1, D_MODEL)),
        ct.TensorType(name="encoder_output", shape=(1, max_text_len, D_MODEL)),
        ct.TensorType(name="encoder_mask", shape=(1, max_text_len), dtype=np.bool_),
    ]
    for i in range(n_layers):
        inputs.append(ct.TensorType(name=f"cache_k{i}", shape=(1, max_seq_len, H, D)))
        inputs.append(ct.TensorType(name=f"cache_v{i}", shape=(1, max_seq_len, H, D)))
        inputs.append(ct.TensorType(name=f"position{i}", shape=(1,)))
    inputs.extend([
        ct.TensorType(name="uniforms_1", shape=(NUM_CODEBOOKS,), dtype=np.float32),
        ct.TensorType(name="uniforms_2", shape=(NUM_CODEBOOKS,), dtype=np.float32),
        ct.TensorType(name="forbid_eos_1", shape=(1,), dtype=np.float32),
        ct.TensorType(name="forbid_eos_2", shape=(1,), dtype=np.float32),
        ct.TensorType(name="temperature", shape=(1,), dtype=np.float32),
    ])

    outputs = [
        ct.TensorType(name="codes_1", dtype=np.int32),
        ct.TensorType(name="codes_2", dtype=np.int32),
    ]
    # Final state from iter 2 — same naming as decoder_step's outputs would
    # use, just chained.
    for i in range(n_layers):
        outputs.append(ct.TensorType(name=f"new_cache_k{i}"))
        outputs.append(ct.TensorType(name=f"new_cache_v{i}"))
        outputs.append(ct.TensorType(name=f"new_position{i}"))

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )
    print(f"  ct.convert took {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")

    # Print compact output spec.
    spec = mlmodel.get_spec()
    print("\n=== OUTPUTS ===")
    for o in spec.description.output:
        if o.type.HasField("multiArrayType"):
            shape = list(o.type.multiArrayType.shape)
            print(f"  {o.name}: {shape}")

    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=os.path.expanduser(
        "~/.cache/fluidaudio/Models/magpie-tts"))
    p.add_argument("--nemo-path", default=None,
                   help="Optional .nemo checkpoint; otherwise pulls from HF.")
    p.add_argument("--output", default="build/fused_decoder_step_n2.mlpackage")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-text-len", type=int, default=256)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = p.parse_args()

    convert(
        cache_dir=args.cache_dir,
        output_path=args.output,
        nemo_path=args.nemo_path,
        max_seq_len=args.max_seq_len,
        max_text_len=args.max_text_len,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
