"""EXPERIMENTAL — DO NOT USE IN PRODUCTION.

Convert decoder step model to CoreML — STATEFUL variant (MLState).

Kept as a documented dead-end. Benchmark on Apple M2 / macOS 26.5 / 146-step
real loop showed this variant runs at ~212 ms/step vs ~96 ms/step for the
production rank-4 split-K/V graph (2.2× regression). See sibling file
``traceable/traceable_decoder_step_stateful.py`` (relative to this
script — i.e. ``coreml/experiments/traceable/...``) for full rationale.

KV caches are managed as on-device state buffers via ``ct.StateType`` instead of
being passed in/out of the graph as 36 input/output tensors per step. The model
exposes a tiny IO surface (4 inputs, 2 outputs).

Caveat: stateful CoreML graphs do not target ANE. We force CPU+GPU at runtime,
which is exactly why this variant loses for Magpie (rank-4 production already
gets 97.3% on ANE).

Usage:
    python experiments/convert_decoder_step_stateful.py [--nemo-path /path/to/model.nemo]
"""
import argparse
import os
import sys

import coremltools as ct
import numpy as np
import torch

# Script lives in ``coreml/experiments/``; add the parent (``coreml/``) to
# sys.path so the experimental traceable under
# ``coreml/experiments/traceable/`` resolves via an absolute import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.traceable.traceable_decoder_step_stateful import StatefulDecoderStep


def convert_decoder_step_stateful(nemo_path=None, max_seq_len=512, max_text_len=256,
                                  output_path="build/decoder_step_stateful.mlpackage"):
    print("Loading MagpieTTS model...")
    from nemo.collections.tts.models import MagpieTTSModel
    if nemo_path:
        model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
    else:
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()

    cfg = model.cfg
    dec_cfg = dict(cfg.decoder)
    d_model = dec_cfg["d_model"]
    n_layers = dec_cfg["n_layers"]
    sa_n_heads = dec_cfg["sa_n_heads"]
    d_head = d_model // sa_n_heads

    print("Creating stateful traceable decoder step...")
    decoder = StatefulDecoderStep.from_magpie(model)
    decoder.eval()
    decoder.reset_state()

    # Example inputs. Position is a 1-elem int32 scalar.
    B = 1
    T_enc = max_text_len

    audio_embed = torch.randn(B, 1, d_model)
    encoder_output = torch.randn(B, T_enc, d_model)
    encoder_mask = torch.ones(B, T_enc, dtype=torch.bool)
    position = torch.tensor([0], dtype=torch.int32)

    example_inputs = (audio_embed, encoder_output, encoder_mask, position)

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(decoder, example_inputs, strict=False)

    print("Converting to CoreML (stateful)...")
    inputs = [
        ct.TensorType(name="audio_embed", shape=(1, 1, d_model)),
        ct.TensorType(name="encoder_output", shape=(1, T_enc, d_model)),
        ct.TensorType(name="encoder_mask", shape=(1, T_enc), dtype=np.bool_),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
    ]

    states = []
    for i in range(n_layers):
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, max_seq_len, sa_n_heads, d_head),
                dtype=np.float16,
            ),
            name=f"k_cache_{i}",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, max_seq_len, sa_n_heads, d_head),
                dtype=np.float16,
            ),
            name=f"v_cache_{i}",
        ))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        states=states,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.macOS15,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")

    spec = mlmodel.get_spec()
    print("\n=== INPUTS ===")
    for inp in spec.description.input:
        if inp.type.HasField("multiArrayType"):
            shape = list(inp.type.multiArrayType.shape)
            print(f"  {inp.name}: {shape}")
    print("\n=== OUTPUTS ===")
    for out in spec.description.output:
        if out.type.HasField("multiArrayType"):
            shape = list(out.type.multiArrayType.shape)
            print(f"  {out.name}: {shape}")
    print("\n=== STATES ===")
    if hasattr(spec.description, "state"):
        for s in spec.description.state:
            # State features use the ``stateType`` oneof, which wraps an
            # ``arrayType`` (multiArrayType-equivalent) on the inside.
            try:
                shape = list(s.type.stateType.arrayType.shape)
                print(f"  {s.name}: {shape}")
            except Exception as exc:  # pragma: no cover - inspection only
                print(f"  {s.name}: <unable to inspect shape: {exc}>")

    print("\nTesting CoreML model with state...")
    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    state = coreml_model.make_state()

    test_inputs = {
        "audio_embed": np.random.randn(1, 1, d_model).astype(np.float32),
        "encoder_output": np.random.randn(1, T_enc, d_model).astype(np.float32),
        "encoder_mask": np.ones((1, T_enc), dtype=np.float32),
        "position": np.array([0], dtype=np.int32),
    }

    out = coreml_model.predict(test_inputs, state=state)
    print(f"Output keys: {len(out)}")
    for k, v in sorted(out.items()):
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}")
    print("Done!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-path", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--output", type=str, default="build/decoder_step_stateful.mlpackage")
    args = parser.parse_args()
    convert_decoder_step_stateful(args.nemo_path, args.max_seq_len, args.max_text_len, args.output)
