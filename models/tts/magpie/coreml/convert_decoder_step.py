"""Convert decoder step model to CoreML.

Usage:
    python convert/convert_decoder_step.py [--nemo-path /path/to/model.nemo]
"""
import sys
import os
import argparse

import torch
import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traceable.traceable_decoder_step import TraceableDecoderStep


def convert_decoder_step(nemo_path=None, max_seq_len=512, max_text_len=256,
                         output_path="build/decoder_step.mlpackage"):
    # Load model
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

    # Create traceable decoder step
    print("Creating traceable decoder step...")
    decoder = TraceableDecoderStep.from_magpie(model)
    decoder.eval()

    # Example inputs
    B = 1
    T_enc = max_text_len
    H = sa_n_heads
    D = d_head

    audio_embed = torch.randn(B, 1, d_model)
    encoder_output = torch.randn(B, T_enc, d_model)
    encoder_mask = torch.ones(B, T_enc, dtype=torch.bool)

    # Flat cache and position args
    caches = []
    positions = []
    for i in range(n_layers):
        cache = torch.zeros(2, B, max_seq_len, H, D)
        # Simulate some prefilled context
        cache[:, :, :10, :, :] = torch.randn(2, B, 10, H, D) * 0.1
        caches.append(cache)
        positions.append(torch.tensor([10.0]))

    # Build flat argument tuple
    example_inputs = (audio_embed, encoder_output, encoder_mask)
    for i in range(n_layers):
        example_inputs = example_inputs + (caches[i], positions[i])

    # Trace
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(decoder, example_inputs)

    # Convert to CoreML
    print("Converting to CoreML...")
    inputs = [
        ct.TensorType(name="audio_embed", shape=(1, 1, d_model)),
        ct.TensorType(name="encoder_output", shape=(1, T_enc, d_model)),
        ct.TensorType(name="encoder_mask", shape=(1, T_enc), dtype=np.bool_),
    ]
    for i in range(n_layers):
        inputs.append(ct.TensorType(name=f"cache{i}", shape=(2, 1, max_seq_len, H, D)))
        inputs.append(ct.TensorType(name=f"position{i}", shape=(1,)))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")

    # Print output spec
    spec = mlmodel.get_spec()
    print("\n=== OUTPUTS ===")
    for out in spec.description.output:
        if out.type.HasField("multiArrayType"):
            shape = list(out.type.multiArrayType.shape)
            print(f"  {out.name}: {shape}")

    # Quick test
    print("\nTesting CoreML model...")
    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    test_inputs = {
        "audio_embed": np.random.randn(1, 1, d_model).astype(np.float32),
        "encoder_output": np.random.randn(1, T_enc, d_model).astype(np.float32),
        "encoder_mask": np.ones((1, T_enc), dtype=np.float32),
    }
    for i in range(n_layers):
        test_inputs[f"cache{i}"] = np.zeros((2, 1, max_seq_len, H, D), dtype=np.float32)
        test_inputs[f"position{i}"] = np.array([0.0], dtype=np.float32)

    out = coreml_model.predict(test_inputs)
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
    parser.add_argument("--output", type=str, default="build/decoder_step.mlpackage")
    args = parser.parse_args()
    convert_decoder_step(args.nemo_path, args.max_seq_len, args.max_text_len, args.output)
