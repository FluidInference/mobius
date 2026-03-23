"""Convert text encoder to CoreML.

Usage:
    python convert/convert_text_encoder.py [--nemo-path /path/to/model.nemo]
"""
import sys
import os
import argparse

import torch
import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traceable.traceable_text_encoder import TraceableTextEncoder


def convert_text_encoder(nemo_path=None, max_text_len=256, output_path="build/text_encoder.mlpackage"):
    # Load model
    print("Loading MagpieTTS model...")
    from nemo.collections.tts.models import MagpieTTSModel
    if nemo_path:
        model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
    else:
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()

    # Create traceable encoder
    print("Creating traceable text encoder...")
    encoder = TraceableTextEncoder.from_magpie(model, include_text_embedding=True)
    encoder.eval()

    # Example inputs
    B = 1
    T = max_text_len
    text_tokens = torch.randint(0, 100, (B, T), dtype=torch.long)
    text_mask = torch.ones(B, T, dtype=torch.bool)
    text_mask[:, T // 2:] = False  # Simulate padding

    # Trace
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(encoder, (text_tokens, text_mask))

    # Verify trace output
    with torch.no_grad():
        ref_out = encoder(text_tokens, text_mask)
        traced_out = traced(text_tokens, text_mask)
        diff = (ref_out - traced_out).abs().max().item()
        print(f"Trace verification - max diff: {diff:.6e}")

    # Convert to CoreML
    print("Converting to CoreML...")
    d_model = encoder.d_model

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="text_tokens", shape=(1, T), dtype=np.int32),
            ct.TensorType(name="text_mask", shape=(1, T), dtype=np.bool_),
        ],
        outputs=[
            ct.TensorType(name="encoder_output", dtype=np.float32),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")

    # Test CoreML model
    print("\nTesting CoreML model...")
    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    test_tokens = np.random.randint(0, 100, (1, T)).astype(np.int32)
    # CoreML converts bool inputs to float32 at I/O boundary
    test_mask = np.ones((1, T), dtype=np.float32)
    test_mask[0, T // 2:] = 0.0

    out = coreml_model.predict({
        "text_tokens": test_tokens,
        "text_mask": test_mask,
    })

    enc_out = out["encoder_output"]
    print(f"Output shape: {enc_out.shape}")
    print(f"Output range: [{enc_out.min():.4f}, {enc_out.max():.4f}]")
    print("Done!")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-path", type=str, default=None)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--output", type=str, default="build/text_encoder.mlpackage")
    args = parser.parse_args()
    convert_text_encoder(args.nemo_path, args.max_text_len, args.output)
