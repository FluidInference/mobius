"""Convert decoder prefill model to CoreML.

Processes all speaker context tokens (T_ctx=110) in a single forward pass,
replacing 110 sequential decoder_step calls during prefill.

Usage:
    python convert/convert_decoder_prefill.py [--nemo-path /path/to/model.nemo]
"""
import sys
import os
import argparse
import json

import torch
import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traceable.traceable_decoder_prefill import TraceableDecoderPrefill


def convert_decoder_prefill(nemo_path=None, max_seq_len=512, max_text_len=256,
                            t_ctx=110, output_path="build/decoder_prefill.mlpackage"):
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

    # Read T_ctx from speaker_info if not specified
    constants_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants")
    si_path = os.path.join(constants_dir, "speaker_info.json")
    if os.path.exists(si_path):
        with open(si_path) as f:
            si = json.load(f)
        t_ctx_from_file = si.get("T", t_ctx)
        if t_ctx_from_file != t_ctx:
            print(f"Using T_ctx={t_ctx_from_file} from speaker_info.json (was {t_ctx})")
            t_ctx = t_ctx_from_file

    # Create traceable prefill model
    print(f"Creating traceable prefill model (T_ctx={t_ctx})...")
    prefill = TraceableDecoderPrefill.from_magpie(model, t_ctx=t_ctx)
    prefill.eval()

    # Example inputs
    B = 1
    audio_embed = torch.randn(B, t_ctx, d_model)
    encoder_output = torch.randn(B, max_text_len, d_model)
    encoder_mask = torch.ones(B, max_text_len, dtype=torch.bool)

    # Trace
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(prefill, (audio_embed, encoder_output, encoder_mask))

    # Verify trace output
    with torch.no_grad():
        outputs = traced(audio_embed, encoder_output, encoder_mask)
    print(f"Trace outputs: {len(outputs)} caches")
    for i, cache in enumerate(outputs):
        print(f"  cache{i}: {cache.shape}")

    # Convert to CoreML
    print("Converting to CoreML...")
    inputs = [
        ct.TensorType(name="audio_embed", shape=(1, t_ctx, d_model)),
        ct.TensorType(name="encoder_output", shape=(1, max_text_len, d_model)),
        ct.TensorType(name="encoder_mask", shape=(1, max_text_len), dtype=np.bool_),
    ]

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
    print(f"\n=== OUTPUTS ({len(spec.description.output)}) ===")
    for out in spec.description.output:
        if out.type.HasField("multiArrayType"):
            shape = list(out.type.multiArrayType.shape)
            print(f"  {out.name}: {shape}")

    # Quick test
    print("\nTesting CoreML model...")
    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    test_inputs = {
        "audio_embed": np.random.randn(1, t_ctx, d_model).astype(np.float32),
        "encoder_output": np.random.randn(1, max_text_len, d_model).astype(np.float32),
        "encoder_mask": np.ones((1, max_text_len), dtype=np.float32),
    }

    out = coreml_model.predict(test_inputs)
    print(f"Output keys: {len(out)}")
    for k, v in sorted(out.items()):
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}")

    # Verify cache shapes match decoder_step format
    expected_cache_shape = (2, 1, max_seq_len, sa_n_heads, d_head)
    for k, v in out.items():
        if isinstance(v, np.ndarray) and len(v.shape) == 5:
            assert v.shape == expected_cache_shape, \
                f"Cache {k} shape {v.shape} != expected {expected_cache_shape}"
    print(f"All cache shapes match decoder_step format: {expected_cache_shape}")
    print("Done!")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-path", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--t-ctx", type=int, default=110)
    parser.add_argument("--output", type=str, default="build/decoder_prefill.mlpackage")
    args = parser.parse_args()
    convert_decoder_prefill(args.nemo_path, args.max_seq_len, args.max_text_len,
                            args.t_ctx, args.output)
