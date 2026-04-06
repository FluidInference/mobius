#!/usr/bin/env python3
"""Export cross-attention KV projector for Cohere Transcribe.

This pre-computes cross-attention keys and values from encoder output,
avoiding redundant computation at every decoder step.
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq


class CrossKVProjector(nn.Module):
    """Extract and project cross-attention keys/values from encoder output."""

    def __init__(self, decoder, dec_config):
        super().__init__()
        self.decoder_core = decoder._decoder

        # Store config
        self.num_layers = dec_config["num_layers"]
        self.num_heads = dec_config["num_attention_heads"]
        self.hidden_size = dec_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads

    def forward(self, encoder_hidden_states):
        """
        Project encoder hidden states to cross-attention keys and values.

        Args:
            encoder_hidden_states: (batch, seq_len, hidden_size) - encoder output

        Returns:
            cross_k: (num_layers, num_heads, seq_len, head_dim)
            cross_v: (num_layers, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = encoder_hidden_states.shape

        cross_k_list = []
        cross_v_list = []

        for layer_idx in range(self.num_layers):
            layer = self.decoder_core.layers[layer_idx]
            # second_sub_layer is the cross-attention (encoder-decoder attention)
            cross_attn = layer.second_sub_layer

            # Project to keys using key_net
            k = cross_attn.key_net(encoder_hidden_states)  # (batch, seq_len, hidden_size)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

            # Project to values using value_net
            v = cross_attn.value_net(encoder_hidden_states)  # (batch, seq_len, hidden_size)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

            cross_k_list.append(k)
            cross_v_list.append(v)

        # Stack layers: (num_layers, batch, num_heads, seq_len, head_dim)
        cross_k = torch.stack(cross_k_list, dim=0)
        cross_v = torch.stack(cross_v_list, dim=0)

        # Remove batch dimension (always 1): (num_layers, num_heads, seq_len, head_dim)
        cross_k = cross_k.squeeze(1)
        cross_v = cross_v.squeeze(1)

        return cross_k, cross_v


def export_cross_kv_projector(output_dir: Path, precision: str = "float16"):
    """Export cross-KV projector to CoreML."""
    print("="*70)
    print("Cohere Cross-KV Projector Export")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("   ✓ Loaded")

    print("\n[2/5] Extracting cross-attention projectors...")
    dec_config = model.config.transf_decoder["config_dict"]
    projector = CrossKVProjector(model.transf_decoder, dec_config)
    projector.eval()
    print("   ✓ Extracted")

    print("\n[3/5] Creating example inputs...")
    # Example encoder output shape: (1, 376, 1024)
    # 376 frames from 30s audio with 80ms hop
    example_encoder_hidden = torch.randn(1, 376, 1024)

    print("\n[4/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            projector,
            (example_encoder_hidden,),
            check_trace=False,
        )

    cross_k, cross_v = traced(example_encoder_hidden)
    print(f"   Output shapes:")
    print(f"     cross_k: {tuple(cross_k.shape)}")
    print(f"     cross_v: {tuple(cross_v.shape)}")

    print(f"\n[5/5] Converting to CoreML ({precision})...")
    inputs = [
        ct.TensorType(
            name="encoder_hidden_states",
            shape=example_encoder_hidden.shape,
            dtype=np.float32
        ),
    ]

    compute_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[
            ct.TensorType(name="cross_k"),
            ct.TensorType(name="cross_v"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=compute_precision,
    )

    output_path = output_dir / "cohere_cross_kv_projector.mlpackage"
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
    print(f"   ✓ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")

    # Print model info
    print(f"\n{'='*70}")
    print("MODEL INFO")
    print(f"{'='*70}")
    print(f"  Input: encoder_hidden_states (1, 376, 1024)")
    print(f"  Output: cross_k (8, 8, 376, 128)")
    print(f"  Output: cross_v (8, 8, 376, 128)")
    print(f"  Precision: {precision}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"\n  Usage:")
    print(f"    1. Run encoder: encoder_hidden = encoder(mel)")
    print(f"    2. Project once: cross_k, cross_v = projector(encoder_hidden)")
    print(f"    3. Reuse in all decoder steps (avoid recomputing!)")

    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        export_cross_kv_projector(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
