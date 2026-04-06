#!/usr/bin/env python3
"""Export Cohere decoder with EXTERNAL cache management.

This allows the caller (Swift) to manage KV cache, enabling:
- Neural Network format (iOS 14+, not iOS 18+)
- .mlmodelc compilation
- O(n) complexity (efficient)
- No CoreML State API dependency
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq


class ExternalCacheDecoderWrapper(nn.Module):
    """
    Decoder with external KV cache management.

    Inputs:
        - input_id: Single token [1, 1]
        - encoder_hidden_states: Encoder output [1, 438, 1024]
        - position_id: Current position [1, 1]
        - past_key_values: KV cache from previous step (16 tensors: 8 layers × 2 per layer)

    Outputs:
        - logits: Token probabilities [1, 16384]
        - present_key_values: Updated KV cache for next step (16 tensors)
    """

    def __init__(self, full_model, max_seq_len=108):
        super().__init__()
        self.decoder = full_model.transf_decoder
        self.log_softmax = full_model.log_softmax
        self.max_seq_len = max_seq_len
        self.num_layers = 8
        self.num_heads = 8
        self.head_dim = 128

    def forward(
        self,
        input_id,  # [1, 1]
        encoder_hidden_states,  # [1, 438, 1024]
        position_id,  # [1, 1]
        # Input KV cache: 8 layers × (key, value)
        cache_k_0, cache_v_0,
        cache_k_1, cache_v_1,
        cache_k_2, cache_v_2,
        cache_k_3, cache_v_3,
        cache_k_4, cache_v_4,
        cache_k_5, cache_v_5,
        cache_k_6, cache_v_6,
        cache_k_7, cache_v_7,
    ):
        """
        Single-token decoding with external cache.

        The caller (Swift) manages the KV cache and passes it in/out.
        """
        device = input_id.device

        # Organize input cache
        past_key_values = [
            (cache_k_0.clone(), cache_v_0.clone()),  # Clone to avoid in-place ops
            (cache_k_1.clone(), cache_v_1.clone()),
            (cache_k_2.clone(), cache_v_2.clone()),
            (cache_k_3.clone(), cache_v_3.clone()),
            (cache_k_4.clone(), cache_v_4.clone()),
            (cache_k_5.clone(), cache_v_5.clone()),
            (cache_k_6.clone(), cache_v_6.clone()),
            (cache_k_7.clone(), cache_v_7.clone()),
        ]

        # Current sequence length (from position_id)
        # Use a fixed approach to avoid tracing issues
        seq_len = position_id.shape[1] + past_key_values[0][0].sum(dim=[2, 3]).clamp(0, 1).sum().long()

        # Cache position - use position_id directly
        cache_position = position_id.squeeze(0)

        # Cross attention mask (all encoder positions valid)
        cross_mask = torch.ones(1, encoder_hidden_states.shape[1], device=device)

        # Decoder forward
        decoder_outputs, new_key_values = self.decoder(
            input_ids=input_id,
            positions=position_id,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=None,  # Causal mask handled internally
            cross_attention_mask=cross_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            kv_seq_len=None,  # Let decoder figure it out
        )

        # Get logits for the token
        logits = self.log_softmax(decoder_outputs[:, -1:, :]).squeeze(1)

        # Pack outputs - return NEW cache tensors (not reusing inputs)
        # This avoids CoreML name collision
        out_k_0, out_v_0 = new_key_values[0]
        out_k_1, out_v_1 = new_key_values[1]
        out_k_2, out_v_2 = new_key_values[2]
        out_k_3, out_v_3 = new_key_values[3]
        out_k_4, out_v_4 = new_key_values[4]
        out_k_5, out_v_5 = new_key_values[5]
        out_k_6, out_v_6 = new_key_values[6]
        out_k_7, out_v_7 = new_key_values[7]

        return (
            logits,
            out_k_0, out_v_0,
            out_k_1, out_v_1,
            out_k_2, out_v_2,
            out_k_3, out_v_3,
            out_k_4, out_v_4,
            out_k_5, out_v_5,
            out_k_6, out_v_6,
            out_k_7, out_v_7,
        )


def export_decoder_external_cache(output_dir: Path, precision: str = "float16"):
    print("="*70)
    print("Cohere Decoder Export - External Cache")
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

    print("\n[2/5] Wrapping decoder...")
    wrapped = ExternalCacheDecoderWrapper(model, max_seq_len=108)
    wrapped.eval()
    print("   ✓ Wrapped")

    print("\n[3/5] Creating example inputs...")
    batch_size = 1
    num_layers = 8
    num_heads = 8
    head_dim = 128
    max_seq_len = 108

    # Example inputs
    example_input_id = torch.tensor([[13764]], dtype=torch.long)  # BOS
    example_encoder_hidden = torch.randn(1, 438, 1024)
    example_position_id = torch.tensor([[0]], dtype=torch.long)

    # Empty KV cache (all zeros initially)
    example_past_kvs = []
    for _ in range(num_layers):
        k = torch.zeros(batch_size, num_heads, max_seq_len, head_dim)
        v = torch.zeros(batch_size, num_heads, max_seq_len, head_dim)
        example_past_kvs.extend([k, v])

    print(f"   Input shapes:")
    print(f"     input_id: {example_input_id.shape}")
    print(f"     encoder_hidden: {example_encoder_hidden.shape}")
    print(f"     position_id: {example_position_id.shape}")
    print(f"     cache_k/v (per layer): {example_past_kvs[0].shape}")

    print("\n[4/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapped,
            (
                example_input_id,
                example_encoder_hidden,
                example_position_id,
                *example_past_kvs,
            ),
            check_trace=False,
        )

    # Test inference
    outputs = traced(
        example_input_id,
        example_encoder_hidden,
        example_position_id,
        *example_past_kvs,
    )
    logits = outputs[0]
    print(f"   Output: logits={logits.shape}, {len(outputs)-1} KV tensors")

    print(f"\n[5/5] Converting to CoreML...")

    # Build input specs
    inputs = [
        ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="encoder_hidden_states", shape=(1, 438, 1024), dtype=np.float32),
        ct.TensorType(name="position_id", shape=(1, 1), dtype=np.int32),
    ]

    # Add input KV cache
    for layer_idx in range(num_layers):
        inputs.append(
            ct.TensorType(
                name=f"cache_k_{layer_idx}",
                shape=(1, num_heads, max_seq_len, head_dim),
                dtype=np.float32,
            )
        )
        inputs.append(
            ct.TensorType(
                name=f"cache_v_{layer_idx}",
                shape=(1, num_heads, max_seq_len, head_dim),
                dtype=np.float32,
            )
        )

    # Build output specs
    output_specs = [ct.TensorType(name="logits")]

    # Add output KV cache (updated)
    for layer_idx in range(num_layers):
        output_specs.append(ct.TensorType(name=f"out_k_{layer_idx}"))
        output_specs.append(ct.TensorType(name=f"out_v_{layer_idx}"))

    # Convert - try Neural Network format first (for .mlmodelc support)
    try:
        print("   Attempting Neural Network format (iOS 14+)...")
        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=output_specs,
            minimum_deployment_target=ct.target.iOS14,
            convert_to="neuralnetwork",
        )

        # Apply FP16 quantization
        if precision == "float16":
            from coremltools.models.neural_network import quantization_utils
            mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

        format_type = "Neural Network"
        print("   ✓ Neural Network format successful")

    except Exception as e:
        print(f"   Neural Network failed: {e}")
        print("   Falling back to ML Program format (iOS 17+)...")

        compute_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=output_specs,
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=compute_precision,
        )

        format_type = "ML Program"
        print("   ✓ ML Program format used")

    output_path = output_dir / "cohere_decoder_external_cache.mlpackage"
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
    print(f"   ✓ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Format: {format_type}")

    print("\n" + "="*70)
    print("EXPORT COMPLETE - External Cache")
    print("="*70)
    print("\nKey features:")
    print("  - O(n) complexity (efficient)")
    print("  - Caller manages KV cache (Swift/external)")
    print(f"  - Format: {format_type}")
    print("  - KV cache: 16 tensors (8 layers × 2)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build-external-cache"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        export_decoder_external_cache(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
