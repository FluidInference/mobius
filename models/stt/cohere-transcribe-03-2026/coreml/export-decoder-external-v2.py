#!/usr/bin/env python3
"""Export Cohere decoder with EXTERNAL cache (Parakeet-style).

This mimics Parakeet's approach:
- Decoder takes cache as input
- Decoder returns updated cache as output
- Swift manages cache externally
- Enables Neural Network format → .mlmodelc
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq


class ExternalCacheDecoder(nn.Module):
    """Decoder that takes cache in and returns cache out (like Parakeet LSTM)."""

    def __init__(self, full_model, max_seq_len=108):
        super().__init__()
        self.decoder = full_model.transf_decoder
        self.log_softmax = full_model.log_softmax
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_id,  # [1, 1]
        encoder_hidden_states,  # [1, 438, 1024]
        position_id,  # [1, 1]
        # Past KV cache inputs (flattened to avoid tuple issues)
        past_k_0, past_v_0,
        past_k_1, past_v_1,
        past_k_2, past_v_2,
        past_k_3, past_v_3,
        past_k_4, past_v_4,
        past_k_5, past_v_5,
        past_k_6, past_v_6,
        past_k_7, past_v_7,
    ):
        """Single token decoding with external cache management."""

        # Organize past cache
        past_key_values = [
            (past_k_0, past_v_0),
            (past_k_1, past_v_1),
            (past_k_2, past_v_2),
            (past_k_3, past_v_3),
            (past_k_4, past_v_4),
            (past_k_5, past_v_5),
            (past_k_6, past_v_6),
            (past_k_7, past_v_7),
        ]

        # Current position
        positions = position_id

        # Cross attention mask (all encoder frames visible)
        cross_mask = torch.ones(1, encoder_hidden_states.shape[1], device=input_id.device)

        # Decoder forward
        decoder_outputs, new_key_values = self.decoder(
            input_ids=input_id,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=None,
            cross_attention_mask=cross_mask,
            past_key_values=past_key_values,
            cache_position=position_id.squeeze(0),
            kv_seq_len=None,
        )

        # Get logits
        logits = self.log_softmax(decoder_outputs[:, -1:, :]).squeeze(1)  # [1, vocab]

        # Return logits + ALL new cache tensors
        # DON'T reuse input names - use completely different output names
        return (
            logits,
            new_key_values[0][0], new_key_values[0][1],  # layer 0 k, v
            new_key_values[1][0], new_key_values[1][1],  # layer 1 k, v
            new_key_values[2][0], new_key_values[2][1],  # layer 2 k, v
            new_key_values[3][0], new_key_values[3][1],  # layer 3 k, v
            new_key_values[4][0], new_key_values[4][1],  # layer 4 k, v
            new_key_values[5][0], new_key_values[5][1],  # layer 5 k, v
            new_key_values[6][0], new_key_values[6][1],  # layer 6 k, v
            new_key_values[7][0], new_key_values[7][1],  # layer 7 k, v
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build-external-v2"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Cohere Decoder Export - External Cache (Parakeet-style)")
    print("="*70)

    # Load model
    print("\n[1/4] Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()

    # Wrap
    print("\n[2/4] Wrapping decoder...")
    wrapped = ExternalCacheDecoder(model, max_seq_len=108)
    wrapped.eval()

    # Create example inputs
    print("\n[3/4] Creating example inputs...")
    example_input_id = torch.tensor([[13764]], dtype=torch.long)
    example_encoder_hidden = torch.randn(1, 438, 1024)
    example_position_id = torch.tensor([[0]], dtype=torch.long)

    # Empty cache (8 layers × 2 tensors)
    example_caches = []
    for _ in range(8):
        k = torch.zeros(1, 8, 108, 128)  # [batch, heads, max_seq, head_dim]
        v = torch.zeros(1, 8, 108, 128)
        example_caches.extend([k, v])

    # Trace
    print("\n[4/4] Tracing and converting...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapped,
            (example_input_id, example_encoder_hidden, example_position_id, *example_caches),
            check_trace=False,
        )

    # Test
    outputs = traced(example_input_id, example_encoder_hidden, example_position_id, *example_caches)
    print(f"   Traced: logits={outputs[0].shape}, {len(outputs)-1} cache tensors")

    # Build CoreML inputs
    inputs = [
        ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="encoder_hidden_states", shape=(1, 438, 1024), dtype=np.float32),
        ct.TensorType(name="position_id", shape=(1, 1), dtype=np.int32),
    ]

    # Add cache inputs (past_*)
    for i in range(8):
        inputs.append(ct.TensorType(name=f"past_k_{i}", shape=(1, 8, 108, 128), dtype=np.float32))
        inputs.append(ct.TensorType(name=f"past_v_{i}", shape=(1, 8, 108, 128), dtype=np.float32))

    # Build outputs (logits + new_*)
    outputs_spec = [ct.TensorType(name="logits")]
    for i in range(8):
        outputs_spec.append(ct.TensorType(name=f"new_k_{i}"))
        outputs_spec.append(ct.TensorType(name=f"new_v_{i}"))

    # Try Neural Network format (for .mlmodelc compatibility)
    try:
        print("   Attempting Neural Network format (iOS 14+)...")
        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=outputs_spec,
            minimum_deployment_target=ct.target.iOS14,
            convert_to="neuralnetwork",
        )

        # Apply FP16 quantization
        if args.precision == "float16":
            from coremltools.models.neural_network import quantization_utils
            mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

        format_type = "Neural Network"
        print("   ✅ Neural Network format successful!")

    except Exception as e:
        print(f"   ❌ Neural Network failed: {e}")
        print("   Falling back to ML Program...")

        compute_precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32

        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=outputs_spec,
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=compute_precision,
        )

        format_type = "ML Program"

    # Save
    output_path = output_dir / "cohere_decoder_external.mlpackage"
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
    print(f"\n   ✓ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Format: {format_type}")

    print("\n" + "="*70)
    print(f"EXPORT COMPLETE - {format_type}")
    print("="*70)
    print("\nUsage in Swift (like Parakeet):")
    print("  Input: token + position + encoder_hidden + 16 cache tensors")
    print("  Output: logits + 16 updated cache tensors")
    print("  Swift manages: Cache lifecycle and passing")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
