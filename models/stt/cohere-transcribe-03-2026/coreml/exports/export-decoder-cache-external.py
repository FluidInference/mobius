#!/usr/bin/env python3
"""Export Cohere decoder with external cache management (Parakeet pattern).

This follows Parakeet TDT's approach:
- Cache is passed IN as model inputs
- Cache is returned OUT as model outputs
- Swift manages cache lifetime and passes it through each iteration
- Model updates cache by creating new tensors (not in-place mutation)

Key trick to avoid .item() tracing issue:
- Use attention_mask.shape[-1] to infer current position (not past_kv_len.item())
- attention_mask grows from [1,1,1,1] to [1,1,1,108] as we decode
- This is traceable because shape is dynamic input, not a constant

Usage:
    uv run export-decoder-cache-external.py --output-dir build
"""

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSpeechSeq2Seq

NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
VOCAB_SIZE = 16384
MAX_SEQ_LEN = 108


class CacheExternalCohereDecoder(nn.Module):
    """Cohere decoder with cache passed in/out (Parakeet TDT pattern).

    Inputs:
    - input_id, position_id: current token
    - encoder outputs: cross-attention context
    - attention_mask: [1, 1, 1, end_step] - size tells us current position!
    - k_cache_0..7, v_cache_0..7: current cache state

    Outputs:
    - logits: [1, 16384]
    - k_cache_0_out..7_out, v_cache_0_out..7_out: updated caches

    Swift manages cache arrays, passes them through each iteration.
    """

    def __init__(self, decoder_wrapper, lm_head):
        super().__init__()
        self.embedding = decoder_wrapper._embedding
        self.layers = decoder_wrapper._decoder.layers
        self.final_norm = decoder_wrapper._decoder.final_layer_norm
        self.lm_head = lm_head

    def forward(
        self,
        input_id: torch.Tensor,  # [1, 1]
        position_id: torch.Tensor,  # [1, 1]
        encoder_hidden_states: torch.Tensor,  # [1, 438, 1024]
        cross_attention_mask: torch.Tensor,  # [1, 1, 1, 438]
        attention_mask: torch.Tensor,  # [1, 1, 1, end_step] - VARIABLE SIZE
        # KV caches (16 inputs, 16 outputs)
        k_cache_0, v_cache_0, k_cache_1, v_cache_1,
        k_cache_2, v_cache_2, k_cache_3, v_cache_3,
        k_cache_4, v_cache_4, k_cache_5, v_cache_5,
        k_cache_6, v_cache_6, k_cache_7, v_cache_7,
    ):
        # Infer current position from attention_mask shape (Qwen3 trick)
        # No .item() needed - shape inference is traceable!
        end_step = attention_mask.shape[-1]  # Current sequence length (1, 2, 3, ...)
        past_kv_len = end_step - 1  # Positions already in cache (0, 1, 2, ...)

        k_caches_in = [k_cache_0, k_cache_1, k_cache_2, k_cache_3,
                        k_cache_4, k_cache_5, k_cache_6, k_cache_7]
        v_caches_in = [v_cache_0, v_cache_1, v_cache_2, v_cache_3,
                        v_cache_4, v_cache_5, v_cache_6, v_cache_7]

        # Get embedding
        hidden_states = self.embedding(input_id, position_id)

        # Output caches
        k_caches_out = []
        v_caches_out = []

        # Process layers
        for layer_idx, layer in enumerate(self.layers):
            k_cache = k_caches_in[layer_idx]
            v_cache = v_caches_in[layer_idx]

            # Self-attention
            residual = hidden_states
            hidden_states = layer.layer_norm_1(hidden_states)

            # Project Q, K, V
            query = layer.first_sub_layer.query_net(hidden_states)
            key = layer.first_sub_layer.key_net(hidden_states)
            value = layer.first_sub_layer.value_net(hidden_states)

            # Reshape
            query = layer.first_sub_layer._reshape(query)
            key = layer.first_sub_layer._reshape(key)
            value = layer.first_sub_layer._reshape(value)

            # Update cache using slicing (traceable because past_kv_len is computed from shape)
            # Clone to create new tensors (important for CoreML)
            k_cache_new = k_cache.clone()
            v_cache_new = v_cache.clone()

            # Write new K/V at position past_kv_len
            # This works because past_kv_len is derived from attention_mask.shape, not .item()
            k_cache_new[:, :, past_kv_len:end_step, :] = key
            v_cache_new[:, :, past_kv_len:end_step, :] = value

            # Read valid cache entries
            k_valid = k_cache_new[:, :, :end_step, :]
            v_valid = v_cache_new[:, :, :end_step, :]

            # Attention
            attn_output = F.scaled_dot_product_attention(
                query, k_valid, v_valid,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=layer.first_sub_layer.scale,
            )

            attn_output = (
                attn_output.transpose(1, 2).contiguous().view(1, 1, HIDDEN_SIZE)
            )
            attn_output = layer.first_sub_layer.out_projection(attn_output)
            hidden_states = residual + attn_output

            # Save updated caches
            k_caches_out.append(k_cache_new)
            v_caches_out.append(v_cache_new)

            # Cross-attention
            residual = hidden_states
            hidden_states = layer.layer_norm_2(hidden_states)
            cross_out = layer.second_sub_layer(
                hidden_states=hidden_states,
                context_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                past_key_values=None,
                cache_position=None,
                is_cross_attention=True,
                kv_seq_len=None,
            )
            hidden_states = residual + cross_out

            # FFN
            residual = hidden_states
            hidden_states = layer.layer_norm_3(hidden_states)
            hidden_states = residual + layer.third_sub_layer(hidden_states)

        # Final norm and logits
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states).squeeze(1)

        # Return logits + all updated caches
        return (
            logits,
            k_caches_out[0], v_caches_out[0],
            k_caches_out[1], v_caches_out[1],
            k_caches_out[2], v_caches_out[2],
            k_caches_out[3], v_caches_out[3],
            k_caches_out[4], v_caches_out[4],
            k_caches_out[5], v_caches_out[5],
            k_caches_out[6], v_caches_out[6],
            k_caches_out[7], v_caches_out[7],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--output-dir", default="build")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Cohere Decoder - Cache-External (Parakeet Pattern)")
    print("="*70)
    print()
    print("Key insight: Use attention_mask.shape[-1] to infer position")
    print("  • Avoids .item() tracing issue")
    print("  • attention_mask grows dynamically: [1,1,1,1] → [1,1,1,108]")
    print("  • Cache slicing uses derived indices, fully traceable")
    print()

    # Load
    print("[1/3] Loading model...")
    t0 = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    print(f"   ✓ {time.time()-t0:.1f}s")

    # Create wrapper
    print("\n[2/3] Creating wrapper...")
    decoder = CacheExternalCohereDecoder(
        model.transf_decoder,
        model.log_softmax.mlp.layer0
    )
    decoder.eval()

    # Trace
    print("\n[3/3] Tracing...")

    # Example: first token (step 0)
    input_id = torch.tensor([[4]], dtype=torch.long)
    position_id = torch.tensor([[0]], dtype=torch.long)
    encoder_hidden = torch.randn(1, 438, HIDDEN_SIZE)
    cross_mask = torch.ones(1, 1, 1, 438)
    # attention_mask: [1,1,1,1] for first token
    attention_mask = torch.zeros(1, 1, 1, 1)

    k_caches = [torch.zeros(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM) for _ in range(NUM_LAYERS)]
    v_caches = [torch.zeros(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM) for _ in range(NUM_LAYERS)]

    with torch.no_grad():
        traced = torch.jit.trace(decoder, (
            input_id, position_id, encoder_hidden, cross_mask, attention_mask,
            *k_caches, *v_caches
        ))

    print("   Converting to CoreML...")

    # Inputs with RangeDim for attention_mask
    attn_mask_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)
    inputs = [
        ct.TensorType("input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType("position_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType("encoder_hidden_states", shape=(1, 438, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("cross_attention_mask", shape=(1, 1, 1, 438), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, 1, attn_mask_dim), dtype=np.float32),
    ]

    for i in range(NUM_LAYERS):
        inputs.extend([
            ct.TensorType(f"k_cache_{i}", shape=(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32),
            ct.TensorType(f"v_cache_{i}", shape=(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32),
        ])

    # Outputs
    outputs = [ct.TensorType("logits", dtype=np.float32)]
    for i in range(NUM_LAYERS):
        outputs.extend([
            ct.TensorType(f"k_cache_{i}_out", dtype=np.float32),
            ct.TensorType(f"v_cache_{i}_out", dtype=np.float32),
        ])

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.macOS14,
    )

    mlmodel.author = "FluidInference"
    mlmodel.short_description = "Cohere Transcribe decoder (cache-external, Parakeet pattern)"

    output_path = output_dir / "cohere_decoder_cache_external.mlpackage"
    mlmodel.save(str(output_path))

    print(f"\n✅ Saved: {output_path}")

    import subprocess
    size_mb = subprocess.check_output(["du", "-sh", str(output_path)]).decode().split()[0]
    print(f"   Size: {size_mb}")

    print("\n" + "="*70)
    print("Next: Implement Swift CohereDecoderState + runDecoder()")
    print("="*70)
    print("\nSwift will:")
    print("  1. Maintain 16 MLMultiArray cache buffers")
    print("  2. Pass them to model each step")
    print("  3. Extract updated caches from output")
    print("  4. Update attention_mask size each step")


if __name__ == "__main__":
    main()
